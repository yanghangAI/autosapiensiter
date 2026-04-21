"""Transformer decoder head for 3D pose regression.

Takes a ViT feature map ``(B, C, H', W')`` and regresses:
  - ``(B, num_joints, 3)`` root-relative joint XYZ in metres
  - ``(B, 1)``             pelvis depth (forward distance) in metres
  - ``(B, 2)``             pelvis 2D position in crop pixels, normalised to [-1, 1]

idea014 — discretized pelvis depth classification head. Modes selected via
``depth_head_type``:
  - 'regression' (baseline)
  - 'classification' (Design 001): K-way softmax over fixed log-uniform bins in
    [depth_range_min, depth_range_max] + SORD soft-target cross-entropy.
  - 'classification_hybrid' (Design 002): as Design 001 + auxiliary SmoothL1 on
    soft-argmax expected depth (loss_weight = depth_aux_reg_weight).
  - 'classification_adaptive' (Design 003): per-sample bin widths à la AdaBins
    predicted by a second head ``depth_bins_head``; hybrid loss same as 002.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                  Predictions)
from mmpose.models.heads.base_head import BaseHead
from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs


def _build_2d_sincos_pos_enc(h: int, w: int, embed_dim: int) -> torch.Tensor:
    """Build 2D sine/cosine positional encoding (DETR-style)."""
    assert embed_dim % 4 == 0, f'embed_dim must be divisible by 4, got {embed_dim}'
    half = embed_dim // 2
    quarter = embed_dim // 4

    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (10000.0 ** omega)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij',
    )

    enc_y = grid_y.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    enc_x = grid_x.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)

    enc_y = torch.cat([enc_y.sin(), enc_y.cos()], dim=-1)
    enc_x = torch.cat([enc_x.sin(), enc_x.cos()], dim=-1)

    pos = torch.cat([enc_y, enc_x], dim=-1)
    return pos.reshape(1, h * w, embed_dim)


class _DecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn → cross-attn → FFN."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor,
                spatial_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        q = self.norm2(queries)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
        queries = queries + self.dropout2(q2)

        queries = queries + self.ffn(self.norm3(queries))

        return queries


@MODELS.register_module()
class Pose3dTransformerHead(BaseHead):
    """Transformer decoder head for 3D joint prediction and pelvis localisation."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_joints: int = 70,
        num_heads: int = 8,
        dropout: float = 0.1,
        loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                       beta=0.05, loss_weight=1.0),
        loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                      beta=0.05, loss_weight=1.0),
        loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
        loss_weight_depth: float = 1.0,
        loss_weight_uv: float = 1.0,
        depth_head_type: str = 'regression',
        num_depth_bins: int = 64,
        depth_range_min: float = 1.0,
        depth_range_max: float = 15.0,
        depth_soft_label_sigma: float = 1.5,
        depth_aux_reg_weight: float = 0.0,
        init_cfg: OptConfigType = None,
    ):
        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.loss_weight_depth = loss_weight_depth
        self.loss_weight_uv = loss_weight_uv

        # ── idea014: depth-head configuration ────────────────────────────────
        assert depth_head_type in ('regression', 'classification',
                                   'classification_hybrid',
                                   'classification_adaptive'), (
            f"Invalid depth_head_type='{depth_head_type}'. Must be one of: "
            f"'regression', 'classification', 'classification_hybrid', "
            f"'classification_adaptive'.")
        if depth_head_type != 'regression':
            assert num_depth_bins >= 4, (
                f"num_depth_bins must be >= 4, got {num_depth_bins}")
            assert 0.0 < depth_range_min < depth_range_max, (
                f"Require 0 < depth_range_min < depth_range_max, got "
                f"({depth_range_min}, {depth_range_max})")
            assert depth_soft_label_sigma > 0.0, (
                f"depth_soft_label_sigma must be > 0, got "
                f"{depth_soft_label_sigma}")
            assert depth_aux_reg_weight >= 0.0, (
                f"depth_aux_reg_weight must be >= 0, got "
                f"{depth_aux_reg_weight}")

        self.depth_head_type = depth_head_type
        self.num_depth_bins = num_depth_bins
        self.depth_range_min = depth_range_min
        self.depth_range_max = depth_range_max
        self.depth_soft_label_sigma = depth_soft_label_sigma
        self.depth_aux_reg_weight = depth_aux_reg_weight

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.joint_queries = nn.Embedding(num_joints, hidden_dim)

        self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)

        # Output projections
        self.joints_out = nn.Linear(hidden_dim, 3)
        if self.depth_head_type == 'regression':
            self.depth_out = nn.Linear(hidden_dim, 1)
        else:
            # Classification modes: emit logits over K bins.
            self.depth_out = nn.Linear(hidden_dim, self.num_depth_bins)
            log_min = math.log(self.depth_range_min)
            log_max = math.log(self.depth_range_max)
            log_centres = torch.linspace(log_min, log_max, self.num_depth_bins)
            self.register_buffer('log_bin_centres', log_centres,
                                 persistent=False)
            if self.depth_head_type == 'classification_adaptive':
                # AdaBins-style second head: per-sample bin widths.
                self.depth_bins_head = nn.Linear(hidden_dim,
                                                 self.num_depth_bins)
        self.uv_out = nn.Linear(hidden_dim, 2)

        self._pos_enc_hw: Tuple[int, int] | None = None

        self._init_head_weights()

    @property
    def default_init_cfg(self):
        return []

    def _init_head_weights(self) -> None:
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        modules_to_init = [self.joints_out, self.depth_out, self.uv_out]
        if self.depth_head_type == 'classification_adaptive':
            modules_to_init.append(self.depth_bins_head)
        for m in modules_to_init:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_pos_enc(self, h: int, w: int,
                     device: torch.device) -> torch.Tensor:
        if self._pos_enc_hw != (h, w):
            pos = _build_2d_sincos_pos_enc(h, w, self.hidden_dim)
            self.register_buffer('pos_enc', pos, persistent=False)
            self._pos_enc_hw = (h, w)
        return self.pos_enc.to(device)

    def forward(
        self, feats: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        feat = feats[-1]
        B, C, H, W = feat.shape

        spatial = feat.flatten(2).transpose(1, 2)
        spatial = self.input_proj(spatial)
        pos_enc = self._get_pos_enc(H, W, feat.device)
        spatial = spatial + pos_enc

        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

        decoded = self.decoder_layer(queries, spatial)

        joints = self.joints_out(decoded)

        pelvis_token = decoded[:, 0, :]

        if self.depth_head_type == 'regression':
            pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
            depth_logits = None
            depth_bin_centres = None
        else:
            depth_logits = self.depth_out(pelvis_token)  # (B, K)
            if self.depth_head_type == 'classification_adaptive':
                # ── AdaBins-style per-sample bin widths. ──
                width_logits = self.depth_bins_head(pelvis_token)  # (B, K)
                widths = torch.softmax(width_logits, dim=-1)  # (B, K), sum=1
                widths = widths * (
                    self.depth_range_max - self.depth_range_min)  # sum=R
                edges = torch.cumsum(widths, dim=-1)  # (B, K) in (0, R]
                zero_col = torch.zeros(widths.size(0), 1,
                                       device=widths.device,
                                       dtype=widths.dtype)
                edges = torch.cat([zero_col, edges], dim=-1)  # (B, K+1)
                edges = edges + self.depth_range_min  # (B, K+1) in [zmin, zmax]
                bin_centres = 0.5 * (edges[:, :-1] + edges[:, 1:])  # (B, K)
                depth_bin_centres = bin_centres  # (B, K)
            else:
                bin_centres = self.log_bin_centres.exp()  # (K,)
                depth_bin_centres = bin_centres.unsqueeze(0).expand(
                    depth_logits.size(0), -1)  # (B, K)
            probs = torch.softmax(depth_logits, dim=-1)  # (B, K)
            expected_depth = (probs * depth_bin_centres).sum(
                dim=-1, keepdim=True)  # (B, 1)
            pelvis_depth = expected_depth

        pelvis_uv = self.uv_out(pelvis_token)  # (B, 2)

        return {
            'joints': joints,
            'pelvis_depth': pelvis_depth,
            'pelvis_uv': pelvis_uv,
            'depth_logits': depth_logits,
            'depth_bin_centres': depth_bin_centres,
        }

    def loss(
        self,
        feats: Tuple[torch.Tensor, ...],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {},
    ) -> Dict[str, torch.Tensor]:
        pred = self.forward(feats)

        gt_joints = torch.cat([
            d.gt_instances.lifting_target
            for d in batch_data_samples
        ], dim=0)
        if gt_joints.dim() == 4:
            gt_joints = gt_joints.squeeze(1)
        gt_joints = gt_joints.to(pred['joints'].device)

        gt_depth = torch.stack([
            d.gt_instance_labels.pelvis_depth
            for d in batch_data_samples
        ]).to(pred['pelvis_depth'].device)
        if gt_depth.dim() == 1:
            gt_depth = gt_depth.unsqueeze(-1)

        gt_uv = torch.cat([
            d.gt_instance_labels.pelvis_uv
            for d in batch_data_samples
        ], dim=0).to(pred['pelvis_uv'].device)

        # Restrict joint loss to body joints only (indices 0-21)
        _BODY = list(range(0, 22))
        losses = dict()
        losses['loss/joints/train'] = self.loss_joints_module(
            pred['joints'][:, _BODY], gt_joints[:, _BODY])

        if self.depth_head_type == 'regression':
            losses['loss/depth/train'] = (
                self.loss_weight_depth
                * self.loss_depth_module(pred['pelvis_depth'], gt_depth))
        else:
            # Classification depth loss: SORD soft-target cross-entropy.
            bin_centres = pred['depth_bin_centres']  # (B, K)
            K = bin_centres.size(-1)

            log_bin_centres_per_sample = bin_centres.clamp(
                min=self.depth_range_min * 1e-3).log()  # (B, K)

            if self.depth_head_type == 'classification_adaptive':
                log_diffs = (log_bin_centres_per_sample[:, 1:]
                             - log_bin_centres_per_sample[:, :-1]).abs()
                bin_width_log_per_sample = log_diffs.median(
                    dim=-1, keepdim=True).values  # (B, 1)
                sigma_log = (self.depth_soft_label_sigma
                             * bin_width_log_per_sample)  # (B, 1)
            else:
                log_min = math.log(self.depth_range_min)
                log_max = math.log(self.depth_range_max)
                bin_width_log = (log_max - log_min) / max(K - 1, 1)
                sigma_log = torch.full(
                    (bin_centres.size(0), 1),
                    self.depth_soft_label_sigma * bin_width_log,
                    device=bin_centres.device,
                    dtype=bin_centres.dtype)  # (B, 1)

            z_gt = gt_depth.clamp(min=self.depth_range_min,
                                  max=self.depth_range_max)  # (B, 1)
            log_z_gt = z_gt.log()  # (B, 1)

            log_diff = log_bin_centres_per_sample - log_z_gt  # (B, K)
            target_logits = -(log_diff ** 2) / (2.0 * sigma_log ** 2)
            target = torch.softmax(target_logits, dim=-1)  # (B, K)
            # Detach target so gradients flow only through log_probs
            # (critical for adaptive mode; no-op for fixed mode).
            target = target.detach()

            log_probs = torch.log_softmax(pred['depth_logits'], dim=-1)
            ce_per_sample = -(target * log_probs).sum(dim=-1)  # (B,)
            L_depth_ce = ce_per_sample.mean()
            losses['loss/depth/train'] = self.loss_weight_depth * L_depth_ce

            if self.depth_aux_reg_weight > 0.0:
                L_depth_reg = F.smooth_l1_loss(
                    pred['pelvis_depth'],
                    gt_depth.to(pred['pelvis_depth'].device),
                    reduction='mean', beta=0.05)
                losses['loss/depth_reg/train'] = (
                    self.depth_aux_reg_weight * L_depth_reg)

        losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
            pred['pelvis_uv'], gt_uv)

        # ── MPJPE (mm) — stored as attributes for TrainMPJPEAveragingHook.
        with torch.no_grad():
            self._train_mpjpe = (
                (pred['joints'][:, _BODY] - gt_joints[:, _BODY]).norm(dim=-1).mean() * 1000.0)
            self._train_mpjpe_abs = _compute_mpjpe_abs(
                pred['joints'], gt_joints,
                pred['pelvis_depth'], gt_depth,
                pred['pelvis_uv'], gt_uv,
                batch_data_samples)

        return losses, pred

    def predict(
        self,
        feats: Tuple[torch.Tensor, ...],
        batch_data_samples: OptSampleList,
        test_cfg: ConfigType = {},
    ) -> Predictions:
        pred = self.forward(feats)
        B = pred['joints'].size(0)

        preds: List[InstanceData] = []
        for i in range(B):
            inst = InstanceData()
            inst.keypoints = pred['joints'][i:i+1].detach().cpu().numpy()
            inst.keypoint_scores = torch.ones(
                1, self.num_joints, dtype=torch.float32
            ).numpy()
            inst.pelvis_depth = pred['pelvis_depth'][i].detach().cpu().numpy()
            inst.pelvis_uv = pred['pelvis_uv'][i:i+1].detach().cpu().numpy()
            preds.append(inst)

        return preds
