"""Transformer decoder head for 3D pose regression.

Takes a ViT feature map ``(B, C, H', W')`` and regresses:
  - ``(B, num_joints, 3)`` root-relative joint XYZ in metres
  - ``(B, 1)``             pelvis depth (forward distance) in metres
  - ``(B, 2)``             pelvis 2D position in crop pixels, normalised to [-1, 1]

Architecture::

    feats[-1]  (B, C, H', W')
        → flatten to (B, H'*W', C)
        → input_proj: Linear(C, hidden_dim)
        → add 2D sinusoidal positional encoding
        → transformer decoder (1 layer):
            self-attention over 70 joint queries
            cross-attention: queries attend to spatial tokens
            FFN with residual
        → Linear(hidden_dim, 3) per token   → joints    (B, num_joints, 3)
        → Linear(hidden_dim, 1) on token 0  → pelvis_depth (B, 1)
        → Linear(hidden_dim, 2) on token 0  → pelvis_uv    (B, 2)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                  Predictions)
from mmpose.models.heads.base_head import BaseHead
from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs
from pelvis_utils import recover_pelvis_3d, project_joints_to_grid_coords


def _build_gaussian_heatmap_target(
    joint_grid_coords: torch.Tensor,
    feat_h: int,
    feat_w: int,
    sigma: float,
) -> torch.Tensor:
    """Build soft Gaussian heatmap target over the spatial grid.

    Args:
        joint_grid_coords: (J, 2) joint positions in feature grid units (h, w).
        feat_h: Feature grid height.
        feat_w: Feature grid width.
        sigma: Gaussian standard deviation in grid cells.

    Returns:
        (J, feat_h * feat_w) float tensor, normalised to sum to 1.
    """
    J = joint_grid_coords.shape[0]
    device = joint_grid_coords.device
    gh = torch.arange(feat_h, device=device, dtype=torch.float32)
    gw = torch.arange(feat_w, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(gh, gw, indexing='ij')  # (H', W')
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)  # (H'W', 2)

    mu = joint_grid_coords.unsqueeze(1)   # (J, 1, 2)
    g = grid.unsqueeze(0)                  # (1, H'W', 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)   # (J, H'W')
    heatmap = torch.exp(-dist2 / (2.0 * sigma ** 2))
    heatmap = heatmap / (heatmap.sum(dim=-1, keepdim=True).clamp(min=1e-6))
    return heatmap  # (J, H'W')


def _build_2d_sincos_pos_enc(h: int, w: int, embed_dim: int) -> torch.Tensor:
    """Build 2D sine/cosine positional encoding (DETR-style).

    Args:
        h: Height of the feature grid.
        w: Width of the feature grid.
        embed_dim: Embedding dimension (must be divisible by 4).

    Returns:
        Tensor of shape ``(1, h*w, embed_dim)``.
    """
    assert embed_dim % 4 == 0, f'embed_dim must be divisible by 4, got {embed_dim}'
    half = embed_dim // 2
    quarter = embed_dim // 4

    # Temperature for frequency bands
    omega = torch.arange(quarter, dtype=torch.float32) / quarter
    omega = 1.0 / (10000.0 ** omega)  # (quarter,)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing='ij',
    )  # each (h, w)

    # Outer products → (h, w, quarter)
    enc_y = grid_y.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    enc_x = grid_x.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)

    # Interleave sin/cos → (h, w, half) each
    enc_y = torch.cat([enc_y.sin(), enc_y.cos()], dim=-1)
    enc_x = torch.cat([enc_x.sin(), enc_x.cos()], dim=-1)

    # Concatenate y and x → (h, w, embed_dim)
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
        """
        Args:
            queries: ``(B, num_queries, embed_dim)``
            spatial_tokens: ``(B, num_spatial, embed_dim)``

        Returns:
            ``(B, num_queries, embed_dim)``
        """
        # Self-attention
        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        # Cross-attention
        q = self.norm2(queries)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
        queries = queries + self.dropout2(q2)

        # FFN
        queries = queries + self.ffn(self.norm3(queries))

        return queries


@MODELS.register_module()
class Pose3dTransformerHead(BaseHead):
    """Transformer decoder head for 3D joint prediction and pelvis localisation.

    Args:
        in_channels (int): Embedding dimension from the backbone (e.g. 1024).
        hidden_dim (int): Internal dimension for the decoder. If smaller than
            in_channels, an input_proj Linear projects down to save memory.
        num_joints (int): Number of output joints (70 for BEDLAM2 active set).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability in the decoder layer.
        loss_joints (ConfigType): Config for the joint coordinate loss.
        loss_depth (ConfigType): Config for the pelvis depth loss.
        loss_uv (ConfigType): Config for the pelvis 2D position loss.
        loss_weight_depth (float): Weight for the depth loss term.
        loss_weight_uv (float): Weight for the UV loss term.
        init_cfg: Standard MMEngine init config.
    """

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
        use_heatmap_init: bool = False,
        heatmap_loss_weight: float = 0.1,
        heatmap_target: str = 'onehot',
        heatmap_sigma: float = 2.0,
        heatmap_temperature: float = 1.0,
        heatmap_learnable_temp: bool = False,
        feat_h: int = 40,
        feat_w: int = 24,
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

        # Heatmap init settings
        self.use_heatmap_init = use_heatmap_init
        self.heatmap_loss_weight = heatmap_loss_weight
        self.heatmap_target = heatmap_target
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_temperature = heatmap_temperature
        self.heatmap_learnable_temp = heatmap_learnable_temp
        self.feat_h = feat_h
        self.feat_w = feat_w

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        # Project backbone features to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Learnable joint query embeddings
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)

        # Transformer decoder (1 layer)
        self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)

        # Output projections
        self.joints_out = nn.Linear(hidden_dim, 3)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out = nn.Linear(hidden_dim, 2)

        # Heatmap projection module (zero-initialised so warm-start = global avg pool)
        if self.use_heatmap_init:
            self.heatmap_proj = nn.Linear(hidden_dim, 22)
            nn.init.zeros_(self.heatmap_proj.weight)
            nn.init.zeros_(self.heatmap_proj.bias)
            if self.heatmap_learnable_temp:
                self.heatmap_temp = nn.Parameter(torch.ones(22))

        # Side-channel for heatmap logits (set in forward, read in loss)
        self._heatmap_logits = None

        # Positional encoding buffer — registered lazily on first forward
        self._pos_enc_hw: Tuple[int, int] | None = None

        self._init_head_weights()

    @property
    def default_init_cfg(self):
        return []

    def _init_head_weights(self) -> None:
        # Query embeddings
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        # Output projections
        for m in [self.joints_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _get_pos_enc(self, h: int, w: int,
                     device: torch.device) -> torch.Tensor:
        """Get or recompute 2D positional encoding buffer."""
        if self._pos_enc_hw != (h, w):
            pos = _build_2d_sincos_pos_enc(h, w, self.hidden_dim)
            self.register_buffer('pos_enc', pos, persistent=False)
            self._pos_enc_hw = (h, w)
        return self.pos_enc.to(device)

    def forward(
        self, feats: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            feats: Tuple of feature tensors from the backbone.
                   ``feats[-1]`` is ``(B, C, H', W')``.

        Returns:
            Dict with keys:
                ``joints``:       ``(B, num_joints, 3)`` root-relative metres
                ``pelvis_depth``: ``(B, 1)`` pelvis forward distance metres
                ``pelvis_uv``:    ``(B, 2)`` pelvis (u, v) normalised to [-1, 1]
        """
        feat = feats[-1]  # (B, C, H, W)
        B, C, H, W = feat.shape

        # Flatten spatial dims, project to hidden_dim, add positional encoding
        spatial = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        spatial = self.input_proj(spatial)          # (B, H*W, hidden_dim)
        pos_enc = self._get_pos_enc(H, W, feat.device)
        spatial = spatial + pos_enc

        # Heatmap-guided query warm-start
        if self.use_heatmap_init:
            # (B, H'W', 22) — one score per spatial token per body joint
            heatmap_logits = self.heatmap_proj(spatial)

            # Per-joint temperature: shape (1, 22, 1) so it broadcasts over H'W'
            if self.heatmap_learnable_temp:
                temp = F.softplus(self.heatmap_temp).view(1, 22, 1)  # (1, 22, 1)
            else:
                temp = self.heatmap_temperature

            # Soft attention over spatial tokens: (B, 22, H'W')
            attn_weights = F.softmax(heatmap_logits.transpose(1, 2) / temp, dim=-1)

            # Soft pooling: (B, 22, hidden_dim)
            pooled_features = torch.bmm(attn_weights, spatial)

            # Zero-pad to full num_joints and add to static joint query embeddings
            pad = torch.zeros(B, self.num_joints - 22, self.hidden_dim,
                              device=spatial.device, dtype=spatial.dtype)
            delta = torch.cat([pooled_features, pad], dim=1)  # (B, num_joints, hidden_dim)
            queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1) + delta

            # Store for loss() to read
            self._heatmap_logits = heatmap_logits
        else:
            # Broadcast joint queries to batch
            queries = self.joint_queries.weight.unsqueeze(0).expand(
                B, -1, -1)  # (B, num_joints, hidden_dim)
            self._heatmap_logits = None

        # Decoder
        decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)

        # Output projections
        joints = self.joints_out(decoded)  # (B, num_joints, 3)

        pelvis_token = decoded[:, 0, :]  # (B, hidden_dim)
        pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
        pelvis_uv = self.uv_out(pelvis_token)  # (B, 2)

        return {
            'joints': joints,
            'pelvis_depth': pelvis_depth,
            'pelvis_uv': pelvis_uv,
        }

    def loss(
        self,
        feats: Tuple[torch.Tensor, ...],
        batch_data_samples: OptSampleList,
        train_cfg: ConfigType = {},
    ) -> Dict[str, torch.Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Returns:
            Tuple of (losses dict, predictions dict).
        """
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
        losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
            pred['pelvis_depth'], gt_depth)
        losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
            pred['pelvis_uv'], gt_uv)

        # ── Heatmap loss ─────────────────────────────────────────────────────
        if self.use_heatmap_init and self._heatmap_logits is not None:
            heatmap_loss = 0.0
            B_hm = len(batch_data_samples)
            for i in range(B_hm):
                ds = batch_data_samples[i]
                K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
                img_shape = ds.metainfo.get('img_shape', (640, 384))
                crop_h, crop_w = int(img_shape[0]), int(img_shape[1])

                # GT absolute joints for body joints (0-21)
                gt_pelvis_3d = recover_pelvis_3d(
                    gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
                gt_abs_joints = gt_joints[i, :22] + gt_pelvis_3d          # (22, 3)

                # Project to feature grid coordinates
                grid_coords = project_joints_to_grid_coords(
                    gt_abs_joints, K, crop_h, crop_w,
                    self.feat_h, self.feat_w)  # (22, 2)

                if self.heatmap_target == 'onehot':
                    # Hard one-hot: cross-entropy with nearest grid cell
                    h_idx = grid_coords[:, 0].long().clamp(0, self.feat_h - 1)
                    w_idx = grid_coords[:, 1].long().clamp(0, self.feat_w - 1)
                    target_idx = h_idx * self.feat_w + w_idx  # (22,)
                    logits_i = self._heatmap_logits[i].T       # (22, H'W')
                    heatmap_loss = heatmap_loss + F.cross_entropy(logits_i, target_idx)
                else:
                    # Soft Gaussian target: KL divergence
                    gt_hm = _build_gaussian_heatmap_target(
                        grid_coords, self.feat_h, self.feat_w,
                        self.heatmap_sigma)  # (22, H'W')
                    logits_i = self._heatmap_logits[i].T        # (22, H'W')
                    log_probs = F.log_softmax(logits_i, dim=-1)  # (22, H'W')
                    heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum()

            losses['loss/heatmap/train'] = (
                self.heatmap_loss_weight * heatmap_loss / B_hm)
            self._heatmap_logits = None  # clear stale reference

        # ── MPJPE (mm) — stored as attributes for TrainMPJPEAveragingHook.
        # Not included in the losses dict to avoid MMEngine auto-logging
        # noisy per-batch scalars (the hook writes epoch-averaged values).
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
        """Predict results from feature maps.

        Returns:
            List of InstanceData, one per sample.
        """
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
