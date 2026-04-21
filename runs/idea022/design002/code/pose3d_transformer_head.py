"""Transformer decoder head for 3D pose regression.

Takes a ViT feature map ``(B, C, H', W')`` and regresses:
  - ``(B, num_joints, 3)`` root-relative joint XYZ in metres
  - ``(B, 1)``             pelvis depth (forward distance) in metres
  - ``(B, 2)``             pelvis 2D position in crop pixels, normalised to [-1, 1]

Architecture (2-layer cascaded decoder with dynamic Gaussian reprojection bias
and auxiliary intermediate supervision)::

    feats[-1]  (B, C, H', W')
        → flatten to (B, H'*W', C)
        → input_proj: Linear(C, hidden_dim)
        → add 2D sinusoidal positional encoding
        → transformer decoder layer 0 (standard):
            self-attention over 70 joint queries
            cross-attention: queries attend to spatial tokens
            FFN with residual
        → intermediate joint/pelvis predictions + auxiliary body-joint loss
        → compute dynamic Gaussian cross-attention bias from layer-0 predictions
        → transformer decoder layer 1 (geometry-guided):
            self-attention over joint queries
            cross-attention with Gaussian bias injected into logits
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
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          recover_pelvis_3d,
                          project_joints_to_feat_grid)


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

    # Outer products -> (h, w, quarter)
    enc_y = grid_y.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)
    enc_x = grid_x.unsqueeze(-1) * omega.unsqueeze(0).unsqueeze(0)

    # Interleave sin/cos -> (h, w, half) each
    enc_y = torch.cat([enc_y.sin(), enc_y.cos()], dim=-1)
    enc_x = torch.cat([enc_x.sin(), enc_x.cos()], dim=-1)

    # Concatenate y and x -> (h, w, embed_dim)
    pos = torch.cat([enc_y, enc_x], dim=-1)
    return pos.reshape(1, h * w, embed_dim)


def _build_gaussian_bias(
    joint_feat_coords: torch.Tensor,
    feat_h: int,
    feat_w: int,
    sigma: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Build dynamic Gaussian cross-attention additive bias.

    Args:
        joint_feat_coords: (B, J, 2) -- (h_frac, w_frac) in feature grid units.
        feat_h: Feature grid height (40).
        feat_w: Feature grid width (24).
        sigma: (J,) per-joint bandwidth in grid cells. Must be clamped >= 0.5.
        gamma: (J,) per-joint amplitude.

    Returns:
        (B, J, feat_h * feat_w) additive bias for cross-attention logits.
    """
    B, J, _ = joint_feat_coords.shape
    device = joint_feat_coords.device
    dtype = joint_feat_coords.dtype

    grid_h = torch.arange(feat_h, device=device, dtype=dtype)  # (feat_h,)
    grid_w = torch.arange(feat_w, device=device, dtype=dtype)  # (feat_w,)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')     # each (feat_h, feat_w)
    grid = torch.stack([gh, gw], dim=-1).reshape(-1, 2)        # (feat_h*feat_w, 2)

    mu = joint_feat_coords.unsqueeze(-2)   # (B, J, 1, 2)
    g = grid.view(1, 1, -1, 2)             # (1, 1, H'W', 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)   # (B, J, H'W')

    # sigma: (J,) -> (1, J, 1); clamp to avoid near-zero bandwidth
    s = sigma.view(1, -1, 1).clamp(min=0.5)
    g_ = gamma.view(1, -1, 1)
    bias = g_ * torch.exp(-dist2 / (2.0 * s ** 2))  # (B, J, H'W')
    return bias


class _DecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn -> cross-attn -> FFN."""

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
                spatial_tokens: torch.Tensor,
                cross_attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            queries: ``(B, num_queries, embed_dim)``
            spatial_tokens: ``(B, num_spatial, embed_dim)``
            cross_attn_bias: optional ``(B, num_queries, num_spatial)`` additive
                attention bias injected into cross-attention logits.

        Returns:
            ``(B, num_queries, embed_dim)``
        """
        # Self-attention
        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        # Cross-attention
        q = self.norm2(queries)
        if cross_attn_bias is not None:
            B, J, _ = q.shape
            nheads = self.cross_attn.num_heads
            # Expand per-sample bias to (B*nheads, J, H'W') for batch_first=True MHA
            mask = cross_attn_bias.unsqueeze(1).expand(-1, nheads, -1, -1)  # (B, nheads, J, H'W')
            mask = mask.reshape(B * nheads, J, -1)                           # (B*nheads, J, H'W')
            q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                                  attn_mask=mask.to(q.dtype))[0]
        else:
            q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
        queries = queries + self.dropout2(q2)

        # FFN
        queries = queries + self.ffn(self.norm3(queries))

        return queries


@MODELS.register_module()
class Pose3dTransformerHead(BaseHead):
    """Transformer decoder head for 3D joint prediction and pelvis localisation.

    Supports a cascaded multi-layer decoder with optional dynamic Gaussian
    reprojection bias (geometry-guided cross-attention) and auxiliary
    intermediate supervision.

    Design B: 2-layer decoder with fixed Gaussian bias AND auxiliary body-joint
    loss (weight=0.4) on layer-0 output. The auxiliary loss uses normal autograd
    so gradients flow through layer-0 weights directly, bootstrapping the quality
    of the reprojection bias from early training epochs.

    Args:
        in_channels (int): Embedding dimension from the backbone (e.g. 1024).
        hidden_dim (int): Internal dimension for the decoder.
        num_joints (int): Number of output joints (70 for BEDLAM2 active set).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability in the decoder layer.
        num_decoder_layers (int): Number of stacked decoder layers.
        use_reproj_bias (bool): Whether to compute dynamic Gaussian reprojection bias.
        reproj_bias_sigma (float): Fixed Gaussian bandwidth in feature grid cells.
        reproj_bias_gamma (float): Fixed Gaussian amplitude.
        reproj_bias_learnable (bool): If True, sigma and gamma are learnable per-joint
            nn.Parameter tensors (Design C). If False, fixed scalars (Design A/B).
        aux_loss_weight (float): Weight for auxiliary body-joint loss on layer-0 output.
        feat_h (int): Feature grid height (40 for 640px input with stride 16).
        feat_w (int): Feature grid width (24 for 384px input with stride 16).
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
        num_decoder_layers: int = 1,
        use_reproj_bias: bool = False,
        reproj_bias_sigma: float = 4.0,
        reproj_bias_gamma: float = 2.0,
        reproj_bias_learnable: bool = False,
        aux_loss_weight: float = 0.0,
        feat_h: int = 40,
        feat_w: int = 24,
        loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                       beta=0.05, loss_weight=1.0),
        loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                      beta=0.05, loss_weight=1.0),
        loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
        loss_weight_depth: float = 1.0,
        loss_weight_uv: float = 1.0,
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

        # New parameters for cascaded decoder + reprojection bias
        self.num_decoder_layers = num_decoder_layers
        self.use_reproj_bias = use_reproj_bias
        self.reproj_bias_sigma = reproj_bias_sigma
        self.reproj_bias_gamma = reproj_bias_gamma
        self.reproj_bias_learnable = reproj_bias_learnable
        self.aux_loss_weight = aux_loss_weight
        self.feat_h = feat_h
        self.feat_w = feat_w

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        # Project backbone features to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Learnable joint query embeddings
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)

        # Transformer decoder (N layers)
        self.decoder_layers = nn.ModuleList([
            _DecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Learnable per-joint Gaussian bandwidth and amplitude (Design C only)
        if reproj_bias_learnable:
            self.bias_sigma = nn.Parameter(
                torch.ones(num_joints) * reproj_bias_sigma)  # (J,) init to 4.0
            self.bias_gamma = nn.Parameter(
                torch.ones(num_joints) * reproj_bias_gamma)  # (J,) init to 2.0

        # Output projections
        self.joints_out = nn.Linear(hidden_dim, 3)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out = nn.Linear(hidden_dim, 2)

        # Positional encoding buffer -- registered lazily on first forward
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

        # Broadcast joint queries to batch
        queries = self.joint_queries.weight.unsqueeze(0).expand(
            B, -1, -1)  # (B, num_joints, hidden_dim)

        # Decoder -- layer 0 always runs without bias
        decoded = self.decoder_layers[0](queries, spatial)

        # Subsequent layers: use reprojection bias if enabled and bias is available
        for layer_idx in range(1, self.num_decoder_layers):
            bias = getattr(self, '_reproj_bias', None)
            decoded = self.decoder_layers[layer_idx](decoded, spatial,
                                                      cross_attn_bias=bias)

        # Output projections
        joints = self.joints_out(decoded)  # (B, num_joints, 3)

        pelvis_token = decoded[:, 0, :]  # (B, hidden_dim)
        pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
        pelvis_uv = self.uv_out(pelvis_token)  # (B, 2)

        # Clear stored bias after use (set in loss(), not used at test time)
        if hasattr(self, '_reproj_bias'):
            self._reproj_bias = None

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
        # ── Compute reprojection bias from intermediate layer-0 predictions ──
        layer1_joints = None  # will be set if use_reproj_bias

        if self.use_reproj_bias and self.num_decoder_layers > 1:
            # Prepare spatial tokens (same as forward())
            feat = feats[-1]
            B_tmp, C_tmp, H_tmp, W_tmp = feat.shape
            spatial_tmp = feat.flatten(2).transpose(1, 2)
            spatial_tmp = self.input_proj(spatial_tmp)
            pos_enc_tmp = self._get_pos_enc(H_tmp, W_tmp, feat.device)
            spatial_tmp = spatial_tmp + pos_enc_tmp
            queries_tmp = self.joint_queries.weight.unsqueeze(0).expand(B_tmp, -1, -1)

            # Design B: run layer 0 with normal autograd (no torch.no_grad())
            # so that the auxiliary loss can backpropagate through layer-0 weights
            decoded_l1 = self.decoder_layers[0](queries_tmp, spatial_tmp)

            layer1_joints = self.joints_out(decoded_l1)        # (B, J, 3)
            layer1_depth  = self.depth_out(decoded_l1[:, 0])   # (B, 1)
            layer1_uv     = self.uv_out(decoded_l1[:, 0])      # (B, 2)

            # Recover absolute 3D positions and project to feature grid
            feat_coords_list = []
            for i in range(B_tmp):
                ds = batch_data_samples[i]
                K = np.asarray(ds.metainfo['K'], dtype=np.float32)
                img_shape = ds.metainfo.get('img_shape', (640, 384))
                crop_h, crop_w = int(img_shape[0]), int(img_shape[1])
                pelvis = recover_pelvis_3d(
                    layer1_depth[i:i+1], layer1_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
                abs_j = layer1_joints[i] + pelvis   # (J, 3) -- broadcast over joints
                fc = project_joints_to_feat_grid(
                    abs_j.unsqueeze(0), K, crop_h, crop_w,
                    self.feat_h, self.feat_w)  # (1, J, 2)
                feat_coords_list.append(fc[0])

            feat_coords = torch.stack(feat_coords_list)  # (B, J, 2)

            # Build fixed-parameter Gaussian bias (Design B: fixed sigma/gamma)
            sigma = torch.full(
                (self.num_joints,), self.reproj_bias_sigma,
                device=feat_coords.device, dtype=feat_coords.dtype)
            gamma = torch.full(
                (self.num_joints,), self.reproj_bias_gamma,
                device=feat_coords.device, dtype=feat_coords.dtype)
            self._reproj_bias = _build_gaussian_bias(
                feat_coords, self.feat_h, self.feat_w, sigma, gamma)  # (B, J, H'W')

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

        # Auxiliary intermediate loss on layer-0 output (body joints only, Design B/C)
        if self.aux_loss_weight > 0.0 and layer1_joints is not None:
            losses['loss/joints_aux/train'] = (
                self.aux_loss_weight * self.loss_joints_module(
                    layer1_joints[:, _BODY], gt_joints[:, _BODY]))

        # ── MPJPE (mm) -- stored as attributes for TrainMPJPEAveragingHook.
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

        At test time, _reproj_bias is never set, so the second decoder layer
        runs standard cross-attention (conservative safe fallback).

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
