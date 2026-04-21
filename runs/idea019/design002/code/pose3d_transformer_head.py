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
        → transformer decoder (1+ layers):
            self-attention over joint queries
            cross-attention: queries attend to spatial tokens
              (deformable: per-query K_s=8 bilinear-sampled points, or
               standard: full 960-token dense cross-attention)
            FFN with residual
        → Linear(hidden_dim, 3) per token   → joints    (B, num_joints, 3)
        → Linear(hidden_dim, 1) on token 0  → pelvis_depth (B, 1)
        → Linear(hidden_dim, 2) on token 0  → pelvis_uv    (B, 2)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import (ConfigType, OptConfigType, OptSampleList,
                                  Predictions)
from mmpose.models.heads.base_head import BaseHead
from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs


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


class _DeformableDecoderLayer(nn.Module):
    """Transformer decoder layer with per-query deformable sparse cross-attention.

    Self-attention is unchanged (all queries attend to each other via MHA).
    Cross-attention is replaced: each query independently predicts K_s=num_points
    2D offset vectors from a learnable reference point, bilinearly samples
    num_points spatial features from the 2D feature grid, then performs a
    learned weighted sum over the num_points sampled features.

    Args:
        embed_dim: Embedding/hidden dimension (256).
        num_heads: Number of attention heads for self-attention (8).
        dropout: Dropout probability (0.1).
        num_points: Number of sampling points per query K_s (8).
        deform_hidden_dim: Bottleneck width in offset MLP (64).
        num_queries: Number of joint queries (must match ref_points shape) (70).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_points: int = 8,
        deform_hidden_dim: int = 64,
        num_queries: int = 70,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_points = num_points
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Self-attention (unchanged from baseline _DecoderLayer)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Offset network: shared across all queries
        # Input: (B, num_queries, embed_dim)
        # Output: (B, num_queries, num_points * 2) — 2D offsets per sampling point
        self.offset_net = nn.Sequential(
            nn.Linear(embed_dim, deform_hidden_dim),
            nn.GELU(),
            nn.Linear(deform_hidden_dim, num_points * 2),
        )

        # Learnable reference points: one (u, v) centre per query, in [0,1]^2
        # Initialised to (0.5, 0.5) = grid centre for all queries
        self.ref_points = nn.Parameter(torch.full((num_queries, 2), 0.5))

        # Lightweight cross-attention over K_s sampled features:
        # per-query scalar attention weights (not full MHA — single-head linear attention)
        self.attn_weight_net = nn.Linear(embed_dim, num_points)

        # Value and output projections for the sparse cross-attention
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN (identical to baseline)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms and dropouts
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sample_spatial_features(
        self,
        queries: torch.Tensor,       # (B, num_queries, embed_dim)
        spatial_grid: torch.Tensor,  # (B, embed_dim, H', W')
    ) -> torch.Tensor:
        """Sample K_s=num_points features per query via bilinear interpolation.

        Returns:
            Tensor of shape (B, num_queries, num_points, embed_dim).
        """
        B, Nq, D = queries.shape
        _, _, H, W = spatial_grid.shape

        # Predict per-query offset vectors from query features
        offsets = self.offset_net(queries)                       # (B, Nq, num_points*2)
        offsets = offsets.view(B, Nq, self.num_points, 2)        # (B, Nq, K_s, 2)

        # Reference points broadcast: (1, Nq, 1, 2)
        ref = self.ref_points.unsqueeze(0).unsqueeze(2)          # (1, Nq, 1, 2)

        # Sampling locations in [0,1]^2; offsets scaled by 0.1 (±10% grid)
        sample_locs = ref + offsets * 0.1                        # (B, Nq, K_s, 2)
        sample_locs = sample_locs.clamp(0.0, 1.0)

        # Convert to [-1,1] for F.grid_sample (expects x=W-axis, y=H-axis)
        # sample_locs[:,:,:,0] = u (width direction → x for grid_sample)
        # sample_locs[:,:,:,1] = v (height direction → y for grid_sample)
        grid = sample_locs * 2.0 - 1.0                          # (B, Nq, K_s, 2)

        # Reshape for grid_sample: needs (B, H_out, W_out, 2)
        # Treat Nq*K_s points as a 1-row grid of width Nq*K_s
        grid = grid.view(B, Nq * self.num_points, 1, 2)          # (B, Nq*K_s, 1, 2)

        # Cast grid to match spatial_grid dtype for AMP compatibility
        grid = grid.to(spatial_grid.dtype)

        sampled = torch.nn.functional.grid_sample(
            spatial_grid, grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )  # (B, embed_dim, Nq*K_s, 1)

        sampled = sampled.squeeze(-1).transpose(1, 2)            # (B, Nq*K_s, embed_dim)
        sampled = sampled.view(B, Nq, self.num_points, D)        # (B, Nq, K_s, embed_dim)
        return sampled

    def forward(
        self,
        queries: torch.Tensor,       # (B, num_queries, embed_dim)
        spatial_grid: torch.Tensor,  # (B, embed_dim, H', W') — NOT flattened
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, embed_dim)
            spatial_grid: (B, embed_dim, H', W') — 2D feature map after input_proj + pos_enc

        Returns:
            (B, num_queries, embed_dim)
        """
        # 1. Self-attention (pre-norm, unchanged from baseline)
        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        # 2. Deformable cross-attention
        q = self.norm2(queries)
        sampled = self._sample_spatial_features(q, spatial_grid)
        # sampled: (B, Nq, K_s, embed_dim)

        # Per-query attention weights over K_s points (scalar, linear attention)
        attn_w = self.attn_weight_net(q)                         # (B, Nq, K_s)
        attn_w = attn_w.softmax(dim=-1).unsqueeze(-1)            # (B, Nq, K_s, 1)

        # Project sampled values and aggregate
        values = self.value_proj(sampled)                        # (B, Nq, K_s, embed_dim)
        attended = (attn_w * values).sum(dim=2)                  # (B, Nq, embed_dim)
        attended = self.out_proj(attended)                       # (B, Nq, embed_dim)
        queries = queries + self.dropout2(attended)

        # 3. FFN (pre-norm)
        queries = queries + self.ffn(self.norm3(queries))
        return queries


@MODELS.register_module()
class Pose3dTransformerHead(BaseHead):
    """Transformer decoder head for 3D joint prediction and pelvis localisation.

    Supports both standard dense cross-attention (baseline) and per-query
    deformable sparse cross-attention (idea019 designs).

    Args:
        in_channels (int): Embedding dimension from the backbone (e.g. 1024).
        hidden_dim (int): Internal dimension for the decoder.
        num_joints (int): Number of output joints (70 for BEDLAM2 active set).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability in the decoder layer.
        deform_num_points (int): Number of deformable sampling points per query.
            0 = disabled (use standard cross-attention).
        deform_hidden_dim (int): Bottleneck width in deformable offset MLP.
        num_body_queries (int): Number of body joint queries for deformable decoder.
            70 = all joints (Design 001); 22 = body-only (Design 002/003).
        num_decoder_layers (int): Number of stacked decoder layers.
        hand_aux_loss_weight (float): Weight for auxiliary hand loss (0.0 = disabled).
        aux_body_loss_weight (float): Weight for intermediate body supervision loss.
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
        deform_num_points: int = 0,
        deform_hidden_dim: int = 64,
        num_body_queries: int = 70,
        num_decoder_layers: int = 1,
        hand_aux_loss_weight: float = 0.0,
        aux_body_loss_weight: float = 0.0,
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
        self.use_deform = deform_num_points > 0
        self.num_body_queries = num_body_queries
        self.hand_aux_loss_weight = hand_aux_loss_weight
        self.aux_body_loss_weight = aux_body_loss_weight
        self.num_decoder_layers = num_decoder_layers

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        # Project backbone features to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Learnable joint query embeddings — num_body_queries for deformable designs
        self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)

        # Build decoder layers
        if self.use_deform:
            self.decoder_layers = nn.ModuleList([
                _DeformableDecoderLayer(
                    hidden_dim, num_heads, dropout,
                    num_points=deform_num_points,
                    deform_hidden_dim=deform_hidden_dim,
                    num_queries=num_body_queries,
                )
                for _ in range(num_decoder_layers)
            ])
        else:
            self.decoder_layers = nn.ModuleList([
                _DecoderLayer(hidden_dim, num_heads, dropout)
                for _ in range(num_decoder_layers)
            ])
        # Note: no separate self.decoder_layer alias to avoid duplicate params

        # Hand projection (Design 002/003: body-only decoder recovers hand joints)
        self.has_hand_proj = (num_body_queries < num_joints)
        if self.has_hand_proj:
            self.hand_proj = nn.Linear(
                num_body_queries * hidden_dim,          # 22 * 256 = 5632
                (num_joints - num_body_queries) * 3,    # 48 * 3 = 144
            )

        # Intermediate supervision heads (Design 003: aux supervision after layer 0)
        self.has_intermediate_sup = (num_decoder_layers > 1 and aux_body_loss_weight > 0.0)
        if self.has_intermediate_sup:
            self.intermediate_joints_out = nn.ModuleList([
                nn.Linear(hidden_dim, 3)
                for _ in range(num_decoder_layers - 1)
            ])

        # Output projections
        self.joints_out = nn.Linear(hidden_dim, 3)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out = nn.Linear(hidden_dim, 2)

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

        # Deformable offset network near-zero init
        if self.use_deform:
            for layer_mod in self.decoder_layers:
                # Near-zero init for offset net output: all offsets ≈ 0 at start
                nn.init.zeros_(layer_mod.offset_net[-1].weight)
                nn.init.zeros_(layer_mod.offset_net[-1].bias)
                # Near-zero init for attention weight net: uniform weights at start
                nn.init.zeros_(layer_mod.attn_weight_net.weight)
                nn.init.zeros_(layer_mod.attn_weight_net.bias)
                # Standard small-std init for value/output projections
                nn.init.trunc_normal_(layer_mod.value_proj.weight, std=0.02)
                if layer_mod.value_proj.bias is not None:
                    nn.init.zeros_(layer_mod.value_proj.bias)
                nn.init.trunc_normal_(layer_mod.out_proj.weight, std=0.02)
                if layer_mod.out_proj.bias is not None:
                    nn.init.zeros_(layer_mod.out_proj.bias)

        # Intermediate supervision head init
        if self.has_intermediate_sup:
            for head in self.intermediate_joints_out:
                nn.init.trunc_normal_(head.weight, std=0.02)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

        # Hand projection init
        if self.has_hand_proj:
            nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
            nn.init.zeros_(self.hand_proj.bias)

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

        # Project backbone features to hidden_dim, add positional encoding
        spatial_flat = feat.flatten(2).transpose(1, 2)   # (B, H*W, C)
        spatial_proj = self.input_proj(spatial_flat)      # (B, H*W, hidden_dim)
        pos_enc = self._get_pos_enc(H, W, feat.device)
        spatial_proj = spatial_proj + pos_enc             # (B, H*W, hidden_dim)

        # Broadcast joint queries to batch
        queries = self.joint_queries.weight.unsqueeze(0).expand(
            B, -1, -1)  # (B, num_body_queries, hidden_dim)

        if self.use_deform:
            # Reshape projected features back to 2D grid for grid_sample
            spatial_grid = spatial_proj.transpose(1, 2).view(
                B, self.hidden_dim, H, W)                 # (B, hidden_dim, H', W')

            intermediate_decoded = []
            for i, layer in enumerate(self.decoder_layers):
                queries = layer(queries, spatial_grid)
                if i < len(self.decoder_layers) - 1:
                    # Collect intermediate outputs for supervision (all but last layer)
                    intermediate_decoded.append(queries)
            decoded = queries                              # (B, num_body_queries, hidden_dim)
            self._intermediate_decoded = intermediate_decoded
        else:
            # Standard baseline path (non-deformable)
            for layer in self.decoder_layers:
                queries = layer(queries, spatial_proj)
            decoded = queries                              # (B, num_joints, hidden_dim)
            self._intermediate_decoded = []

        # Body joints from decoder output
        body_joints = self.joints_out(decoded)            # (B, num_body_queries, 3)

        # Hand recovery via linear projection (Design 002/003: num_body_queries=22)
        if self.has_hand_proj:
            body_flat = decoded.reshape(
                B, self.num_body_queries * self.hidden_dim)     # (B, 22*256=5632)
            num_hand = self.num_joints - self.num_body_queries  # 48
            hand_joints = self.hand_proj(body_flat).reshape(
                B, num_hand, 3)                                 # (B, 48, 3)
            joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)
        else:
            joints = body_joints                          # (B, 70, 3) when num_body_queries=70

        pelvis_token = decoded[:, 0, :]                   # (B, hidden_dim)
        pelvis_depth = self.depth_out(pelvis_token)       # (B, 1)
        pelvis_uv = self.uv_out(pelvis_token)             # (B, 2)

        return {
            'joints': joints,          # always (B, 70, 3)
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

        # Intermediate layer body joint supervision (Design 003: aux_body_loss_weight=0.4)
        if self.has_intermediate_sup and hasattr(self, '_intermediate_decoded'):
            for idx, inter_decoded in enumerate(self._intermediate_decoded):
                inter_body_joints = self.intermediate_joints_out[idx](inter_decoded)  # (B, 22, 3)
                losses[f'loss/joints_inter{idx}/train'] = (
                    self.aux_body_loss_weight * self.loss_joints_module(
                        inter_body_joints[:, _BODY], gt_joints[:, _BODY]))

        # Auxiliary hand loss (Design 002/003: hand_aux_loss_weight=0.1)
        if self.hand_aux_loss_weight > 0.0 and self.has_hand_proj:
            _HAND = list(range(self.num_body_queries, self.num_joints))  # range(22, 70)
            losses['loss/hand_aux/train'] = self.hand_aux_loss_weight * self.loss_joints_module(
                pred['joints'][:, _HAND], gt_joints[:, _HAND])

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
