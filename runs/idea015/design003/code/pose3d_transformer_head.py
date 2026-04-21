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
        → [optional] slot-attention pooling: (B, H'*W', hidden_dim) → (B, K, hidden_dim)
        → transformer decoder (N layers):
            self-attention over 70 joint queries
            cross-attention: queries attend to spatial tokens (or super-tokens)
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
        num_super_tokens (int): Number of slot-attention super-tokens. 0 = disabled
            (baseline path, cross-attends over all spatial tokens). >0 enables
            slot pooling: spatial tokens are compressed to K super-tokens before
            being passed to the decoder.
        slot_pos_init (bool): If True, initialise slot query embeddings from
            block-averaged 2D sinusoidal positional encodings (spatially grounded
            warm start). Requires num_super_tokens == 64.
        num_decoder_layers (int): Number of stacked decoder layers. Default 1
            (same as baseline).
        aux_loss_weight (float): Weight for auxiliary joint loss on intermediate
            decoder layer outputs. 0.0 = no auxiliary loss.
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
        num_super_tokens: int = 0,
        slot_pos_init: bool = False,
        num_decoder_layers: int = 1,
        aux_loss_weight: float = 0.0,
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
        self.num_super_tokens = num_super_tokens
        self.slot_pos_init = slot_pos_init
        self.num_decoder_layers = num_decoder_layers
        self.aux_loss_weight = aux_loss_weight

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        # Project backbone features to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Learnable joint query embeddings
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)

        # Slot-attention pooling (enabled when num_super_tokens > 0)
        if self.num_super_tokens > 0:
            self.slot_queries = nn.Embedding(num_super_tokens, hidden_dim)
            self.slot_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.slot_norm = nn.LayerNorm(hidden_dim)

        # Transformer decoder (N layers, default 1)
        self.decoder_layers = nn.ModuleList([
            _DecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
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
        # Slot query embeddings
        if self.num_super_tokens > 0:
            nn.init.trunc_normal_(self.slot_queries.weight, std=0.02)
            if self.slot_pos_init:
                # Build 2D sincos pos enc for H'=24, W'=40
                assert self.num_super_tokens == 64, (
                    'slot_pos_init=True requires num_super_tokens=64')
                pos = _build_2d_sincos_pos_enc(24, 40, self.hidden_dim)  # (1, 960, hidden_dim)
                pos = pos.squeeze(0).reshape(24, 40, self.hidden_dim)    # (24, 40, hidden_dim)
                # Partition into 8x8=64 blocks, each block is 3 rows x 5 cols
                block_means = []
                for r in range(8):
                    for c in range(8):
                        block = pos[3*r : 3*r+3, 5*c : 5*c+5, :]  # (3, 5, hidden_dim)
                        block_means.append(block.reshape(-1, self.hidden_dim).mean(0))  # (hidden_dim,)
                slot_init = torch.stack(block_means, dim=0)  # (64, hidden_dim)
                self.slot_queries.weight.data.copy_(slot_init)

    def _get_pos_enc(self, h: int, w: int,
                     device: torch.device) -> torch.Tensor:
        """Get or recompute 2D positional encoding buffer."""
        if self._pos_enc_hw != (h, w):
            pos = _build_2d_sincos_pos_enc(h, w, self.hidden_dim)
            self.register_buffer('pos_enc', pos, persistent=False)
            self._pos_enc_hw = (h, w)
        return self.pos_enc.to(device)

    def _forward_with_intermediates(
        self, feats: Tuple[torch.Tensor, ...]
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Forward pass returning both final outputs and intermediate decoder outputs.

        Returns:
            (pred_dict, intermediate_outputs):
                pred_dict: same as forward() return dict
                intermediate_outputs: list of (B, num_joints, hidden_dim) tensors,
                    one per decoder layer, in order
        """
        feat = feats[-1]  # (B, C, H, W)
        B, C, H, W = feat.shape

        # Flatten spatial dims, project to hidden_dim, add positional encoding
        spatial = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        spatial = self.input_proj(spatial)          # (B, H*W, hidden_dim)
        pos_enc = self._get_pos_enc(H, W, feat.device)
        spatial = spatial + pos_enc

        # Slot-attention pooling
        if self.num_super_tokens > 0:
            S = self.slot_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, hidden_dim)
            S_normed = self.slot_norm(S)
            super_tokens, _ = self.slot_attn(S_normed, spatial, spatial)  # (B, K, hidden_dim)
            spatial_for_decoder = super_tokens
        else:
            spatial_for_decoder = spatial  # (B, H*W, hidden_dim) — baseline path

        # Broadcast joint queries to batch
        queries = self.joint_queries.weight.unsqueeze(0).expand(
            B, -1, -1)  # (B, num_joints, hidden_dim)

        # Decoder loop over N layers
        decoded = queries
        intermediate_outputs = []
        for layer in self.decoder_layers:
            decoded = layer(decoded, spatial_for_decoder)
            intermediate_outputs.append(decoded)

        # Output projections
        joints = self.joints_out(decoded)  # (B, num_joints, 3)

        pelvis_token = decoded[:, 0, :]  # (B, hidden_dim)
        pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
        pelvis_uv = self.uv_out(pelvis_token)  # (B, 2)

        pred = {
            'joints': joints,
            'pelvis_depth': pelvis_depth,
            'pelvis_uv': pelvis_uv,
        }
        return pred, intermediate_outputs

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
        pred, _ = self._forward_with_intermediates(feats)
        return pred

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
        pred, intermediate_outputs = self._forward_with_intermediates(feats)

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

        # Auxiliary loss on intermediate decoder outputs (all layers except last)
        if self.aux_loss_weight > 0.0:
            for i, inter_decoded in enumerate(intermediate_outputs[:-1]):
                inter_joints = self.joints_out(inter_decoded)  # (B, num_joints, 3)
                losses[f'loss/joints_aux_{i}/train'] = self.aux_loss_weight * self.loss_joints_module(
                    inter_joints[:, _BODY], gt_joints[:, _BODY])

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
