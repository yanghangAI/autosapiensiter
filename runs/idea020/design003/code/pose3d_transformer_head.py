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


def _temp_scaled_attn(
    mha_module: nn.MultiheadAttention,
    query: torch.Tensor,        # (B, Nq, D)
    key: torch.Tensor,          # (B, Ns, D)
    value: torch.Tensor,        # (B, Ns, D)
    temperature: torch.Tensor,  # (Nq,)  — nn.Parameter or derived tensor
    training: bool = True,
) -> torch.Tensor:
    """Cross-attention with per-query temperature scaling of logits.

    Implements:  attn = softmax( (Q @ K^T / sqrt(dh)) / tau )  @ V
    where tau is a per-query scalar (shape (Nq,)).

    The temperature is clamped to >= 0.1 to prevent logit overflow under AMP.
    tau = 1.0 at init -> identical to standard scaled dot-product attention.

    Uses mha_module.in_proj_weight / in_proj_bias for Q/K/V projections and
    mha_module.out_proj for the output projection.  Requires
    mha_module._qkv_same_embed_dim == True (verified in head __init__).
    """
    B, Nq, D = query.shape
    _, Ns, _ = key.shape
    Nh = mha_module.num_heads
    dh = D // Nh

    w = mha_module.in_proj_weight   # (3D, D)
    b = mha_module.in_proj_bias     # (3D,) or None

    Q = torch.nn.functional.linear(
        query, w[:D],    b[:D]    if b is not None else None)   # (B, Nq, D)
    K = torch.nn.functional.linear(
        key,   w[D:2*D], b[D:2*D] if b is not None else None)  # (B, Ns, D)
    V = torch.nn.functional.linear(
        value, w[2*D:],  b[2*D:]  if b is not None else None)  # (B, Ns, D)

    # Reshape to multi-head layout
    Q = Q.view(B, Nq, Nh, dh).transpose(1, 2)  # (B, Nh, Nq, dh)
    K = K.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)
    V = V.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)

    scale = dh ** -0.5
    attn = (Q @ K.transpose(-2, -1)) * scale    # (B, Nh, Nq, Ns)

    # Per-query temperature: clamp prevents collapse; cast to attn dtype for AMP
    tau = temperature.clamp(min=0.1).to(attn.dtype).view(1, 1, Nq, 1)
    attn = (attn / tau).softmax(dim=-1)          # (B, Nh, Nq, Ns)

    if training and mha_module.dropout > 0:
        attn = torch.nn.functional.dropout(attn, p=mha_module.dropout)

    out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)  # (B, Nq, D)
    return mha_module.out_proj(out)                                  # (B, Nq, D)


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

    def forward(
        self,
        queries: torch.Tensor,
        spatial_tokens: torch.Tensor,
        cross_temp: 'torch.Tensor | None' = None,
        self_temp: 'torch.Tensor | None' = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: ``(B, num_queries, embed_dim)``
            spatial_tokens: ``(B, num_spatial, embed_dim)``
            cross_temp: per-query cross-attention temperature (Nq,) or None.
            self_temp: per-query self-attention temperature (Nq,) or None.

        Returns:
            ``(B, num_queries, embed_dim)``
        """
        # Self-attention
        q = self.norm1(queries)
        if self_temp is not None:
            q2 = _temp_scaled_attn(
                self.self_attn, q, q, q,
                self_temp, training=self.training)
        else:
            q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        # Cross-attention
        q = self.norm2(queries)
        if cross_temp is not None:
            q2 = _temp_scaled_attn(
                self.cross_attn, q, spatial_tokens, spatial_tokens,
                cross_temp, training=self.training)
        else:
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
        use_cross_temp: bool = False,
        use_self_temp: bool = False,
        temp_log_space: bool = False,
        temp_reg_weight: float = 0.0,
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
        self.use_cross_temp = use_cross_temp
        self.use_self_temp = use_self_temp
        self.temp_log_space = temp_log_space
        self.temp_reg_weight = temp_reg_weight

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        # Project backbone features to hidden_dim
        self.input_proj = nn.Linear(in_channels, hidden_dim)

        # Learnable joint query embeddings
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)

        # Learnable temperature parameters (stored only in the head; passed to
        # _DecoderLayer.forward() by value each call to avoid duplicate param
        # registration that would occur if stored in both head and decoder layer)
        if use_cross_temp:
            if temp_log_space:
                # Log-space parameterisation: tau = softplus(log_cross_temp)
                self.log_cross_temp = nn.Parameter(torch.zeros(num_joints))
            else:
                # Direct parameterisation, clamped to >= 0.1 at use time
                self.cross_temp = nn.Parameter(torch.ones(num_joints))

        if use_self_temp:
            self.self_temp = nn.Parameter(torch.ones(num_joints))

        # Transformer decoder (1 layer)
        # Temperature parameters are stored in Pose3dTransformerHead (not in
        # _DecoderLayer) to avoid duplicate parameter registration. They are
        # passed explicitly in forward() each call.
        self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)

        # Verify standard QKV projection layout (required by _temp_scaled_attn)
        assert self.decoder_layer.cross_attn._qkv_same_embed_dim, \
            '_temp_scaled_attn requires _qkv_same_embed_dim=True'
        assert self.decoder_layer.self_attn._qkv_same_embed_dim, \
            '_temp_scaled_attn requires _qkv_same_embed_dim=True for self_attn'

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

        # Compute effective cross-attention temperature
        cross_temp = None
        if self.use_cross_temp:
            if self.temp_log_space:
                cross_temp = torch.nn.functional.softplus(self.log_cross_temp)
            else:
                cross_temp = self.cross_temp

        # Compute effective self-attention temperature
        self_temp = None
        if self.use_self_temp:
            self_temp = self.self_temp

        # Decoder
        decoded = self.decoder_layer(
            queries, spatial,
            cross_temp=cross_temp,
            self_temp=self_temp)  # (B, num_joints, hidden_dim)

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

        # Temperature L2 regularisation (Design 002: temp_reg_weight > 0)
        if self.temp_reg_weight > 0 and hasattr(self, 'log_cross_temp'):
            losses['loss/temp_reg/train'] = (
                self.temp_reg_weight * self.log_cross_temp.pow(2).mean())

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
