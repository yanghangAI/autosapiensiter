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
        kinematic_parametrization: bool = False,
        bone_parents: list = None,
        bone_length_loss_weight: float = 0.0,
        per_limb_heads: bool = False,
        limb_index: list = None,
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
        self.kinematic_parametrization = kinematic_parametrization
        self.bone_length_loss_weight = bone_length_loss_weight
        self.per_limb_heads = per_limb_heads

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

        # Kinematic-chain bone-vector output parameterization setup.
        if kinematic_parametrization:
            assert bone_parents is not None and len(bone_parents) == 22, (
                f"kinematic_parametrization=True requires bone_parents "
                f"(len-22 list of int), got {bone_parents!r}")
            assert bone_parents[0] == -1, (
                f"bone_parents[0] must be -1 (root), got {bone_parents[0]}")
            for child in range(1, 22):
                assert 0 <= bone_parents[child] < child, (
                    f"bone_parents[{child}]={bone_parents[child]} must satisfy "
                    f"0 <= p < child for a valid topologically-ordered "
                    f"kinematic tree.")
            self.register_buffer(
                'bone_parents',
                torch.tensor(bone_parents, dtype=torch.long),
                persistent=False)
            # Host-side Python list for the forward-kinematics loop — avoids
            # per-iteration device sync from `.item()`.
            self._bone_parents_list = list(bone_parents)
        else:
            self.bone_parents = None
            self._bone_parents_list = None

        # Per-limb decoupled body-bone-vec heads (Design 003).
        if per_limb_heads:
            assert kinematic_parametrization, (
                "per_limb_heads=True requires kinematic_parametrization=True.")
            assert limb_index is not None and len(limb_index) == 22, (
                f"per_limb_heads=True requires limb_index (len-22 list of "
                f"int), got {limb_index!r}")
            num_limbs = int(max(limb_index)) + 1
            assert num_limbs == 5, (
                f"Expected 5 limb groups (spine/left_leg/right_leg/"
                f"left_arm/right_arm), got {num_limbs} distinct limb indices.")
            for val in limb_index:
                assert 0 <= val < num_limbs, (
                    f"limb_index values must be in [0, {num_limbs - 1}], "
                    f"got {val}")

            self.body_limb_heads = nn.ModuleList([
                nn.Linear(self.hidden_dim, 3) for _ in range(num_limbs)
            ])

            self.register_buffer(
                'limb_index',
                torch.tensor(limb_index, dtype=torch.long),
                persistent=False)

            self._limb_token_lists = [
                [i for i in range(22) if limb_index[i] == limb_id]
                for limb_id in range(num_limbs)
            ]
            # Sanity: every body-token index 0..21 must appear exactly once.
            _covered = set()
            for tl in self._limb_token_lists:
                _covered.update(tl)
            assert _covered == set(range(22)), (
                f"limb_index must cover body-token indices 0..21 exactly "
                f"once; got coverage {sorted(_covered)}")

            # Register per-limb index buffers to avoid per-forward allocation.
            for i, token_list in enumerate(self._limb_token_lists):
                self.register_buffer(
                    f'_limb_idx_{i}',
                    torch.tensor(token_list, dtype=torch.long),
                    persistent=False)
        else:
            self.body_limb_heads = None
            self.limb_index = None
            self._limb_token_lists = None

        # Positional encoding buffer — registered lazily on first forward
        self._pos_enc_hw: Tuple[int, int] | None = None

        self._init_head_weights()

    @property
    def default_init_cfg(self):
        return []

    def _init_head_weights(self) -> None:
        # Query embeddings
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        # Original (shared) output projections
        for m in [self.joints_out, self.depth_out, self.uv_out]:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Per-limb heads (Design 003 only)
        if self.per_limb_heads:
            for m in self.body_limb_heads:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Scale-init body bone-vec head(s) to keep recovered-joint variance
        # comparable to baseline's direct-regression variance after the
        # cumulative-sum forward-kinematics transform.
        if self.kinematic_parametrization:
            num_body_bones = 21  # 22 body joints - 1 root
            scale = 1.0 / math.sqrt(num_body_bones)
            with torch.no_grad():
                self.joints_out.weight.mul_(scale)
                if self.per_limb_heads:
                    for m in self.body_limb_heads:
                        m.weight.mul_(scale)

    def _forward_kinematics(self, bone_vecs: torch.Tensor) -> torch.Tensor:
        """Recover root-relative body joint positions from bone-translation
        vectors via cumulative sum along the parent chain.

        Args:
            bone_vecs: Tensor of shape (B, 22, 3). Entry [:, 0, :] is the
                root and is ignored (overwritten with zero). Entries
                [:, 1..21, :] are interpreted as bone vectors from
                parent[i] to joint i.

        Returns:
            Tensor of shape (B, 22, 3) with root-relative joint positions.
            The output at index 0 is exactly the zero vector.
        """
        # Clone so in-place writes do not corrupt the upstream `joints`
        # tensor (or the raw bone_vec output of the per-limb heads).
        body_rr = bone_vecs.clone()
        body_rr[:, 0, :] = 0.0
        parents = self._bone_parents_list  # Python list[int], host-side.
        for child in range(1, 22):
            parent = parents[child]
            body_rr[:, child, :] = body_rr[:, parent, :] + bone_vecs[:, child, :]
        return body_rr

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

        # Decoder
        decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)

        # Output projections
        if self.per_limb_heads:
            # Per-limb body-bone-vec heads (Design 003).
            # Each body token (0..21) is routed to the head of its kinematic
            # subtree; hand tokens go through the shared `joints_out` head.
            body_bone_vecs = decoded.new_zeros(B, 22, 3)
            for limb_id, token_list in enumerate(self._limb_token_lists):
                if len(token_list) == 0:
                    continue
                idx = getattr(self, f'_limb_idx_{limb_id}')
                sel = decoded.index_select(1, idx)                      # (B, k, hidden_dim)
                bone_vecs_limb = self.body_limb_heads[limb_id](sel)      # (B, k, 3)
                body_bone_vecs = body_bone_vecs.index_copy(1, idx, bone_vecs_limb)

            hand_decoded = decoded[:, 22:self.num_joints, :]
            hand_coords = self.joints_out(hand_decoded)                  # (B, 48, 3)

            body_rr = self._forward_kinematics(body_bone_vecs)           # (B, 22, 3)
            joints = torch.cat([body_rr, hand_coords], dim=1)            # (B, num_joints, 3)
        else:
            joints = self.joints_out(decoded)  # (B, num_joints, 3)

            if self.kinematic_parametrization:
                # Interpret the first 22 entries as bone-translation vectors
                # and recover root-relative body coords via forward kinematics.
                body_bone_vecs = joints[:, 0:22, :]
                hand_coords = joints[:, 22:self.num_joints, :]
                body_rr = self._forward_kinematics(body_bone_vecs)
                joints = torch.cat([body_rr, hand_coords], dim=1)

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

        # Auxiliary bone-length loss on body joints (Design 002 only).
        # Plain L1 on bone-vector magnitudes.
        if self.kinematic_parametrization and self.bone_length_loss_weight > 0.0:
            device = pred['joints'].device
            child_idx = torch.arange(1, 22, device=device)           # (21,)
            parent_idx = self.bone_parents[1:22].to(device)          # (21,)

            gt_body = gt_joints[:, _BODY]                            # (B, 22, 3)
            gt_bones = gt_body[:, child_idx, :] - gt_body[:, parent_idx, :]

            pred_body = pred['joints'][:, _BODY]
            pred_bones = pred_body[:, child_idx, :] - pred_body[:, parent_idx, :]

            gt_bone_len = gt_bones.norm(dim=-1)                      # (B, 21)
            pred_bone_len = pred_bones.norm(dim=-1)                  # (B, 21)

            L_bone_len = (pred_bone_len - gt_bone_len).abs().mean()
            losses['loss/bone_length/train'] = (
                self.bone_length_loss_weight * L_bone_len)

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
