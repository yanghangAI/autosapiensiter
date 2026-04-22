"""Transformer decoder head for 3D pose regression.

idea034 / design003 (Variant C): metric 3D PE via depth unprojection,
embedded through a 2-layer MLP and added ONLY to the cross-attention keys
(values remain pure appearance + PE_2D).

Takes a ViT feature map ``(B, C, H', W')`` and regresses:
  - ``(B, num_joints, 3)`` root-relative joint XYZ in metres
  - ``(B, 1)``             pelvis depth (forward distance) in metres
  - ``(B, 2)``             pelvis 2D position in crop pixels, normalised to [-1, 1]
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
from pelvis_utils import unproject_grid_to_metric_3d


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


class _Metric3DPE(nn.Module):
    """Embed per-token metric 3D coordinates (X, Y, Z in metres) → hidden_dim.

    Architecture: Linear(3, mlp_hidden) → GELU → Linear(mlp_hidden, hidden_dim).
    Final Linear is zero-initialised so PE_3D = 0 at step 0 (identity wrt baseline).
    """

    def __init__(self, hidden_dim: int, mlp_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(3, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_dim)
        self.act = nn.GELU()
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(p)))


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
                spatial_values: torch.Tensor,
                spatial_keys: torch.Tensor | None = None) -> torch.Tensor:
        if spatial_keys is None:
            spatial_keys = spatial_values

        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        q = self.norm2(queries)
        q2 = self.cross_attn(q, spatial_keys, spatial_values)[0]
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
        use_metric_pe_3d: bool = False,
        metric_pe_variant: str = 'keys_only',
        metric_pe_mlp_hidden: int = 256,
        metric_pe_depth_clamp_min: float = 0.1,
        metric_pe_depth_clamp_max: float = 50.0,
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

        self.loss_joints_module = MODELS.build(loss_joints)
        self.loss_depth_module = MODELS.build(loss_depth)
        self.loss_uv_module = MODELS.build(loss_uv)

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.joint_queries = nn.Embedding(num_joints, hidden_dim)
        self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)

        self.joints_out = nn.Linear(hidden_dim, 3)
        self.depth_out = nn.Linear(hidden_dim, 1)
        self.uv_out = nn.Linear(hidden_dim, 2)

        self._pos_enc_hw: Tuple[int, int] | None = None

        # Metric 3D PE config (Variant A)
        self.use_metric_pe_3d = bool(use_metric_pe_3d)
        self.metric_pe_variant = str(metric_pe_variant)
        self.metric_pe_depth_clamp_min = float(metric_pe_depth_clamp_min)
        self.metric_pe_depth_clamp_max = float(metric_pe_depth_clamp_max)
        if self.use_metric_pe_3d:
            assert self.metric_pe_variant == 'keys_only', \
                f"design003 requires metric_pe_variant='keys_only', got {self.metric_pe_variant}"
            self.metric_pe_3d = _Metric3DPE(
                hidden_dim, mlp_hidden=int(metric_pe_mlp_hidden))

        self._init_head_weights()

    @property
    def default_init_cfg(self):
        return []

    def _init_head_weights(self) -> None:
        nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
        for m in [self.joints_out, self.depth_out, self.uv_out]:
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

    def _extract_depth_map(
        self,
        batch_data_samples: OptSampleList,
        target_h: int,
        target_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Load and resize raw depth maps to ``(B, 1, target_h, target_w)``."""
        depth_maps = []
        for ds in batch_data_samples:
            depth_npy_path = ds.metainfo.get('depth_npy_path', None)
            img_shape = ds.metainfo.get('img_shape', None)
            try:
                raw = np.load(depth_npy_path)
                if isinstance(raw, np.lib.npyio.NpzFile):
                    key = 'depth' if 'depth' in raw else list(raw.keys())[0]
                    raw = raw[key]
                if raw.ndim == 3:
                    raw = raw[0]
                if img_shape is not None:
                    ch, cw = int(img_shape[0]), int(img_shape[1])
                    raw = raw[:ch, :cw]
                depth_tensor = torch.from_numpy(raw.astype(np.float32))
                depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
            except Exception:
                if img_shape is not None:
                    ch, cw = int(img_shape[0]), int(img_shape[1])
                else:
                    ch, cw = target_h, target_w
                depth_tensor = torch.zeros(1, 1, ch, cw, dtype=torch.float32)
            depth_maps.append(depth_tensor)

        resized = []
        for d in depth_maps:
            r = F.interpolate(d, size=(target_h, target_w), mode='bilinear',
                              align_corners=False)
            resized.append(r)
        depth_batch = torch.cat(resized, dim=0).to(device)
        return depth_batch

    def _build_K_batch(
        self,
        batch_data_samples: OptSampleList,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(K_batch (B,3,3), crop_hw (B,2))`` in fp32 on ``device``."""
        K_list = []
        crop_list = []
        for ds in batch_data_samples:
            K_np = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
            if K_np.shape != (3, 3):
                K_np = np.eye(3, dtype=np.float32)
            img_shape = ds.metainfo.get('img_shape', (640, 384))
            ch, cw = int(img_shape[0]), int(img_shape[1])
            K_list.append(torch.from_numpy(K_np))
            crop_list.append(torch.tensor([float(ch), float(cw)],
                                          dtype=torch.float32))
        K_batch = torch.stack(K_list, dim=0).to(device)
        crop_hw = torch.stack(crop_list, dim=0).to(device)
        return K_batch, crop_hw

    def _compute_metric_xyz(
        self,
        feats: Tuple[torch.Tensor, ...],
        batch_data_samples: OptSampleList,
    ) -> torch.Tensor | None:
        if not self.use_metric_pe_3d:
            return None
        feat_last = feats[-1]
        feat_h, feat_w = feat_last.shape[2], feat_last.shape[3]
        device = feat_last.device
        depth_grid = self._extract_depth_map(
            batch_data_samples, feat_h, feat_w, device)
        K_batch, crop_hw = self._build_K_batch(batch_data_samples, device)
        metric_xyz = unproject_grid_to_metric_3d(
            depth_grid, K_batch, crop_hw, feat_h, feat_w,
            d_min=self.metric_pe_depth_clamp_min,
            d_max=self.metric_pe_depth_clamp_max,
        )
        return metric_xyz

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
        metric_xyz: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        feat = feats[-1]
        B, C, H, W = feat.shape

        spatial = feat.flatten(2).transpose(1, 2)
        spatial = self.input_proj(spatial)
        pos_enc = self._get_pos_enc(H, W, feat.device)
        spatial_values = spatial + pos_enc

        spatial_keys = None
        if self.use_metric_pe_3d and metric_xyz is not None:
            pe3d = self.metric_pe_3d(metric_xyz.to(spatial_values.dtype))
            spatial_keys = spatial_values + pe3d

        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder_layer(queries, spatial_values, spatial_keys)

        joints = self.joints_out(decoded)

        pelvis_token = decoded[:, 0, :]
        pelvis_depth = self.depth_out(pelvis_token)
        pelvis_uv = self.uv_out(pelvis_token)

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
        metric_xyz = self._compute_metric_xyz(feats, batch_data_samples)
        pred = self.forward(feats, metric_xyz=metric_xyz)

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

        _BODY = list(range(0, 22))
        losses = dict()
        losses['loss/joints/train'] = self.loss_joints_module(
            pred['joints'][:, _BODY], gt_joints[:, _BODY])
        losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
            pred['pelvis_depth'], gt_depth)
        losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
            pred['pelvis_uv'], gt_uv)

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
        metric_xyz = self._compute_metric_xyz(feats, batch_data_samples)
        pred = self.forward(feats, metric_xyz=metric_xyz)
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
