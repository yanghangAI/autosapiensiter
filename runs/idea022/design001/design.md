**Design Description:** 2-layer cascaded decoder with dynamic Gaussian reprojection bias (fixed σ=4, γ=2), no auxiliary loss.

**Starting Point:** `baseline/`

---

## Overview

**Algorithm:** Two-stage cascaded decoding with geometry-guided cross-attention. Layer 1 produces an intermediate 3D pose estimate; those predicted joints are reprojected to 2D image/feature-grid coordinates using camera intrinsics K; a per-sample Gaussian additive bias is constructed over the 40×24 feature grid centred on each projected 2D joint location and injected into layer 2's cross-attention logits. Layer 2 then refines the queries with spatially-focused attention.

Add a second transformer decoder layer. After the first layer produces an intermediate joint and pelvis prediction, project those predicted 3D joints to 2D feature-grid coordinates using camera intrinsics K (available in `batch_data_samples`). Build a per-sample Gaussian additive attention bias centred on each projected joint location and inject it into the second decoder layer's cross-attention logits. No intermediate loss on layer-1 output — layer-1 learns only through gradient backpropagated from layer-2's final loss and the bias construction chain.

This is the minimal test: does dynamic geometric feedback improve over a plain 2-layer decoder (idea001/design001)?

---

## Files to Change

### 1. `pelvis_utils.py`

Add one new helper function at the end of the file:

```python
def project_joints_to_feat_grid(
    joints_abs: torch.Tensor,
    K,
    crop_h: int,
    crop_w: int,
    feat_h: int = 40,
    feat_w: int = 24,
) -> torch.Tensor:
    """Project absolute 3D joints (camera frame) to feature-grid coordinates.

    BEDLAM2 convention: X=forward (depth), Y=left, Z=up.
    Projection: u_px = fx*(-Y/X) + cx,  v_px = fy*(-Z/X) + cy

    Args:
        joints_abs: (B, J, 3) absolute camera-frame joints in metres.
        K: (3, 3) crop intrinsic matrix (numpy array or anything with K[0,0]).
        crop_h: Crop height in pixels.
        crop_w: Crop width in pixels.
        feat_h: Feature grid height (default 40).
        feat_w: Feature grid width (default 24).

    Returns:
        (B, J, 2) float tensor: (h_frac, w_frac) in feature grid units,
        clamped to [0, feat_h) x [0, feat_w).
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    X = joints_abs[..., 0].clamp(min=0.01)  # (B, J) forward depth, avoid div-by-zero
    Y = joints_abs[..., 1]                   # (B, J) lateral
    Z = joints_abs[..., 2]                   # (B, J) vertical

    u_px = -Y / X * fx + cx                  # (B, J) pixel u in crop
    v_px = -Z / X * fy + cy                  # (B, J) pixel v in crop

    h_frac = v_px / crop_h * feat_h          # (B, J)
    w_frac = u_px / crop_w * feat_w          # (B, J)

    h_frac = h_frac.clamp(0.0, float(feat_h) - 1e-4)
    w_frac = w_frac.clamp(0.0, float(feat_w) - 1e-4)

    return torch.stack([h_frac, w_frac], dim=-1)  # (B, J, 2)
```

No other changes to `pelvis_utils.py`.

### 2. `pose3d_transformer_head.py`

#### 2a. Imports

Add to the existing imports at the top of the file:

```python
import torch.nn.functional as F
import numpy as np
from pelvis_utils import recover_pelvis_3d, project_joints_to_feat_grid
```

Note: `recover_pelvis_3d` is already imported via `compute_mpjpe_abs` helper but must now be imported directly for use in `loss()`. If `from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs` is the existing import, add a separate line:

```python
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          recover_pelvis_3d,
                          project_joints_to_feat_grid)
```

#### 2b. New module-level function `_build_gaussian_bias`

Add after the `_build_2d_sincos_pos_enc` function and before `_DecoderLayer`:

```python
def _build_gaussian_bias(
    joint_feat_coords: torch.Tensor,
    feat_h: int,
    feat_w: int,
    sigma: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Build dynamic Gaussian cross-attention additive bias.

    Args:
        joint_feat_coords: (B, J, 2) — (h_frac, w_frac) in feature grid units.
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
```

#### 2c. `_DecoderLayer.forward` — add optional `cross_attn_bias` argument

Replace the existing `forward` signature and cross-attention block in `_DecoderLayer`:

Existing:
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor) -> torch.Tensor:
```
Replace with:
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            cross_attn_bias: torch.Tensor = None) -> torch.Tensor:
```

Replace the existing cross-attention block:
```python
# Cross-attention
q = self.norm2(queries)
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```
With:
```python
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
```

No other changes to `_DecoderLayer`.

#### 2d. `Pose3dTransformerHead.__init__` — add new parameters and replace single decoder layer with ModuleList

New constructor signature (add parameters after `dropout`, before `loss_joints`):

```python
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
    loss_joints: ConfigType = ...,
    loss_depth: ConfigType = ...,
    loss_uv: ConfigType = ...,
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    init_cfg: OptConfigType = None,
):
```

Store new attributes after the existing `self.loss_weight_uv = loss_weight_uv` line:

```python
self.num_decoder_layers = num_decoder_layers
self.use_reproj_bias = use_reproj_bias
self.reproj_bias_sigma = reproj_bias_sigma
self.reproj_bias_gamma = reproj_bias_gamma
self.reproj_bias_learnable = reproj_bias_learnable
self.aux_loss_weight = aux_loss_weight
self.feat_h = feat_h
self.feat_w = feat_w
```

Replace:
```python
# Transformer decoder (1 layer)
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```
With:
```python
# Transformer decoder (N layers)
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

For Design A (no learnable bias params), no `nn.Parameter` additions needed.

#### 2e. `Pose3dTransformerHead.forward` — replace decoder call with multi-layer loop with optional bias

Replace:
```python
# Decoder
decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)
```
With:
```python
# Decoder — layer 0 always runs without bias
decoded = self.decoder_layers[0](queries, spatial)

# Subsequent layers: use reprojection bias if enabled and bias is available
for layer_idx in range(1, self.num_decoder_layers):
    bias = getattr(self, '_reproj_bias', None)
    decoded = self.decoder_layers[layer_idx](decoded, spatial,
                                              cross_attn_bias=bias)
```

At the end of `forward()`, clear the stored bias to avoid stale values leaking across calls:

```python
# Clear stored bias after use (set in loss(), not used at test time)
if hasattr(self, '_reproj_bias'):
    self._reproj_bias = None
```

Add this line immediately before the `return` statement in `forward()`.

#### 2f. `Pose3dTransformerHead.loss` — compute reprojection bias before calling forward

In `loss()`, replace the existing `pred = self.forward(feats)` call with the following block:

```python
# ── Compute reprojection bias from intermediate layer-1 predictions ──────
if self.use_reproj_bias and self.num_decoder_layers > 1:
    # Run layer 0 only to get intermediate predictions
    feat = feats[-1]
    B_tmp, C_tmp, H_tmp, W_tmp = feat.shape
    spatial_tmp = feat.flatten(2).transpose(1, 2)
    spatial_tmp = self.input_proj(spatial_tmp)
    pos_enc_tmp = self._get_pos_enc(H_tmp, W_tmp, feat.device)
    spatial_tmp = spatial_tmp + pos_enc_tmp
    queries_tmp = self.joint_queries.weight.unsqueeze(0).expand(B_tmp, -1, -1)
    with torch.no_grad():
        decoded_l1 = self.decoder_layers[0](queries_tmp, spatial_tmp)
    layer1_joints = self.joints_out(decoded_l1)        # (B, J, 3)
    layer1_depth  = self.depth_out(decoded_l1[:, 0])   # (B, 1)
    layer1_uv     = self.uv_out(decoded_l1[:, 0])      # (B, 2)

    # Recover absolute 3D positions and project to feature grid
    abs_joints_list = []
    feat_coords_list = []
    for i in range(B_tmp):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo['K'], dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h, crop_w = int(img_shape[0]), int(img_shape[1])
        pelvis = recover_pelvis_3d(
            layer1_depth[i:i+1], layer1_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
        abs_j = layer1_joints[i] + pelvis   # (J, 3) — broadcast over joints
        abs_joints_list.append(abs_j)
        fc = project_joints_to_feat_grid(
            abs_j.unsqueeze(0), K, crop_h, crop_w,
            self.feat_h, self.feat_w)  # (1, J, 2)
        feat_coords_list.append(fc[0])

    feat_coords = torch.stack(feat_coords_list)  # (B, J, 2)

    # Build fixed-parameter Gaussian bias (Design A: no learnable params)
    sigma = torch.full(
        (self.num_joints,), self.reproj_bias_sigma,
        device=feat_coords.device, dtype=feat_coords.dtype)
    gamma = torch.full(
        (self.num_joints,), self.reproj_bias_gamma,
        device=feat_coords.device, dtype=feat_coords.dtype)
    self._reproj_bias = _build_gaussian_bias(
        feat_coords, self.feat_h, self.feat_w, sigma, gamma)  # (B, J, H'W')

pred = self.forward(feats)
```

**Important**: the intermediate layer-0 forward is done with `torch.no_grad()` in Design A because no auxiliary loss is applied. This means gradients do NOT flow from the bias construction back to layer-0 weights (the only gradient path to layer 0 is through the full forward pass in `self.forward(feats)`). This is intentional for Design A — the geometric bias is treated as a data-dependent but gradient-free prior. The full `self.forward(feats)` call (which re-runs all layers) provides the gradient path.

After `pred = self.forward(feats)`, all downstream loss computation (joint loss, depth loss, UV loss, MPJPE tracking) is unchanged from baseline.

#### 2g. No changes to `predict()`

`predict()` calls `self.forward(feats)` directly. At test time, `_reproj_bias` is never set, so `getattr(self, '_reproj_bias', None)` returns `None` and the second decoder layer runs standard cross-attention. This is safe and conservative.

---

## Config Changes (`config.py`)

In the `head=dict(...)` block, add the following keys (all literals, no imports):

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_decoder_layers=2,
    use_reproj_bias=True,
    reproj_bias_sigma=4.0,
    reproj_bias_gamma=2.0,
    reproj_bias_learnable=False,
    aux_loss_weight=0.0,
    feat_h=40,
    feat_w=24,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config sections (optimizer, LR schedule, data pipeline, hooks) are identical to the baseline.

---

## Invariants the Builder Must Preserve

1. Loss is restricted to body joints (indices 0–21) in all joint loss terms.
2. `persistent_workers=False` — do not change.
3. `batch_first=True` on all `nn.MultiheadAttention` instances — required for `(B*nheads, J, H'W')` attn_mask shape.
4. Feature grid dimensions: `feat_h=40, feat_w=24` (feature stride 16 from 640×384 input). The spatial token ordering matches `feat.flatten(2).transpose(1, 2)` which produces row-major (H, W) order — consistent with the Gaussian bias construction (`indexing='ij'`).
5. The `_reproj_bias` attribute is cleared at the end of `forward()` before `return`. This prevents stale biases from leaking into subsequent forward calls (e.g., during validation).
6. AMP compatibility: the `cross_attn_bias` tensor is cast to `q.dtype` via `.to(q.dtype)` before passing to `nn.MultiheadAttention` — this ensures float16 compatibility.
7. The intermediate layer-0 forward in `loss()` uses `torch.no_grad()` to avoid double-computing gradients through the spatial features. The full `self.forward(feats)` call re-runs all layers with autograd.
8. No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, or any infra files.
9. Output dict keys from `forward()` remain `{'joints', 'pelvis_depth', 'pelvis_uv'}` with unchanged shapes.

---

## Expected Behaviour

- Stage-1 `composite_val` target: < 340 (vs. baseline 338.78 at stage-1 from idea001/design001 which was best 2-layer config).
- Stage-2 `composite_val` target: < 224 (vs. best prior 224.52 from idea001/design001).
- `mpjpe_body_val` stage-1 target: < 195 mm (vs. baseline 195.7 mm).
- The dynamic Gaussian bias should direct layer-2 cross-attention to the image regions predicted by layer 1, reducing irrelevant spatial token contributions.
