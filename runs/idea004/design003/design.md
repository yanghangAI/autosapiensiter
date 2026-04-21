**Design Description:** Depth+2D MLP positional encoding — learned 2-layer MLP takes normalised (x, y, depth) per token as input, replacing the fixed 2D sinusoidal positional encoding entirely.

**Starting Point:** `baseline/`

---

## Overview

This design replaces the fixed 2D sinusoidal positional encoding with a learned 3-input positional MLP (NeRF-style). Each spatial token receives its normalised 2D grid position `(norm_x, norm_y)` in `[-1, 1]` and its normalised depth `norm_depth` in `[0, 1]`, concatenated into a `(B, H'*W', 3)` tensor. A 2-layer MLP with GELU activation maps this to a `hidden_dim`-dimensional positional embedding:

```
pos_input  = cat([norm_x, norm_y, norm_depth], dim=-1)   # (B, H'*W', 3)
spatial    = input_proj(feat) + pos_mlp(pos_input)        # pos_mlp: 3 → 64 → hidden_dim
```

The fixed `_build_2d_sincos_pos_enc` function is no longer called in the hot path (it is retained in the file for completeness but not used). The MLP is the sole source of positional information.

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `pos_mlp` module, add `_extract_depth_map` and `_build_3d_pos_grid` helpers, update `__init__`, `forward`, `loss`, `predict`.
2. `config.py` — add `depth_pos_enc_type='mlp'` kwarg to the head config dict.

`pelvis_utils.py` is **not** modified.

---

## Exact Changes — `pose3d_transformer_head.py`

### New imports

Add to existing imports at the top:

```python
import numpy as np
import torch.nn.functional as F
```

### `__init__` signature change

Add `depth_pos_enc_type: str = 'mlp'` as a new parameter after `loss_weight_uv` and before `init_cfg`:

```python
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
    depth_pos_enc_type: str = 'mlp',
    init_cfg: OptConfigType = None,
):
```

### `__init__` additions

After the line `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
# Depth-aware spatial positional encoding (Design C: 3-input MLP)
self.depth_pos_enc_type = depth_pos_enc_type
pos_mlp_hidden = 64  # hidden dim of the positional MLP
self.pos_mlp = nn.Sequential(
    nn.Linear(3, pos_mlp_hidden),
    nn.GELU(),
    nn.Linear(pos_mlp_hidden, hidden_dim),
)
# Initialise pos_mlp layers with trunc_normal std=0.02, zero bias
for layer in self.pos_mlp:
    if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, std=0.02)
        nn.init.zeros_(layer.bias)
```

**Initialisation rationale**: `trunc_normal_(std=0.02)` is consistent with all other projection layers in the head. Zero bias means the MLP starts with a near-zero output (since inputs are in [-1, 1] and weights are small), which is a conservative starting point. The MLP learns to produce meaningful positional embeddings during training.

**`pos_mlp_hidden = 64`**: hardcoded intermediate dimension, not exposed as a config parameter (matches the idea description and is sufficient for a 3-input → 256-output MLP).

### New helper method `_extract_depth_map`

Add this method to `Pose3dTransformerHead` (before `forward`). It is **identical** to the helper described in Design A:

```python
def _extract_depth_map(
    self,
    batch_data_samples: OptSampleList,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract, load, and resize raw depth maps from batch data samples.

    Returns:
        ``(B, 1, target_h, target_w)`` float tensor with raw depth in metres.
        Zero-filled for samples with missing depth data.
    """
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
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, ch, cw)
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
    depth_batch = torch.cat(resized, dim=0).to(device)  # (B, 1, target_h, target_w)
    return depth_batch
```

### New helper method `_build_3d_pos_grid`

Add this method to `Pose3dTransformerHead` (before `forward`):

```python
def _build_3d_pos_grid(
    self,
    h: int,
    w: int,
    depth_map: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Build (x, y, depth) position grid for the MLP positional encoding.

    Args:
        h: Feature map height H'.
        w: Feature map width W'.
        depth_map: ``(B, 1, h, w)`` depth values in metres, or None.
        device: Target device.

    Returns:
        ``(B, h*w, 3)`` grid of normalised (x, y, depth) positions.
        x and y are in [-1, 1]; depth is in [0, 1].
        If depth_map is None, depth channel is filled with 0.5 (mid-range).
    """
    # Build (y, x) grid normalised to [-1, 1]
    grid_y = torch.linspace(-1.0, 1.0, h, device=device)  # (h,)
    grid_x = torch.linspace(-1.0, 1.0, w, device=device)  # (w,)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing='ij')  # (h, w) each
    # Flatten to (h*w,)
    yy_flat = yy.reshape(-1)   # (h*w,)
    xx_flat = xx.reshape(-1)   # (h*w,)

    if depth_map is not None:
        B = depth_map.shape[0]
        # Normalise depth: clamp [0, 10] m → [0, 1]
        depth_flat = depth_map.flatten(2).squeeze(1)  # (B, h*w)
        depth_flat = depth_flat.clamp(min=0.0, max=10.0) / 10.0
    else:
        # Fallback: depth = 0.5 (mid-range), broadcast to batch size 1
        # (B is unknown here; handled in forward by expanding)
        B = 1
        depth_flat = torch.full((1, h * w), 0.5, device=device)

    # Stack (x, y, depth) → (B, h*w, 3)
    xx_batch = xx_flat.unsqueeze(0).expand(B, -1)  # (B, h*w)
    yy_batch = yy_flat.unsqueeze(0).expand(B, -1)  # (B, h*w)
    pos_grid = torch.stack([xx_batch, yy_batch, depth_flat], dim=-1)  # (B, h*w, 3)
    return pos_grid
```

**Note on x/y ordering**: the MLP takes `(x=col direction, y=row direction, depth)`. The convention is: x increases left-to-right, y increases top-to-bottom. Both normalised to `[-1, 1]`.

**Note on fallback (depth=0.5)**: when depth_map is None, depth is set to 0.5 (mid-range in normalised [0,1] space, corresponding to ~5 m). This is a non-zero fallback so the MLP still produces a spatial encoding consistent with a neutral depth. Unlike Design A/B which use zero fallback, this is more informative.

### `forward` signature and body changes

Update `forward` to accept an optional `depth_map` keyword argument:

```python
def forward(
    self,
    feats: Tuple[torch.Tensor, ...],
    depth_map: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
```

Replace the spatial token construction block inside `forward`:

```python
feat = feats[-1]  # (B, C, H, W)
B, C, H, W = feat.shape

# Flatten spatial dims, project to hidden_dim
spatial = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
spatial = self.input_proj(spatial)          # (B, H*W, hidden_dim)

# 3-input MLP positional encoding (x, y, depth)
pos_grid = self._build_3d_pos_grid(H, W, depth_map, feat.device)
# If depth_map is None, pos_grid is (1, H*W, 3); expand to batch
if depth_map is None:
    pos_grid = pos_grid.expand(B, -1, -1)  # (B, H*W, 3)
pos_embed = self.pos_mlp(pos_grid)          # (B, H*W, hidden_dim)
spatial = spatial + pos_embed

# Broadcast joint queries to batch
queries = self.joint_queries.weight.unsqueeze(0).expand(
    B, -1, -1)  # (B, num_joints, hidden_dim)

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
```

**Important**: `_get_pos_enc` is **no longer called** in `forward`. The `pos_enc` buffer registration mechanism (`_pos_enc_hw`, `register_buffer`) is still present in `__init__` and `_get_pos_enc` because they are defined in the class — but `forward` does not call them. The Builder must **not** call `self._get_pos_enc` or add `pos_enc` to `spatial` in Design C's `forward`.

### `loss` and `predict` changes

Identical pattern to Design A and B — extract depth map before calling `forward`:

**In `loss()`**, add before calling `self.forward`:
```python
feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
depth_map = self._extract_depth_map(
    batch_data_samples, feat_h, feat_w, feats[-1].device)
pred = self.forward(feats, depth_map=depth_map)
```

**In `predict()`**, add before calling `self.forward`:
```python
feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
depth_map = self._extract_depth_map(
    batch_data_samples, feat_h, feat_w, feats[-1].device)
pred = self.forward(feats, depth_map=depth_map)
```

All remaining code in `loss()` and `predict()` is unchanged from baseline.

---

## Exact Changes — `config.py`

In the `head` dict inside `model`, add `depth_pos_enc_type='mlp'`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    depth_pos_enc_type='mlp',
),
```

All other config values (LR `1e-4`, weight decay `0.03`, batch size 4, accumulation 8, warmup 3 epochs, cosine LR, data pipeline, hooks, seed 2026) are **identical to baseline**.

---

## Algorithm Summary

| Component | Baseline | Design C |
|---|---|---|
| Positional encoding | fixed `_build_2d_sincos_pos_enc` | learned `pos_mlp(x, y, depth)` |
| `pos_mlp` architecture | — | `Linear(3,64) → GELU → Linear(64,256)` |
| Input coords | — | `(x, y)` in [-1,1]; `depth` in [0,1] |
| Fallback depth | — | 0.5 (mid-range neutral) |
| Extra params | — | `3×64 + 64 + 64×256 + 256` = **16,832** |
| `_get_pos_enc` called? | Yes | **No** |

---

## Parameter Count

- `pos_mlp[0].weight`: `(3, 64)` = 192; bias `(64,)` = 64
- `pos_mlp[2].weight`: `(64, 256)` = 16,384; bias `(256,)` = 256
- **Total new params: 16,896** — negligible.

---

## Constraints and Invariants

1. **Loss restriction**: body joints 0-21 only. Unchanged.
2. **Pelvis pathway**: `decoded[:, 0, :]`. Unchanged.
3. **`_get_pos_enc` NOT called in forward**: Design C replaces the 2D sinusoidal encoding entirely. The `_get_pos_enc` method and `_build_2d_sincos_pos_enc` function remain in the file (they are defined but just not called in `forward`).
4. **`pos_mlp_hidden = 64`**: hardcoded as a local variable in `__init__`, not exposed as a config kwarg. The Builder must not change this value.
5. **x/y normalisation range `[-1, 1]`**: use `torch.linspace(-1.0, 1.0, h)` for both axes. Do not use `[0, h-1]` or `[0, 1]`.
6. **depth normalisation `[0, 1]`**: clamp to `[0, 10]` metres, then divide by 10.
7. **fallback depth = 0.5**: when `depth_map is None`, depth channel is filled with 0.5 (not 0.0 as in Designs A/B). This ensures the fallback produces a non-degenerate positional encoding consistent with a mid-range depth assumption.
8. **`_build_3d_pos_grid` returns `(1, h*w, 3)` when `depth_map is None`**: the `forward` method must call `.expand(B, -1, -1)` before passing to `pos_mlp` when this fallback is triggered.
9. **No `pos_enc` addition in forward**: the line `spatial = spatial + pos_enc` from baseline must **not** appear in this design's `forward`.
10. **persistent_workers=False**: unchanged.
11. **Seed**: `randomness = dict(seed=2026)` unchanged.
12. **No Python imports in config.py**: `depth_pos_enc_type='mlp'` is a string literal.

---

## Expected Behavior

- **At initialization**: `pos_mlp` has near-zero outputs (small `trunc_normal` weights, zero bias, small input magnitudes). Spatial tokens receive near-zero positional signal initially — the model must learn positional structure from scratch.
- **Convergence**: may be slower than baseline at early epochs because the MLP positional encoding starts near zero rather than providing the baseline's meaningful 2D sinusoidal signal. The 3-epoch warmup at lr_factor 0.333 partially mitigates this.
- **During training**: `pos_mlp` jointly encodes 2D position and depth. The GELU nonlinearity allows the MLP to model nonlinear interactions between position and depth (e.g., corners at close range are treated differently than corners at far range).
- **Body MPJPE**: expected improvement ~−10 to −20 mm over baseline at convergence (primary bet among the three designs).
- **Pelvis MPJPE**: expected improvement ~−15 to −25 mm.
- **Composite target**: aim for `composite_val` < 163 (vs. baseline 176.4).
- **Risk**: if the MLP positional encoding does not learn a good 2D layout within the first few epochs, early-epoch performance may be significantly worse than baseline. The training curve may dip before recovering. This is the highest-risk, highest-reward design in this idea.
