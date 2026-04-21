**Design Description:** Depth sinusoidal encoding per spatial token — 1D sin/cos encoding of scalar depth value concatenated with 2D sinusoidal pos enc, projected back to `hidden_dim` via a small linear.

**Starting Point:** `baseline/`

---

## Overview

Instead of a single learned linear mapping (Design A), this design applies a sinusoidal frequency decomposition to the depth scalar at each spatial token — analogous to how the 2D positional encoding encodes x and y. The resulting `hidden_dim//2`-dimensional depth sinusoidal vector is concatenated with the existing 2D sinusoidal positional encoding (`hidden_dim`-dimensional), and a small `nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)` projects the concatenated signal back to `hidden_dim`:

```
depth_sine  = build_1d_sincos_enc(depth_grid, hidden_dim // 2)    # (B, H'*W', 128)
pos_concat  = concat([2d_sincos_pos_enc.expand(B,-1,-1), depth_sine], dim=-1)  # (B, H'*W', 384)
pos_embed   = depth_pos_proj(pos_concat)                           # (B, H'*W', 256)
spatial     = input_proj(feat) + pos_embed
```

This provides a frequency-rich geometric encoding of depth that generalises better to unseen depth values without learning a dense mapping per frequency bin.

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `_build_1d_sincos_enc` helper function, add `depth_pos_proj` linear, add `_extract_depth_map` helper method, update `__init__`, `forward`, `loss`, `predict`.
2. `config.py` — add `depth_pos_enc_type='sinusoidal'` kwarg to the head config dict.

`pelvis_utils.py` is **not** modified.

---

## Exact Changes — `pose3d_transformer_head.py`

### New imports

Add to existing imports at the top:

```python
import numpy as np
import torch.nn.functional as F
```

### New module-level helper function `_build_1d_sincos_enc`

Add this function after the existing `_build_2d_sincos_pos_enc` function:

```python
def _build_1d_sincos_enc(
    depth_flat: torch.Tensor, embed_dim: int
) -> torch.Tensor:
    """Build 1D sinusoidal encoding for scalar depth values.

    Args:
        depth_flat: ``(B, N, 1)`` depth values, normalised to [0, 1].
        embed_dim: Output dimension (must be even).

    Returns:
        ``(B, N, embed_dim)`` sinusoidal encoding.
    """
    assert embed_dim % 2 == 0, f'embed_dim must be even, got {embed_dim}'
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32, device=depth_flat.device) / half
    omega = 1.0 / (10000.0 ** omega)  # (half,)
    # depth_flat: (B, N, 1); omega: (half,)
    angles = depth_flat * omega.unsqueeze(0).unsqueeze(0)  # (B, N, half)
    enc = torch.cat([angles.sin(), angles.cos()], dim=-1)   # (B, N, embed_dim)
    return enc
```

### `__init__` signature change

Add `depth_pos_enc_type: str = 'sinusoidal'` as a new parameter after `loss_weight_uv` and before `init_cfg`:

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
    depth_pos_enc_type: str = 'sinusoidal',
    init_cfg: OptConfigType = None,
):
```

### `__init__` additions

After the line `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
# Depth-aware spatial positional encoding (Design B: sinusoidal)
self.depth_pos_enc_type = depth_pos_enc_type
# Projects [2d_sincos (hidden_dim) || depth_sine (hidden_dim//2)] → hidden_dim
depth_enc_in_dim = hidden_dim + hidden_dim // 2   # 256 + 128 = 384
self.depth_pos_proj = nn.Linear(depth_enc_in_dim, hidden_dim)
# Near-identity init: the 2D-sincos component should dominate at start.
# Use trunc_normal for the weight; zero bias so the combined signal
# is initially a linear projection of the 2D-only pos enc.
nn.init.trunc_normal_(self.depth_pos_proj.weight, std=0.02)
nn.init.zeros_(self.depth_pos_proj.bias)
```

**Initialisation rationale**: `trunc_normal_(std=0.02)` matches the standard initialisation used for projection layers throughout the head. The projection of `(2d_sincos || 0)` (depth=0 initially after normalisation) through a `trunc_normal` weight gives a noisy but reasonable starting positional signal. This differs from Design A's strict zero-init because the 2D sinusoidal information must survive the projection.

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
        ``(B, 1, target_h, target_w)`` float tensor with raw depth values
        in metres. Zero-filled for samples with missing depth data.
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

# Build combined positional encoding: 2D sincos + depth sinusoidal
pos_enc_2d = self._get_pos_enc(H, W, feat.device)  # (1, H*W, hidden_dim)

if depth_map is not None:
    # Normalise depth: clamp [0, 10] m → [0, 1]
    depth_flat = depth_map.flatten(2).transpose(1, 2)          # (B, H*W, 1)
    depth_flat = depth_flat.clamp(min=0.0, max=10.0) / 10.0
    depth_sine = _build_1d_sincos_enc(depth_flat, self.hidden_dim // 2)  # (B, H*W, 128)
    # Expand 2D pos enc to batch, concat with depth sine
    pos_2d_expanded = pos_enc_2d.expand(B, -1, -1)              # (B, H*W, 256)
    pos_concat = torch.cat([pos_2d_expanded, depth_sine], dim=-1)  # (B, H*W, 384)
    pos_embed = self.depth_pos_proj(pos_concat)                  # (B, H*W, 256)
else:
    # Fallback: project 2D pos enc alone through a zero-padded input
    # (pad depth_sine with zeros, maintaining same code path)
    pos_2d_expanded = pos_enc_2d.expand(B, -1, -1)              # (B, H*W, 256)
    depth_sine_zero = torch.zeros(
        B, H * W, self.hidden_dim // 2, device=feat.device)    # (B, H*W, 128)
    pos_concat = torch.cat([pos_2d_expanded, depth_sine_zero], dim=-1)  # (B, H*W, 384)
    pos_embed = self.depth_pos_proj(pos_concat)                  # (B, H*W, 256)

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

**Note**: the fallback path (depth_map=None) still sends the 2D pos enc through `depth_pos_proj` with a zero-padded depth dimension. This means the positional signal is always projected through `depth_pos_proj`, which learns to reconstruct a useful 2D encoding even from the non-depth portion. The Builder must implement exactly this unified code path.

### `loss` and `predict` changes

Identical pattern to Design A — extract depth map before calling `forward`:

**In `loss()`**, add before `pred = self.forward(feats)`:
```python
feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
depth_map = self._extract_depth_map(
    batch_data_samples, feat_h, feat_w, feats[-1].device)
pred = self.forward(feats, depth_map=depth_map)
```

**In `predict()`**, add before `pred = self.forward(feats)`:
```python
feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
depth_map = self._extract_depth_map(
    batch_data_samples, feat_h, feat_w, feats[-1].device)
pred = self.forward(feats, depth_map=depth_map)
```

All remaining code in `loss()` and `predict()` is unchanged from baseline.

---

## Exact Changes — `config.py`

In the `head` dict inside `model`, add `depth_pos_enc_type='sinusoidal'`:

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
    depth_pos_enc_type='sinusoidal',
),
```

All other config values (LR `1e-4`, weight decay `0.03`, batch size 4, accumulation 8, warmup 3 epochs, cosine LR, data pipeline, hooks, seed 2026) are **identical to baseline**.

---

## Algorithm Summary

| Component | Baseline | Design B |
|---|---|---|
| Spatial pos enc | `2d_sincos` (fixed, 256-d) | `depth_pos_proj(concat[2d_sincos, depth_sine])` |
| `depth_sine` | — | `_build_1d_sincos_enc(depth, 128)` — sin/cos at 64 freqs |
| `depth_pos_proj` | — | `nn.Linear(384, 256)`, trunc_normal weight, zero bias |
| Extra params | — | 384×256 + 256 = 98,560 + 256 ≈ **98.8 K** |
| Depth range | — | clamp [0,10] m / 10 → [0,1] |

---

## Parameter Count

- `depth_pos_proj.weight`: `(384, 256)` = 98,304 params
- `depth_pos_proj.bias`: `(256,)` = 256 params
- **Total new params: 98,560** — <0.1% overhead on a 300M parameter backbone. Negligible.

---

## Constraints and Invariants

1. **Loss restriction**: body joints 0-21 only. Unchanged.
2. **Pelvis pathway**: `decoded[:, 0, :]`. Unchanged.
3. **`_get_pos_enc` unchanged**: the `_build_2d_sincos_pos_enc` function and `_get_pos_enc` method are **not modified**. Design B calls `_get_pos_enc` and then passes its output into the concat+project pipeline.
4. **`hidden_dim // 2` = 128 for default `hidden_dim=256`**: `_build_1d_sincos_enc` is called with `embed_dim=128`. The Builder must use `self.hidden_dim // 2` (not hardcoded 128) so it scales with any non-default `hidden_dim`.
5. **`depth_enc_in_dim = hidden_dim + hidden_dim // 2`**: `depth_pos_proj` input dim must be computed as `hidden_dim + hidden_dim // 2` in `__init__`, not hardcoded as 384.
6. **Fallback is not a skip**: when `depth_map is None`, the code still passes a zero-padded concatenation through `depth_pos_proj`. This ensures `depth_pos_proj` always sits in the computational graph and avoids conditional execution paths that would make the frozen 2D positional encoding invisible to gradients.
7. **persistent_workers=False**: unchanged.
8. **Seed**: `randomness = dict(seed=2026)` unchanged.
9. **No Python imports in config.py**: all values are string/float literals.
10. **Absolute imports in head file**: already satisfied by baseline structure.

---

## Expected Behavior

- **At initialization**: `depth_pos_proj` has `trunc_normal` weights and zero bias. The spatial tokens start with a learned projection of the 2D sinusoidal encoding (no depth). The network adapts from this starting state.
- **During training**: the sinusoidal depth encoding provides multi-frequency depth signal; `depth_pos_proj` learns to combine 2D position and depth geometry into a single positional embedding.
- **Geometric generalisation**: sinusoidal encoding is continuous and generalises to unseen depth values (unlike a fully learned discrete mapping). This is particularly useful for BEDLAM2 where depth range at test time may differ from training.
- **Body MPJPE**: expected improvement ~−8 to −15 mm.
- **Pelvis MPJPE**: expected improvement ~−10 to −20 mm.
- **Composite target**: aim for `composite_val` < 165 (vs. baseline 176.4).
