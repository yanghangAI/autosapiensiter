**Design Description:** Scalar depth per spatial token — single `nn.Linear(1, hidden_dim)` projects downsampled depth value at each token location, added to spatial tokens alongside 2D sinusoidal positional encoding.

**Starting Point:** `baseline/`

---

## Overview

This is the minimal-change ablation for depth-aware spatial positional encoding. One extra linear layer (`depth_proj: nn.Linear(1, hidden_dim)`) projects the raw depth scalar at each spatial token location to `hidden_dim` dimensions. The resulting per-token depth embedding is added to the spatial tokens after the 2D sinusoidal positional encoding:

```
spatial = input_proj(feat) + 2d_sincos_pos_enc + depth_proj(depth_grid)
```

where `depth_grid` is the bilinearly-downsampled raw depth map reshaped to `(B, H'*W', 1)`.

The depth map is extracted from `batch_data_samples` inside `loss()` and `predict()` and passed as a keyword argument `depth_map` to `forward()`. When `depth_map` is `None` (fallback), `depth_proj` contributes zero (via zero-tensor input, not a special branch — see below).

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `depth_proj` linear, refactor `loss()` and `predict()` to extract depth and pass to `forward()`, update `forward()` signature and body.
2. `config.py` — add `depth_pos_enc_type='linear'` kwarg to the head config dict.

`pelvis_utils.py` is **not** modified.

---

## Exact Changes — `pose3d_transformer_head.py`

### New import

Add to the existing imports at the top:

```python
import numpy as np
import torch.nn.functional as F
```

`numpy` is already used in `pelvis_utils.py` but the head itself does not import it. The depth map is stored as a numpy array (NPZ file) in the data sample's metainfo; we need numpy to load it and `F.interpolate` to resize.

### `__init__` signature change

Add `depth_pos_enc_type: str = 'linear'` as a new parameter after `loss_weight_uv` and before `init_cfg`:

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
    depth_pos_enc_type: str = 'linear',
    init_cfg: OptConfigType = None,
):
```

### `__init__` additions

After the line `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
# Depth-aware spatial positional encoding (Design A: scalar linear)
self.depth_pos_enc_type = depth_pos_enc_type
self.depth_proj = nn.Linear(1, hidden_dim)
# Zero-init bias; small-scale weight so depth signal starts near zero
nn.init.zeros_(self.depth_proj.weight)
nn.init.zeros_(self.depth_proj.bias)
```

**Why zero-init**: this ensures that at the start of training, `depth_proj(depth_grid)` outputs an all-zero tensor, making the model exactly equivalent to the baseline at epoch 0. The network can then learn non-zero weights as training progresses.

### New helper method `_extract_depth_map`

Add this method to `Pose3dTransformerHead` (before `forward`):

```python
def _extract_depth_map(
    self,
    batch_data_samples: OptSampleList,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract, load, and resize raw depth maps from batch data samples.

    Reads the ``depth_npy_path`` field from each sample's metainfo,
    loads the NPZ/NPY file, crops to ``img_shape`` (the crop region), and
    bilinearly resizes to ``(target_h, target_w)``.

    Args:
        batch_data_samples: List of data samples from the dataloader.
        target_h: Target height (feature map H').
        target_w: Target width (feature map W').
        device: Device to place the output tensor on.

    Returns:
        ``(B, 1, target_h, target_w)`` float tensor with raw depth values
        in metres. Values are **not** normalised here — normalisation is
        done in ``forward()``.
        If any sample is missing depth data, that sample's depth map is
        filled with zeros (graceful degradation).
    """
    depth_maps = []
    for ds in batch_data_samples:
        depth_npy_path = ds.metainfo.get('depth_npy_path', None)
        img_shape = ds.metainfo.get('img_shape', None)  # (crop_h, crop_w)
        try:
            raw = np.load(depth_npy_path)
            if isinstance(raw, np.lib.npyio.NpzFile):
                # NPZ: depth stored under key 'depth' or first key
                key = 'depth' if 'depth' in raw else list(raw.keys())[0]
                raw = raw[key]
            # raw is (H_full, W_full) or (1, H_full, W_full)
            if raw.ndim == 3:
                raw = raw[0]
            # Crop to img_shape if provided (crop is top-left aligned)
            if img_shape is not None:
                ch, cw = int(img_shape[0]), int(img_shape[1])
                raw = raw[:ch, :cw]
            depth_tensor = torch.from_numpy(raw.astype(np.float32))  # (ch, cw)
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)    # (1, 1, ch, cw)
        except Exception:
            # Graceful fallback: zero depth
            if img_shape is not None:
                ch, cw = int(img_shape[0]), int(img_shape[1])
            else:
                ch, cw = target_h, target_w
            depth_tensor = torch.zeros(1, 1, ch, cw, dtype=torch.float32)
        depth_maps.append(depth_tensor)

    # Resize each to (target_h, target_w) and stack
    resized = []
    for d in depth_maps:
        r = F.interpolate(d, size=(target_h, target_w), mode='bilinear',
                          align_corners=False)  # (1, 1, target_h, target_w)
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

Inside `forward`, after `spatial = spatial + pos_enc`, add the depth positional signal:

```python
feat = feats[-1]  # (B, C, H, W)
B, C, H, W = feat.shape

# Flatten spatial dims, project to hidden_dim, add positional encoding
spatial = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
spatial = self.input_proj(spatial)          # (B, H*W, hidden_dim)
pos_enc = self._get_pos_enc(H, W, feat.device)
spatial = spatial + pos_enc

# Depth-aware positional signal
if depth_map is not None:
    # depth_map: (B, 1, H, W) — already resized to feature map resolution
    # Flatten to (B, H*W, 1) and normalise to [0, 1] using a soft clamp
    depth_flat = depth_map.flatten(2).transpose(1, 2)  # (B, H*W, 1)
    # Soft normalise: clamp to [0, 10] metres then divide by 10
    depth_flat = depth_flat.clamp(min=0.0, max=10.0) / 10.0
    depth_enc = self.depth_proj(depth_flat)             # (B, H*W, hidden_dim)
    spatial = spatial + depth_enc

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

**Depth normalisation rationale**: BEDLAM2 depth values range from ~0.5 to ~10 metres in typical scenes. Clamping to [0, 10] and dividing by 10 maps the range to [0, 1], which is a stable input range for the linear layer. This is a fixed normalisation, not learned.

### `loss` method change

In `loss()`, before `pred = self.forward(feats)`, extract the depth map and pass it to `forward`:

```python
def loss(
    self,
    feats: Tuple[torch.Tensor, ...],
    batch_data_samples: OptSampleList,
    train_cfg: ConfigType = {},
) -> Dict[str, torch.Tensor]:
    feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
    depth_map = self._extract_depth_map(
        batch_data_samples, feat_h, feat_w, feats[-1].device)
    pred = self.forward(feats, depth_map=depth_map)
    # ... rest of loss() is UNCHANGED from baseline ...
```

All downstream code in `loss()` (GT extraction, loss computation, `_train_mpjpe` tracking) remains identical to baseline.

### `predict` method change

In `predict()`, before `pred = self.forward(feats)`, extract the depth map:

```python
def predict(
    self,
    feats: Tuple[torch.Tensor, ...],
    batch_data_samples: OptSampleList,
    test_cfg: ConfigType = {},
) -> Predictions:
    feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
    depth_map = self._extract_depth_map(
        batch_data_samples, feat_h, feat_w, feats[-1].device)
    pred = self.forward(feats, depth_map=depth_map)
    # ... rest of predict() is UNCHANGED from baseline ...
```

---

## Exact Changes — `config.py`

In the `head` dict inside `model`, add `depth_pos_enc_type='linear'`:

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
    depth_pos_enc_type='linear',
),
```

All other config values (LR `1e-4`, weight decay `0.03`, batch size 4, accumulation 8, warmup 3 epochs, cosine LR, data pipeline, hooks, seed 2026) are **identical to baseline**.

---

## Algorithm Summary

| Component | Baseline | Design A |
|---|---|---|
| Spatial token construction | `input_proj(feat) + 2d_sincos` | `input_proj(feat) + 2d_sincos + depth_proj(depth_grid)` |
| Depth projection | — | `nn.Linear(1, 256)`, zero-init |
| Depth normalisation | — | clamp [0,10] / 10 → [0,1] |
| Extra params | — | 256 + 256 = 512 (weight + bias) |
| Depth source | — | `depth_npy_path` in metainfo, loaded via numpy |
| Fallback (no depth) | — | zero tensor → zero additive signal |

---

## Parameter Count

- `depth_proj.weight`: `(1, 256)` = 256 params
- `depth_proj.bias`: `(256,)` = 256 params
- **Total new params: 512** — negligible overhead.

---

## Constraints and Invariants

1. **Loss restriction**: joint loss stays on body joints indices 0-21 (`_BODY = list(range(0, 22))`). Unchanged.
2. **Pelvis pathway**: `pelvis_depth` and `pelvis_uv` still from `decoded[:, 0, :]`. Unchanged.
3. **Zero-init**: `depth_proj.weight` and `depth_proj.bias` must be zero-initialised so the model is functionally equivalent to baseline at epoch 0.
4. **Depth normalisation**: clamp to [0, 10] m, divide by 10. This is a hardcoded normalisation, not a learnable parameter.
5. **Graceful fallback**: if `depth_npy_path` is missing or loading fails, the helper fills with zeros. The zero input to `depth_proj` produces a zero output (via zero-init bias), so the fallback is exactly baseline.
6. **`depth_map` kwarg default = None**: `forward()` works correctly when called without `depth_map` (e.g., from external callers that do not pass depth). If `depth_map is None`, the depth branch is skipped entirely.
7. **persistent_workers=False**: unchanged.
8. **Seed**: `randomness = dict(seed=2026)` unchanged.
9. **No new top-level imports in config.py**: `depth_pos_enc_type='linear'` is a string literal.
10. **No change to `_init_head_weights`**: `depth_proj` is initialised inline in `__init__`, not via `_init_head_weights`. The Builder must not add `depth_proj` to the `_init_head_weights` loop.
11. **MMEngine invariants**: no Python `import` statements in `config.py`. The head file uses absolute imports. Both are already satisfied.
12. **Feature map H', W' values**: for Sapiens 0.3B with `img_size=(640, 384)` and patch size 16, `H'=640/16=40`, `W'=384/16=24`. Depth map extracted from the crop (640×384) and resized via `F.interpolate` to (40, 24).

---

## Expected Behavior

- **At initialization**: depth contribution is exactly zero (zero-init). Model equals baseline.
- **During training**: `depth_proj` learns to map scalar depth → `hidden_dim` vector that modifies each spatial token's positional embedding.
- **Body MPJPE**: moderate improvement (~−5 to −10 mm) from better per-token depth context.
- **Pelvis MPJPE**: expected improvement (~−10 to −20 mm) because pelvis query can find depth-relevant spatial tokens more directly.
- **Composite target**: aim for `composite_val` < 168 (vs. baseline 176.4). Design A is the cheapest test; substantial gain not guaranteed.
- **Training speed**: one extra `np.load` per sample per iteration. Since data is already memory-mapped in the pipeline (depth NPZ is loaded during `LoadBedlamLabels`), re-loading in the head is redundant. **Important note for Builder**: the depth map is available in `batch_data_samples[i].metainfo['depth_npy_path']`; if the pipeline already stores the loaded depth array under a different key (e.g., `metainfo['depth_map']` as a pre-loaded tensor), use that instead to avoid double I/O. If no cached array is available, load from path as specified above.
