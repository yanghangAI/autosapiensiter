**Design Description:** Decoupled pelvis query with independent decoder + global depth-context token (mean-pooled spatial features projected to hidden_dim, prepended to spatial sequence before pelvis cross-attention).

**Starting Point:** `baseline/`

---

## Overview

Build on Design B (independent pelvis decoder) and additionally fuse a global depth-context token into the spatial sequence before pelvis cross-attention. The global token is computed by mean-pooling `spatial` (after `input_proj` and positional encoding) over the spatial dimension, then projecting through a `depth_proj: nn.Linear(hidden_dim, hidden_dim)`. This scalar summary token is prepended to `spatial` to form `spatial_with_depth = cat([global_token, spatial], dim=1)` (shape `(B, H*W+1, hidden_dim)`). Only the pelvis cross-attention sees this augmented sequence; joint queries use the original `spatial`.

With RGBD input, the backbone feature map `feats[-1]` encodes concatenated RGB+D signal. The mean-pooled global token aggregates scale-relevant information across the entire spatial grid, giving the pelvis decoder a compact anchor for absolute depth inference.

**Algorithm:** The core algorithm change extends Design B by injecting a global context token as an additional key/value entry in the pelvis cross-attention sequence. By mean-pooling the full spatial feature map and projecting it to hidden_dim, we create a "global summary" token that captures aggregate depth-scale information from the entire image. The pelvis decoder can attend to this token with high weight when predicting absolute depth, effectively giving it access to a global scale anchor that spatially-local tokens do not provide. The joint pathway is unaffected (it attends to the unmodified `spatial` sequence).

---

## Files to Change

1. **`pose3d_transformer_head.py`** — primary change
2. **`config.py`** — add `decouple_pelvis=True`, `pelvis_decoder_type='depth_fused'` to head config

`pelvis_utils.py` is **not changed**.

---

## Changes to `pose3d_transformer_head.py`

### 1. Add constructor parameters

In `Pose3dTransformerHead.__init__`, add two new parameters after `dropout`:

```
decouple_pelvis: bool = False,
pelvis_decoder_type: str = 'shared',
```

Store as:
```python
self.decouple_pelvis = decouple_pelvis
self.pelvis_decoder_type = pelvis_decoder_type
```

### 2. Add modules (conditional on `decouple_pelvis=True` and `pelvis_decoder_type='depth_fused'`)

After `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
if self.decouple_pelvis:
    self.pelvis_query = nn.Embedding(1, hidden_dim)
    if self.pelvis_decoder_type in ('independent', 'depth_fused'):
        self.pelvis_decoder = _DecoderLayer(hidden_dim, num_heads, dropout)
    if self.pelvis_decoder_type == 'depth_fused':
        self.depth_proj = nn.Linear(hidden_dim, hidden_dim)
```

**Exact module names:** `self.pelvis_query`, `self.pelvis_decoder`, `self.depth_proj`.

### 3. Update `_init_head_weights`

Add inside `_init_head_weights`:

```python
if self.decouple_pelvis and hasattr(self, 'pelvis_query'):
    nn.init.trunc_normal_(self.pelvis_query.weight, std=0.02)
if hasattr(self, 'depth_proj'):
    nn.init.trunc_normal_(self.depth_proj.weight, std=0.02)
    if self.depth_proj.bias is not None:
        nn.init.zeros_(self.depth_proj.bias)
```

### 4. Update `forward()`

**Current joint+pelvis decode block (lines 244–255):**

```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
decoded = self.decoder_layer(queries, spatial)
joints = self.joints_out(decoded)
pelvis_token = decoded[:, 0, :]
pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

**Replace with:**

```python
# Joint pathway — unchanged
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
decoded = self.decoder_layer(queries, spatial)
joints = self.joints_out(decoded)

# Pelvis pathway
if self.decouple_pelvis and self.pelvis_decoder_type == 'depth_fused':
    # Build global depth-context token by mean-pooling spatial features.
    # spatial shape: (B, H*W, hidden_dim) — already projected and pos-encoded.
    global_depth = spatial.mean(dim=1, keepdim=True)       # (B, 1, hidden_dim)
    global_depth = self.depth_proj(global_depth)            # (B, 1, hidden_dim)

    # Prepend global token to spatial for pelvis cross-attention only.
    spatial_with_depth = torch.cat([global_depth, spatial], dim=1)  # (B, H*W+1, hidden_dim)

    # Pelvis decoder — cross-attention only (skip self-attn for single token).
    pq = self.pelvis_query.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, hidden_dim)
    pq_norm = self.pelvis_decoder.norm2(pq)
    pq_ca = self.pelvis_decoder.cross_attn(pq_norm, spatial_with_depth, spatial_with_depth)[0]
    pq = pq + self.pelvis_decoder.dropout2(pq_ca)
    pq_ffn = self.pelvis_decoder.ffn(self.pelvis_decoder.norm3(pq))
    pelvis_token = (pq + pq_ffn)[:, 0, :]  # (B, hidden_dim)

else:
    pelvis_token = decoded[:, 0, :]

pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

**Critical implementation notes:**

- `spatial` (post `input_proj` + positional encoding, shape `(B, H*W, hidden_dim)`) is the input to mean-pooling. It must be computed **before** this block and stored in the local variable `spatial` (already done in baseline `forward()`).
- `spatial` is **not modified** for the joint pathway — joint queries still see the original `spatial` without the global token.
- `spatial_with_depth` is a new local variable; it is only used in the pelvis cross-attention call.
- `self.depth_proj` applies a learned linear transformation to the mean-pooled token — this allows the projection to gate which channels of the spatial mean are useful for depth anchoring.
- When `decouple_pelvis=False`, falls back to baseline `decoded[:, 0, :]`.
- `joints_out`, `loss()`, and `predict()` are not touched.

### 5. Module-level docstring update

Update to reflect `depth_proj` and the `spatial_with_depth` augmentation.

---

## Changes to `config.py`

In the `head` dict under `model`, add two keys:

```python
decouple_pelvis=True,
pelvis_decoder_type='depth_fused',
```

All other head parameters remain identical to baseline:
- `in_channels=1024`
- `hidden_dim=256`
- `num_joints=70`
- `num_heads=8`
- `dropout=0.1`
- `loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0)`
- `loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0)`
- `loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0)`
- `loss_weight_depth=1.0`
- `loss_weight_uv=1.0`

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, accumulation) are **identical to baseline**.

---

## Invariants to Preserve

- `persistent_workers=False` in both dataloaders — do not change.
- Loss restricted to body joints (indices 0–21): `pred['joints'][:, _BODY]` — unchanged.
- `loss()` method signature and return type `(losses dict, pred dict)` — unchanged.
- `predict()` method reads `pred['pelvis_depth']` and `pred['pelvis_uv']` from `forward()` — these keys must remain in the returned dict.
- No Python `import` statements in `config.py` — only `__import__()` calls.
- All absolute imports in `pose3d_transformer_head.py` remain absolute.
- Seed: 2026 — do not change.
- The `spatial` variable used by joint queries must be the **original** (without `global_depth` prepended).

---

## Expected Behaviour

- **Parameter delta:** +256 (pelvis_query) + full `_DecoderLayer` (≈ 530K params) + `depth_proj` (256×256 + 256 = 65,792 params). Total < 6 MB — no OOM risk.
- **Memory delta:** negligible; `spatial_with_depth` tensor is `(B, H*W+1, hidden_dim)` vs `(B, H*W, hidden_dim)` — one extra row per batch.
- **Joint output:** identical to baseline.
- **Pelvis output:** dedicated decoder with global depth-context token prepended; cross-attention can place high weight on the global token to read scale information, or ignore it and attend to local spatial features — learned end-to-end.
- **Gradient flow:** `depth_proj` and `pelvis_decoder` and `pelvis_query` receive gradients only via depth/UV losses. `input_proj` and `joint_queries` and `decoder_layer` receive gradients via joint loss (and `input_proj` additionally through `spatial` which is also the input to `depth_proj`, so `input_proj` receives gradients from both pathways — this is expected and desirable).
- **Target:** pelvis MPJPE improvement of ~15–25 mm; body MPJPE neutral; composite improvement of ~5–8 points.
