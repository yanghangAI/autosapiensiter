**Design Description:** Decoupled pelvis query with its own independent `_DecoderLayer` (separate weights, cross-attention only — self-attention skipped for single-token case).

**Starting Point:** `baseline/`

---

## Overview

Introduce a dedicated `pelvis_query` embedding (1 × hidden_dim) and a fully independent `pelvis_decoder` (`_DecoderLayer` with its own weights). During forward, the pelvis query runs through `pelvis_decoder` via cross-attention only (self-attention explicitly skipped since a single-token self-attention is a no-op and wastes compute). The joint queries continue through `decoder_layer` exactly as in baseline.

This gives the pelvis pathway complete freedom to develop different cross-attention patterns from the joint pathway — e.g., attending to background depth cues, image-boundary context, or absolute-scale spatial regions.

**Algorithm:** The core algorithm change is to introduce a second decoder pathway with fully independent weights. The pelvis query's cross-attention heads can learn a completely different spatial attention distribution than the joint queries, specialising on absolute-localisation cues rather than body-structure cues. Self-attention is skipped for the pelvis decoder because a single-token sequence has a trivial self-attention (softmax of a scalar = 1.0, so output = input), making it a mathematical no-op that only wastes compute.

---

## Files to Change

1. **`pose3d_transformer_head.py`** — primary change
2. **`config.py`** — add `decouple_pelvis=True`, `pelvis_decoder_type='independent'` to head config

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

### 2. Add `pelvis_query` embedding and `pelvis_decoder` (conditional)

After `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
if self.decouple_pelvis:
    self.pelvis_query = nn.Embedding(1, hidden_dim)
    if self.pelvis_decoder_type == 'independent':
        self.pelvis_decoder = _DecoderLayer(hidden_dim, num_heads, dropout)
```

### 3. Update `_init_head_weights`

Add inside `_init_head_weights`:

```python
if self.decouple_pelvis and hasattr(self, 'pelvis_query'):
    nn.init.trunc_normal_(self.pelvis_query.weight, std=0.02)
    # pelvis_decoder Linear layers are initialised by _DecoderLayer's default
    # (PyTorch default xavier uniform for Linear) — no additional init needed.
```

### 4. Update `forward()`

**Current code (joint decode section, lines 244-255):**

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
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
decoded = self.decoder_layer(queries, spatial)
joints = self.joints_out(decoded)

if self.decouple_pelvis and self.pelvis_decoder_type == 'independent':
    # Run pelvis query through its own decoder — cross-attention only.
    # Self-attention on a single token is a mathematical no-op (softmax over
    # one key collapses to identity), so we skip it explicitly for efficiency.
    pq = self.pelvis_query.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, hidden_dim)
    pq_norm = self.pelvis_decoder.norm2(pq)
    pq_ca = self.pelvis_decoder.cross_attn(pq_norm, spatial, spatial)[0]
    pq = pq + self.pelvis_decoder.dropout2(pq_ca)
    pq_ffn = self.pelvis_decoder.ffn(self.pelvis_decoder.norm3(pq))
    pelvis_token = (pq + pq_ffn)[:, 0, :]  # (B, hidden_dim)
else:
    pelvis_token = decoded[:, 0, :]

pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

**Critical implementation notes:**
- Uses `self.pelvis_decoder.norm2`, `self.pelvis_decoder.cross_attn`, `self.pelvis_decoder.dropout2`, `self.pelvis_decoder.norm3`, `self.pelvis_decoder.ffn` — all from the separate `pelvis_decoder` instance, NOT from `decoder_layer`.
- `self_attn`, `norm1`, `dropout1` of `pelvis_decoder` are defined but intentionally not called — this is correct and safe; they will not receive gradients but the module still registers correctly with PyTorch.
- When `decouple_pelvis=False` or `pelvis_decoder_type != 'independent'`, falls back to `decoded[:, 0, :]` (baseline behaviour).
- `joints` output and `joints_out` are entirely unchanged — all 70 decoded tokens from `decoder_layer` are still used.

### 5. Module-level docstring update

Update to reflect optional `pelvis_decoder` and `pelvis_query`.

---

## Changes to `config.py`

In the `head` dict under `model`, add two keys:

```python
decouple_pelvis=True,
pelvis_decoder_type='independent',
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

---

## Expected Behaviour

- **Parameter delta:** +256 (pelvis_query embedding) + full `_DecoderLayer` (≈ hidden_dim × (4×hidden_dim + hidden_dim) × 4 + norms ≈ 530K params for hidden_dim=256, num_heads=8). Total < 5 MB on 1080 Ti — no OOM risk.
- **Memory delta:** negligible.
- **Joint output:** identical to baseline (joint decoder pathway unchanged).
- **Pelvis output:** dedicated `pelvis_decoder` with independent weights; the cross-attention heads can learn different spatial attention patterns than joint queries (e.g., attend to image boundaries, depth-scale cues, background).
- **Gradient flow:** clean separation — `pelvis_decoder` and `pelvis_query` receive gradients only via depth/UV losses; `decoder_layer` and `joint_queries` receive gradients only via joint loss.
- **Target:** pelvis MPJPE improvement of ~15–20 mm; body MPJPE neutral; composite improvement of ~5–7 points.
