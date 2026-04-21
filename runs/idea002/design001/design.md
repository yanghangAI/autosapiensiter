**Design Description:** Decoupled pelvis query using shared decoder layer weights, cross-attention only (no self-attention mixing with joint queries).

**Starting Point:** `baseline/`

---

## Overview

Introduce a dedicated 71st learnable `pelvis_query` embedding that runs through the existing `decoder_layer` (shared weights) via a separate cross-attention-only call. The pelvis depth/UV heads read from this dedicated token instead of from `decoded[:, 0, :]`. Joint queries are completely unchanged.

This is the minimal-change test: zero extra decoder parameters (only one `nn.Embedding(1, hidden_dim)` vector added), purely testing whether removing self-attention mixing is sufficient to recover pelvis accuracy.

**Algorithm:** The core algorithm change is to route the pelvis query through a cross-attention-only sub-call of the existing decoder layer (skipping self-attention) rather than reading from joint query token 0 after the full self+cross-attention decoder pass. This decouples the body-structure task (joint query 0 self-attention) from the absolute-localisation task (pelvis query cross-attention).

---

## Files to Change

1. **`pose3d_transformer_head.py`** — primary change
2. **`config.py`** — add `decouple_pelvis=True` kwarg to head config

`pelvis_utils.py` is **not changed**.

---

## Changes to `pose3d_transformer_head.py`

### 1. Add `decouple_pelvis` constructor parameter

In `Pose3dTransformerHead.__init__`, add parameter:

```
decouple_pelvis: bool = False,
```

Store as `self.decouple_pelvis = decouple_pelvis`.

### 2. Add `pelvis_query` embedding (conditional)

Immediately after `self.joint_queries = nn.Embedding(num_joints, hidden_dim)`, add:

```python
if self.decouple_pelvis:
    self.pelvis_query = nn.Embedding(1, hidden_dim)
```

### 3. Update `_init_head_weights`

Add initialisation for `pelvis_query` when present:

```python
if self.decouple_pelvis and hasattr(self, 'pelvis_query'):
    nn.init.trunc_normal_(self.pelvis_query.weight, std=0.02)
```

This goes inside `_init_head_weights`, alongside the existing `joint_queries` init.

### 4. Update `forward()`

**Current code (lines 244-255):**

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

if self.decouple_pelvis:
    # Run pelvis query through cross-attention only (no self-attn mixing).
    # Reuse the same decoder_layer weights, but skip self-attn by calling
    # cross_attn directly.
    pq = self.pelvis_query.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 1, hidden_dim)
    pq_norm = self.decoder_layer.norm2(pq)
    pq_ca = self.decoder_layer.cross_attn(pq_norm, spatial, spatial)[0]
    pq = pq + self.decoder_layer.dropout2(pq_ca)
    pq_ffn = self.decoder_layer.ffn(self.decoder_layer.norm3(pq))
    pelvis_token = (pq + pq_ffn)[:, 0, :]  # (B, hidden_dim)
else:
    pelvis_token = decoded[:, 0, :]

pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

**Critical implementation notes:**
- The cross-attention call uses `self.decoder_layer.norm2`, `self.decoder_layer.cross_attn`, `self.decoder_layer.dropout2`, `self.decoder_layer.norm3`, `self.decoder_layer.ffn` — all existing attributes, no new modules.
- The pelvis query does NOT participate in `self_attn` over joint queries at all.
- `decoded[:, 0, :]` is still used for `joints` output (unchanged) — `joints_out` still processes all 70 decoded tokens.
- When `decouple_pelvis=False`, behaviour is identical to baseline.

### 5. Docstring update

Update the module-level docstring to note:
- `pelvis_query` (1 × hidden_dim) is optionally used when `decouple_pelvis=True`
- pelvis depth/UV heads read from the dedicated pelvis token in that case

---

## Changes to `config.py`

In the `head` dict under `model`, add one key:

```python
decouple_pelvis=True,
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
- All absolute imports in `pose3d_transformer_head.py` remain absolute (e.g., `from mmpose.models.heads.base_head import BaseHead`).
- Seed: 2026 (already in baseline config, do not change).

---

## Expected Behaviour

- **Parameter delta:** +256 floats (one `nn.Embedding(1, 256)`) — negligible.
- **Memory delta:** < 1 MB on 1080 Ti — no OOM risk.
- **Joint output:** identical to baseline (joint queries unchanged, `decoded[:, 0, :]` still used for `joints_out`).
- **Pelvis output:** sourced from dedicated `pelvis_query` token that attends to spatial features without self-attention contamination from body joint queries.
- **Gradient flow:** `pelvis_query` receives gradients only from `loss_depth` and `loss_uv`, while `joint_queries[0]` receives gradients only from `loss_joints`. Clean separation.
- **Target:** pelvis MPJPE improvement of ~10–15 mm; body MPJPE neutral; composite improvement of ~3–5 points.
