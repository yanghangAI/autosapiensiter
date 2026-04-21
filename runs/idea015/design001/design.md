**Design Description:** K=32 super-tokens via single slot-attention layer replacing full 960-token cross-attention.

**Starting Point:** `baseline/`

---

## Overview

Replace the flat cross-attention over all 960 spatial tokens with a two-stage mechanism: a single `nn.MultiheadAttention` slot-attention layer compresses the 960 spatial tokens into K=32 learned "super-tokens", and then the existing `_DecoderLayer` cross-attends over these 32 super-tokens instead of 960. This is the minimal diagnostic design — one decoder layer, no positional slot init, K=32.

**Algorithm:** K=32 learnable slot query vectors (an `nn.Embedding(32, 256)`) cross-attend over the 960 spatial tokens via a `nn.MultiheadAttention(256, num_heads=8, batch_first=True)` with pre-norm LayerNorm applied to the slot queries. The resulting K=32 super-tokens serve as keys/values for the joint query cross-attention inside `_DecoderLayer`. The slot attention is computed once per forward pass before the decoder layer. Output tensor shapes from the head are unchanged.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Constructor signature** — add the following new parameters after `dropout`, before `loss_joints`:

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_super_tokens: int = 0,       # NEW: 0 = disabled (baseline), >0 enables slot pooling
    slot_pos_init: bool = False,     # NEW: whether to init slot queries from positional encodings
    num_decoder_layers: int = 1,     # NEW: number of decoder layers
    aux_loss_weight: float = 0.0,    # NEW: weight for auxiliary intermediate decoder losses
    loss_joints: ConfigType = ...,
    ...
):
```

Store these:
```python
self.num_super_tokens = num_super_tokens
self.slot_pos_init = slot_pos_init
self.num_decoder_layers = num_decoder_layers
self.aux_loss_weight = aux_loss_weight
```

**New modules in `__init__`** — add the following block after the existing `self.joint_queries = nn.Embedding(...)` line, before the decoder layer:

```python
# Slot-attention pooling (enabled when num_super_tokens > 0)
if self.num_super_tokens > 0:
    self.slot_queries = nn.Embedding(num_super_tokens, hidden_dim)
    self.slot_attn = nn.MultiheadAttention(
        hidden_dim, num_heads, dropout=dropout, batch_first=True)
    self.slot_norm = nn.LayerNorm(hidden_dim)
```

**Decoder layers** — replace:
```python
# Transformer decoder (1 layer)
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```
With:
```python
# Transformer decoder (N layers, default 1)
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Note: the attribute `self.decoder_layer` (singular) is removed. All references to `self.decoder_layer` in `forward()` must be updated to use `self.decoder_layers`.

**`_init_head_weights()` changes** — add after the existing output projection init:
```python
if self.num_super_tokens > 0:
    nn.init.trunc_normal_(self.slot_queries.weight, std=0.02)
    # slot_pos_init is handled separately (Design B only; False here)
```

**`forward()` changes** — after computing `spatial` tokens (input projection + positional encoding), insert the slot-attention pooling step:

```python
# --- existing code up to this point ---
spatial = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
spatial = self.input_proj(spatial)          # (B, H*W, hidden_dim)
pos_enc = self._get_pos_enc(H, W, feat.device)
spatial = spatial + pos_enc

# --- NEW: slot-attention pooling ---
if self.num_super_tokens > 0:
    S = self.slot_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, hidden_dim)
    S_normed = self.slot_norm(S)
    super_tokens, _ = self.slot_attn(S_normed, spatial, spatial)  # (B, K, hidden_dim)
    spatial_for_decoder = super_tokens
else:
    spatial_for_decoder = spatial  # (B, H*W, hidden_dim) — baseline path

# --- existing decoder, updated to loop over layers ---
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

decoded = queries
intermediate_outputs = []
for layer in self.decoder_layers:
    decoded = layer(decoded, spatial_for_decoder)
    intermediate_outputs.append(decoded)

# Output projections (unchanged interface)
joints = self.joints_out(decoded)       # (B, num_joints, 3)
pelvis_token = decoded[:, 0, :]         # (B, hidden_dim)
pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
pelvis_uv = self.uv_out(pelvis_token)   # (B, 2)
```

**`loss()` changes** — for this design (`aux_loss_weight=0.0`), no changes to the loss body are required. However, the `loss()` method calls `self.forward(feats)` which now returns the same dict as before. The `loss()` method remains identical to baseline except it must call the updated `forward()`. No auxiliary loss computation is needed when `aux_loss_weight == 0.0`.

**`predict()` — no changes.** Still calls `self.forward(feats)` and reads the same output dict keys.

### 2. `config.py`

In the `head=dict(...)` block, add the four new parameters as int/float/bool literals:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_super_tokens=32,
    slot_pos_init=False,
    num_decoder_layers=1,
    aux_loss_weight=0.0,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values remain identical to baseline:
- LR: 1e-4 (head), 1e-5 (backbone, via `lr_mult=0.1`)
- Weight decay: 0.03
- Warmup: 3 epochs linear (`start_factor=0.333`), then cosine to epoch 20
- Batch: 4, accum: 8, effective batch: 32
- Seed: 2026
- `persistent_workers=False` in both dataloaders

---

## Constraints and Invariants to Preserve

1. **Joint loss scope:** restrict to `_BODY = list(range(0, 22))` — indices 0–21 only. Unchanged from baseline.
2. **Pelvis token:** still `decoded[:, 0, :]` from the final decoder layer output.
3. **`persistent_workers=False`** in both dataloaders — do not change.
4. **No Python `import` statements in `config.py`** — use `__import__()` or hardcode literals. `num_super_tokens=32`, `num_decoder_layers=1`, `aux_loss_weight=0.0`, `slot_pos_init=False` are all literals.
5. **Absolute imports in `pose3d_transformer_head.py`** — keep as-is (e.g. `from mmpose.models.heads.base_head import BaseHead`).
6. **`_DecoderLayer` class itself is not modified.**
7. **`default_init_cfg`** returns `[]` — do not change.
8. **Output tensor shapes unchanged:** `joints` is `(B, 70, 3)`, `pelvis_depth` is `(B, 1)`, `pelvis_uv` is `(B, 2)`.
9. **Slot attention uses `batch_first=True`** — consistent with `_DecoderLayer`'s existing MultiheadAttention.
10. **`slot_attn` query is pre-normed** (`self.slot_norm` applied to slot queries before cross-attending over spatial) — this is a pre-norm design consistent with the existing `_DecoderLayer` which uses pre-norm (`norm1`, `norm2`, `norm3` before attention/FFN).
11. **`spatial` (the 960-token map with pos encoding) is still computed** even when `num_super_tokens > 0` — the slot attention pools from it. Only `spatial_for_decoder` (not `spatial`) is passed to the decoder layers.
12. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py`.

---

## Expected Behavior After Change

- Model computes 960 spatial tokens (as baseline), then compresses them to K=32 super-tokens via one slot-attention layer.
- Decoder cross-attention sees K=32 keys/values instead of 960 — ~30× reduction in cross-attention FLOPS.
- New parameters: `slot_queries` (32×256 = 8,192 scalars), `slot_attn` (~1.3M params for 256-dim 8-head MHA), `slot_norm` (512 scalars). Net parameter increase ~1.3M.
- Memory: slot attention adds ~10 MB activation; decoder cross-attention saves ~67 MB activation per layer. Net memory reduction vs. baseline.
- Forward output shapes unchanged — `BedlamMPJPEMetric` and all hooks are unaffected.
- Expected: `composite_val < 335` at stage-1 epoch 20. Diagnostic design to test whether learned spatial aggregation alone helps without increased decoder depth.
