**Design Description:** K=64 super-tokens with positional slot initialization grounding each slot to a spatial block.

**Starting Point:** `baseline/`

---

## Overview

Same two-stage slot-attention pooling as Design A (K=64 instead of K=32), but the slot query embeddings are initialized not from random noise but from the mean 2D sinusoidal positional encoding of each spatial block. The 24×40 feature grid is partitioned into 64 non-overlapping blocks (8 rows × 8 columns, each block spanning 3 rows × 5 columns = 15 tokens), and each slot is initialized with the mean of its block's positional encodings. This spatially grounds the slots at initialization, avoiding the cold-start problem where random slot queries attend uniformly or collapse to a single region in early training epochs.

**Algorithm:**
- K=64 learnable slot vectors; initialized via spatial block-averaged sinusoidal encodings, then fine-tuned during training.
- Initialization logic (executed once at `__init__` time, inside `_init_head_weights`):
  1. Compute `pos_enc = _build_2d_sincos_pos_enc(24, 40, 256)` → shape `(1, 960, 256)`.
  2. Reshape to `(24, 40, 256)`.
  3. Partition into 8×8 = 64 blocks of size 3×5: block `(r, c)` covers rows `[3r : 3r+3]`, columns `[5c : 5c+5]`.
  4. For each block `(r, c)`: average the 15 token encodings → `(256,)` vector.
  5. Stack all 64 averages → `(64, 256)` tensor. Assign as `self.slot_queries.weight.data`.
- At forward time: slot queries (loaded from the learned embedding, initialized as above) are pre-normed via `self.slot_norm` and cross-attend over the 960 spatial tokens via `self.slot_attn`.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Constructor signature** — add the same four new parameters as Design A (identical signatures):

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_super_tokens: int = 0,       # 0 = disabled (baseline)
    slot_pos_init: bool = False,     # True = positional block init for slot queries
    num_decoder_layers: int = 1,
    aux_loss_weight: float = 0.0,
    loss_joints: ConfigType = ...,
    ...
):
```

Store all four: `self.num_super_tokens`, `self.slot_pos_init`, `self.num_decoder_layers`, `self.aux_loss_weight`.

**New modules in `__init__`** — identical to Design A:

```python
if self.num_super_tokens > 0:
    self.slot_queries = nn.Embedding(num_super_tokens, hidden_dim)
    self.slot_attn = nn.MultiheadAttention(
        hidden_dim, num_heads, dropout=dropout, batch_first=True)
    self.slot_norm = nn.LayerNorm(hidden_dim)
```

**Decoder layers** — same replacement as Design A:

```python
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Remove `self.decoder_layer` (singular).

**`_init_head_weights()` changes** — the positional slot initialization logic for Design B:

```python
if self.num_super_tokens > 0:
    nn.init.trunc_normal_(self.slot_queries.weight, std=0.02)
    if self.slot_pos_init:
        # Build 2D sincos pos enc for H'=24, W'=40
        pos = _build_2d_sincos_pos_enc(24, 40, self.hidden_dim)  # (1, 960, hidden_dim)
        pos = pos.squeeze(0).reshape(24, 40, self.hidden_dim)     # (24, 40, hidden_dim)
        # Partition into 8x8=64 blocks, each block is 3 rows x 5 cols
        # num_super_tokens must equal 64 for this to apply
        assert self.num_super_tokens == 64, (
            "slot_pos_init=True requires num_super_tokens=64")
        block_means = []
        for r in range(8):
            for c in range(8):
                block = pos[3*r : 3*r+3, 5*c : 5*c+5, :]  # (3, 5, hidden_dim)
                block_means.append(block.reshape(-1, self.hidden_dim).mean(0))  # (hidden_dim,)
        slot_init = torch.stack(block_means, dim=0)  # (64, hidden_dim)
        self.slot_queries.weight.data.copy_(slot_init)
```

This runs at `__init__` time (called from `_init_head_weights()`, which is called at the end of `__init__`). After this, `self.slot_queries.weight` is a trainable parameter initialized with spatial structure rather than random noise.

Important: `_build_2d_sincos_pos_enc` is already defined as a module-level function in `pose3d_transformer_head.py`, so calling it directly inside `_init_head_weights` is valid — no import needed.

**`forward()` changes** — identical to Design A:

```python
spatial = feat.flatten(2).transpose(1, 2)
spatial = self.input_proj(spatial)
pos_enc = self._get_pos_enc(H, W, feat.device)
spatial = spatial + pos_enc

if self.num_super_tokens > 0:
    S = self.slot_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, hidden_dim)
    S_normed = self.slot_norm(S)
    super_tokens, _ = self.slot_attn(S_normed, spatial, spatial)  # (B, K, hidden_dim)
    spatial_for_decoder = super_tokens
else:
    spatial_for_decoder = spatial

queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

decoded = queries
intermediate_outputs = []
for layer in self.decoder_layers:
    decoded = layer(decoded, spatial_for_decoder)
    intermediate_outputs.append(decoded)

joints = self.joints_out(decoded)
pelvis_token = decoded[:, 0, :]
pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

**`loss()` — no changes** (same as baseline and Design A; `aux_loss_weight=0.0`).

**`predict()` — no changes.**

### 2. `config.py`

In the `head=dict(...)` block:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_super_tokens=64,
    slot_pos_init=True,
    num_decoder_layers=1,
    aux_loss_weight=0.0,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values remain identical to baseline (same LR, weight decay, warmup, batch, seed, data pipeline).

---

## Constraints and Invariants to Preserve

1. **Joint loss scope:** `_BODY = list(range(0, 22))` only. Unchanged.
2. **Pelvis token:** `decoded[:, 0, :]` from final decoder layer.
3. **`persistent_workers=False`** — do not change.
4. **No Python `import` in `config.py`** — `num_super_tokens=64`, `slot_pos_init=True`, `num_decoder_layers=1`, `aux_loss_weight=0.0` are all literals.
5. **Absolute imports in `pose3d_transformer_head.py`** — unchanged.
6. **`_build_2d_sincos_pos_enc` is called at `__init__` time** (inside `_init_head_weights`) — the tensor is discarded after weight assignment; it does not persist as a buffer. This is a one-time CPU computation with no GPU dependency.
7. **The `slot_pos_init` assertion** (`assert self.num_super_tokens == 64`) should be present to catch misconfiguration early. The assert fires at `__init__` time, not at forward time.
8. **Block dimensions:** 8 rows × 8 columns = 64 blocks. Each block is 3 rows × 5 columns of the 24×40 grid. Exact coverage: rows `[3r : 3r+3]` for r in 0..7 covers rows 0–23 (complete); columns `[5c : 5c+5]` for c in 0..7 covers columns 0–39 (complete). No overlap, no gap.
9. **Feature grid assumption:** H'=24, W'=40 is hardcoded in `_init_head_weights`. This matches the backbone output for 640×384 input at 1/16 stride (640/16=40, 384/16=24). If the input resolution changes, the assertion in `slot_pos_init` logic may fail — but that is acceptable since the design targets the fixed 640×384 input.
10. **Output tensor shapes unchanged:** `(B, 70, 3)`, `(B, 1)`, `(B, 2)`.
11. **`_DecoderLayer` class not modified.**
12. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py`.

---

## Expected Behavior After Change

- Model compresses 960 spatial tokens to K=64 super-tokens (vs. K=32 in Design A). The larger K=64 provides more capacity per super-token but less compression (15× vs. 30×).
- Slot queries are spatially grounded at initialization: slot 0 corresponds to the top-left 3×5 block, slot 63 corresponds to the bottom-right 3×5 block. During training, the slots can drift from their initial spatial positions, but the warm start ensures that early gradient signals are meaningful.
- New parameters: `slot_queries` (64×256 = 16,384 scalars), `slot_attn` (~1.3M params), `slot_norm` (512 scalars). Net parameter increase ~1.3M (same as Design A; slot_queries is slightly larger).
- No cold-start risk: slot queries are initialized to spatially distinct regions, so softmax competition among the 64 slots is meaningful from epoch 1.
- Expected: `composite_val < 330` at stage-1 epoch 20. Tests whether warm-started spatial partition slots converge faster than random-init slots within the 20-epoch budget.
