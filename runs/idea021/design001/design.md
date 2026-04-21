# Design 001 — Full Spatial Bias Matrix, Zero-Initialized

**Design Description:** Add a learnable `(70, 960)` additive bias to cross-attention logits (zero-initialized), passed via `attn_mask` to `nn.MultiheadAttention`.

**Starting Point:** `baseline/`

---

## Overview

This is the diagnostic baseline for idea021. The core algorithm change: a single unconstrained learnable parameter matrix `(num_joints, feat_h * feat_w)` = `(70, 960)` is added as an additive bias to the cross-attention logits in `_DecoderLayer` via `attn_mask`. Zero-initialization ensures exact baseline equivalence at training start. Any improvement over the baseline composite_val is attributable entirely to the learned spatial routing.

---

## Files to Change

1. `pose3d_transformer_head.py` — add `cross_attn_bias` argument to `_DecoderLayer.forward()`; add new constructor kwargs and parameter to `Pose3dTransformerHead`; route bias through `forward()`.
2. `config.py` — add `use_cross_attn_bias=True`, `cross_attn_bias_type='full'`, `feat_h=40`, `feat_w=24` to the head dict.

No changes to `pelvis_utils.py`.

---

## Exact Changes

### `pose3d_transformer_head.py`

#### 1. `_DecoderLayer.forward()` — add optional `cross_attn_bias` argument

Change the method signature from:
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor) -> torch.Tensor:
```
to:
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            cross_attn_bias: 'torch.Tensor | None' = None) -> torch.Tensor:
```

In the cross-attention block, replace:
```python
# Cross-attention
q = self.norm2(queries)
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```
with:
```python
# Cross-attention
q = self.norm2(queries)
if cross_attn_bias is not None:
    q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                          attn_mask=cross_attn_bias.to(q.dtype))[0]
else:
    q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```

The `.to(q.dtype)` cast is mandatory for AMP float16 compatibility: the bias is stored as float32 but must match the query dtype at runtime.

#### 2. `Pose3dTransformerHead.__init__()` — new kwargs and parameter

Add the following kwargs (with defaults ensuring backward compatibility with existing baseline configs):
```python
use_cross_attn_bias: bool = False,
cross_attn_bias_type: str = 'full',
feat_h: int = 40,
feat_w: int = 24,
joint_row_prior: list = None,
```

Store them as instance attributes:
```python
self.use_cross_attn_bias = use_cross_attn_bias
self.cross_attn_bias_type = cross_attn_bias_type
self.feat_h = feat_h
self.feat_w = feat_w
```

After the decoder_layer construction, add the parameter allocation block:
```python
if use_cross_attn_bias:
    if cross_attn_bias_type == 'full':
        self.cross_attn_bias = nn.Parameter(
            torch.zeros(num_joints, feat_h * feat_w))
    else:  # 'factored' or 'factored_warmstart'
        self.cross_attn_bias_row = nn.Parameter(
            torch.zeros(num_joints, feat_h))
        self.cross_attn_bias_col = nn.Parameter(
            torch.zeros(num_joints, feat_w))
```

For design001, `use_cross_attn_bias=True` and `cross_attn_bias_type='full'`, so only `self.cross_attn_bias` is created: shape `(70, 960)`, dtype float32, all zeros.

#### 3. `Pose3dTransformerHead._init_head_weights()` — no change needed

`nn.Parameter(torch.zeros(...))` is already zero-initialized. No warm-start logic required for design001.

#### 4. `Pose3dTransformerHead.forward()` — route bias to decoder_layer

Replace:
```python
# Decoder
decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)
```
with:
```python
# Decoder
if self.use_cross_attn_bias:
    if self.cross_attn_bias_type == 'full':
        bias = self.cross_attn_bias     # (num_joints, feat_h * feat_w) = (70, 960)
    else:
        bias = (self.cross_attn_bias_row.unsqueeze(-1) +
                self.cross_attn_bias_col.unsqueeze(-2))   # (num_joints, feat_h, feat_w)
        bias = bias.view(self.num_joints, -1)              # (num_joints, feat_h * feat_w)
    decoded = self.decoder_layer(queries, spatial, cross_attn_bias=bias)
else:
    decoded = self.decoder_layer(queries, spatial)
# (B, num_joints, hidden_dim)
```

`loss()` and `predict()` both call `self.forward(feats)` — no changes needed there.

---

### `config.py`

In the `model` dict, under `head=dict(...)`, add four new key-value pairs after `loss_weight_uv=1.0`:

```python
use_cross_attn_bias=True,
cross_attn_bias_type='full',
feat_h=40,
feat_w=24,
```

Note: `feat_h=40` and `feat_w=24` because the backbone produces `(B, C, H, W)` with H=640/16=40, W=384/16=24. The `feat.flatten(2).transpose(1,2)` in `forward()` flattens in row-major order over (H, W), yielding 40*24=960 spatial tokens. The `attn_mask` shape `(70, 960)` must match this flattening order.

All values are bool/str/int literals — no Python import statements. Fully MMEngine-compliant.

---

## Parameter Count

- `cross_attn_bias`: `70 × 960 = 67,200` float32 scalars = ~262 KB. Negligible relative to ~300M parameter backbone.

---

## `attn_mask` Semantics (PyTorch)

`nn.MultiheadAttention(batch_first=True)` adds `attn_mask` to the raw attention logits before softmax. Shape `(tgt_len, src_len)` = `(70, 960)` is broadcast over the batch and head dimensions. This is correct: the spatial prior for joint `i` is the same for all images and all heads (a prior, not an image-specific map).

Values are finite floats (zero at init). No `-inf` masking occurs — all 960 spatial positions retain positive attention weight after softmax.

---

## Invariants to Preserve

- `_BODY = list(range(0, 22))` in `loss()`: loss restricted to body joints indices 0–21. Unchanged.
- `pelvis_token = decoded[:, 0, :]` for pelvis depth/uv regression. Unchanged.
- `persistent_workers=False`. Unchanged (data loader constraint).
- All new kwargs have default values: `use_cross_attn_bias=False` etc. Existing baseline configs that omit these kwargs must continue to work without modification.
- AMP via `FixedAmpOptimWrapper` with `loss_scale='dynamic'`. The `.to(q.dtype)` cast in `_DecoderLayer.forward()` handles float16 compatibility.
- MMEngine config constraint: no `import` statements. All new config values are bool/str/int literals.

---

## Expected Behavior After Change

- At epoch 0 (before any gradient update): `cross_attn_bias` is all zeros → cross-attention output is identical to baseline.
- During training: each of the 67,200 bias scalars receives a gradient signal and learns a spatial preference. Joints with strong spatial priors in the training data (e.g., head always in upper region) converge to high bias values for the corresponding spatial grid cells.
- At validation: the learned bias is applied, improving cross-attention routing for body joints.
- Expected composite_val < 340 at stage-1 (vs. baseline ~346).
- The checkpoint saves `cross_attn_bias` as part of `state_dict` via `CheckpointHook`. Resume and inference work correctly.

---

## Edge Cases

- **`feat_h * feat_w` must equal the number of spatial tokens**: verified as 40*24=960. If the input resolution changes, `feat_h` and `feat_w` must be updated in config.
- **`batch_first=True` does not affect `attn_mask` shape**: PyTorch's `nn.MultiheadAttention` always expects `attn_mask` of shape `(tgt_len, src_len)` regardless of `batch_first`. Confirmed in PyTorch 2.x documentation.
- **Gradient explosions**: cross-attention gradient w.r.t. `B_i[j]` = `softmax_j(a_{i,j})` ∈ [0, 1]. Well-conditioned; no clipping needed beyond the existing `clip_grad=dict(max_norm=1.0)`.
