# Design 002 — Low-Rank Factored Spatial Bias (row + column, zero-initialized)

**Design Description:** Add a factored additive cross-attention bias `u_i[h] + v_i[w]` (zero-initialized) where row-bias `(70,40)` and col-bias `(70,24)` are separate learnable parameters.

**Starting Point:** `baseline/`

---

## Overview

The core algorithm change: instead of a full `(70, 960)` bias matrix (design001), the cross-attention bias is factored as an outer sum:
```
B_i[h, w] = u_i[h] + v_i[w]
```
where `u_i ∈ R^{H'=40}` (per-joint row bias) and `v_i ∈ R^{W'=24}` (per-joint column bias). The full `(70, 40, 24)` matrix is reconstructed via broadcasting and flattened to `(70, 960)` for `attn_mask`.

This factored form has 15× fewer parameters than design001 (4,480 vs. 67,200 scalars) and imposes the inductive bias that each joint's spatial preference is separable into independent row and column preferences. For human body joints in a centred crop, this is a well-motivated prior: many joints have strong row preferences (head=top, feet=bottom) but approximately uniform or symmetric column preferences.

Both parameters are zero-initialized — exact baseline equivalence at training start.

---

## Files to Change

1. `pose3d_transformer_head.py` — identical changes to _DecoderLayer as design001; different parameter allocation in `__init__` (two parameters instead of one); same forward routing.
2. `config.py` — add `use_cross_attn_bias=True`, `cross_attn_bias_type='factored'`, `feat_h=40`, `feat_w=24` to the head dict.

No changes to `pelvis_utils.py`.

---

## Exact Changes

### `pose3d_transformer_head.py`

#### 1. `_DecoderLayer.forward()` — identical to design001

Add optional `cross_attn_bias: 'torch.Tensor | None' = None` argument to signature. In the cross-attention block:

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

#### 2. `Pose3dTransformerHead.__init__()` — new kwargs and parameters

Same new kwargs as design001:
```python
use_cross_attn_bias: bool = False,
cross_attn_bias_type: str = 'full',
feat_h: int = 40,
feat_w: int = 24,
joint_row_prior: list = None,
```

Store as instance attributes (same as design001). For `cross_attn_bias_type='factored'`, the parameter allocation block creates two parameters:
```python
if use_cross_attn_bias:
    if cross_attn_bias_type == 'full':
        self.cross_attn_bias = nn.Parameter(
            torch.zeros(num_joints, feat_h * feat_w))
    else:  # 'factored' or 'factored_warmstart'
        self.cross_attn_bias_row = nn.Parameter(
            torch.zeros(num_joints, feat_h))   # (70, 40), float32, zeros
        self.cross_attn_bias_col = nn.Parameter(
            torch.zeros(num_joints, feat_w))   # (70, 24), float32, zeros
```

For design002, `cross_attn_bias_type='factored'`, so `self.cross_attn_bias_row` shape `(70, 40)` and `self.cross_attn_bias_col` shape `(70, 24)` are created.

#### 3. `Pose3dTransformerHead._init_head_weights()` — no change

Both parameters are `torch.zeros(...)` — already zero-initialized. No warm-start logic.

#### 4. `Pose3dTransformerHead.forward()` — route factored bias to decoder_layer

Same block as design001 (the `else` branch handles the factored case):
```python
if self.use_cross_attn_bias:
    if self.cross_attn_bias_type == 'full':
        bias = self.cross_attn_bias     # (70, 960)
    else:
        # Factored: outer sum via broadcasting
        # cross_attn_bias_row: (70, 40) → unsqueeze(-1) → (70, 40, 1)
        # cross_attn_bias_col: (70, 24) → unsqueeze(-2) → (70, 1, 24)
        # sum → (70, 40, 24), then flatten last two dims → (70, 960)
        bias = (self.cross_attn_bias_row.unsqueeze(-1) +
                self.cross_attn_bias_col.unsqueeze(-2))   # (70, 40, 24)
        bias = bias.view(self.num_joints, -1)              # (70, 960)
    decoded = self.decoder_layer(queries, spatial, cross_attn_bias=bias)
else:
    decoded = self.decoder_layer(queries, spatial)
```

The outer-product expansion `(70, 40, 24)` is computed once per forward pass from two small parameter tensors — negligible overhead (70×64 extra FLOPs). The `.view(self.num_joints, -1)` relies on `feat_h * feat_w = 40 * 24 = 960` matching the spatial token count.

---

### `config.py`

In the `model` dict, under `head=dict(...)`, add after `loss_weight_uv=1.0`:

```python
use_cross_attn_bias=True,
cross_attn_bias_type='factored',
feat_h=40,
feat_w=24,
```

All values are bool/str/int literals. No Python import statements. Fully MMEngine-compliant.

---

## Parameter Count

- `cross_attn_bias_row`: `70 × 40 = 2,800` scalars
- `cross_attn_bias_col`: `70 × 24 = 1,680` scalars
- Total: `4,480` float32 scalars = ~17.5 KB. 15× fewer than design001.

---

## `attn_mask` Semantics

Identical to design001: `(70, 960)` tensor added to cross-attention logits before softmax, broadcast over batch and head dimensions.

The factored bias at position `(i, h, w)` (joint i, row h, column w after flattening) is `u_i[h] + v_i[w]`. Both `u_i` and `v_i` are zero-initialized, so at training start the bias is identically zero — exact baseline equivalence.

---

## Invariants to Preserve

All invariants from design001 apply:
- Body joint loss restricted to indices 0–21.
- `pelvis_token = decoded[:, 0, :]` unchanged.
- `persistent_workers=False` unchanged.
- All new kwargs have default values for backward compatibility.
- AMP float16 compatibility via `.to(q.dtype)` cast.
- MMEngine config constraint: no `import` statements.
- `feat_h=40, feat_w=24` must match the spatial token dimension from backbone.

---

## Expected Behavior After Change

- At epoch 0: all bias values zero → identical output to baseline.
- During training: `u_i[h]` learns per-joint row preferences (which horizontal band to attend to for joint `i`); `v_i[w]` learns per-joint column preferences. For body joints, strong row preferences are expected (head-top, feet-bottom), while column preferences remain near zero for most joints (approximately centred in the crop).
- The factored parameterization may converge faster than design001 because it has 15× fewer parameters to learn, and the inductive bias (row-column separability) is well-matched to the BEDLAM2 centred-crop geometry.
- Expected composite_val < 333 at stage-1 (vs. design001 target < 340, baseline ~346).
- The checkpoint saves both `cross_attn_bias_row` and `cross_attn_bias_col` as part of `state_dict`.

---

## Edge Cases

- **`bias.view(self.num_joints, -1)`**: requires `self.feat_h * self.feat_w == 960`. Verified: 40*24=960. If resolution changes, both `feat_h` and `feat_w` must be updated in config.
- **`unsqueeze(-1)` on row tensor `(70, 40)` → `(70, 40, 1)` and `unsqueeze(-2)` on col tensor `(70, 24)` → `(70, 1, 24)`**: broadcasting to `(70, 40, 24)` is correct. The resulting `[i, h, w]` entry is `u_i[h] + v_i[w]`. Flattened in row-major order over (h, w) → matches `feat.flatten(2).transpose(1,2)` token ordering.
- **No `joint_row_prior` usage**: design002 ignores the `joint_row_prior` kwarg even if present. The warm-start logic in `_init_head_weights()` is only triggered when `cross_attn_bias_type == 'factored_warmstart'`.
