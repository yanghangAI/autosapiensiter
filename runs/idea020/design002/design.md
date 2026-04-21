# Design 002 — Per-Query Cross-Attention Temperature: Log-Space + L2 Regularisation

**Design Description:** Same per-query cross-attention temperature scaling as Design 001, but parameterised in log-space via SoftPlus (guaranteed positivity, no clamp needed) with L2 regularisation weight=0.01 on log-temperatures.

**Starting Point:** `baseline/`

---

## Overview

The core algorithm change is identical to Design 001: replace the standard `nn.MultiheadAttention` cross-attention call with a custom `_temp_scaled_attn` function that inserts per-query temperature scaling of the dot-product logits before softmax. This design builds on Design 001's mechanism but uses a more numerically stable parameterisation: instead of storing `tau` directly (requiring a clamp to prevent collapse), store `log_cross_temp = nn.Parameter(torch.zeros(num_joints))` and compute `tau = F.softplus(log_cross_temp)`. `softplus(0) ≈ 0.693`, so the effective temperature at initialisation is ~0.693 (slightly sharper than baseline). This parameterisation is smooth, strictly positive, and differentiable everywhere — no clamping required.

Additionally, a small L2 regularisation loss `temp_reg_weight * log_cross_temp.pow(2).mean()` (weight=0.01) is added. This acts as a prior pulling log-temperatures toward 0 (i.e., tau toward softplus(0)≈0.693), preventing individual queries from learning degenerate extreme temperatures.

The `_temp_scaled_attn` helper function is the same as in Design 001 (but receives `tau = F.softplus(self.log_cross_temp)` instead of a directly-stored parameter).

---

## Files to Modify

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` is **not modified**.

---

## Detailed Changes

### `pose3d_transformer_head.py`

#### 1. New module-level helper function `_temp_scaled_attn`

Identical to Design 001. Add after `_build_2d_sincos_pos_enc`, before `_DecoderLayer`. The function signature and body are exactly as specified in Design 001:

```python
def _temp_scaled_attn(
    mha_module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    temperature: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Cross-attention with per-query temperature scaling of logits.
    ... (same as Design 001 docstring)
    """
    B, Nq, D = query.shape
    _, Ns, _ = key.shape
    Nh = mha_module.num_heads
    dh = D // Nh

    w = mha_module.in_proj_weight
    b = mha_module.in_proj_bias

    Q = torch.nn.functional.linear(query, w[:D],    b[:D]    if b is not None else None)
    K = torch.nn.functional.linear(key,   w[D:2*D], b[D:2*D] if b is not None else None)
    V = torch.nn.functional.linear(value, w[2*D:],  b[2*D:]  if b is not None else None)

    Q = Q.view(B, Nq, Nh, dh).transpose(1, 2)
    K = K.view(B, Ns, Nh, dh).transpose(1, 2)
    V = V.view(B, Ns, Nh, dh).transpose(1, 2)

    scale = dh ** -0.5
    attn = (Q @ K.transpose(-2, -1)) * scale

    # AMP-safe temperature cast; for Design 002 tau is already >= 0 from softplus
    tau = temperature.clamp(min=1e-6).to(attn.dtype).view(1, 1, Nq, 1)
    attn = (attn / tau).softmax(dim=-1)

    if training and mha_module.dropout > 0:
        attn = torch.nn.functional.dropout(attn, p=mha_module.dropout)

    out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
    return mha_module.out_proj(out)
```

Note: the clamp is `min=1e-6` rather than `min=0.1` — `softplus` already guarantees positivity far above 1e-6 in practice, but the clamp is kept as a numerical safety guard. Do not remove it.

#### 2. Modify `_DecoderLayer.__init__`

Identical to Design 001 — add `cross_temp` and `self_temp` optional parameters:

```python
def __init__(
    self,
    embed_dim: int,
    num_heads: int = 8,
    dropout: float = 0.1,
    cross_temp: 'nn.Parameter | None' = None,
    self_temp: 'nn.Parameter | None' = None,
):
    ...
    self.cross_temp = cross_temp
    self.self_temp = self_temp
```

#### 3. Modify `_DecoderLayer.forward`

Identical to Design 001 — replace cross-attention block:

```python
# Cross-attention
q = self.norm2(queries)
if self.cross_temp is not None:
    q2 = _temp_scaled_attn(
        self.cross_attn, q, spatial_tokens, spatial_tokens,
        self.cross_temp, training=self.training)
else:
    q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```

Self-attention is unchanged for Design 002.

#### 4. Modify `Pose3dTransformerHead.__init__`

Same four new kwargs as Design 001:

```python
use_cross_temp: bool = False,
use_self_temp: bool = False,
temp_log_space: bool = False,
temp_reg_weight: float = 0.0,
```

**Key difference from Design 001** — when `use_cross_temp=True` and `temp_log_space=True`, create a log-space parameter instead of a direct parameter:

```python
self.use_cross_temp = use_cross_temp
self.use_self_temp = use_self_temp
self.temp_log_space = temp_log_space
self.temp_reg_weight = temp_reg_weight

cross_temp_param = None
self_temp_param = None

if use_cross_temp:
    if temp_log_space:
        # Log-space parameterisation: tau = softplus(log_cross_temp)
        # softplus(0) ≈ 0.693 at init
        self.log_cross_temp = nn.Parameter(torch.zeros(num_joints))
        # Compute effective tau to pass to decoder layer (re-computed each forward)
        # We pass None here; the forward() method will call softplus at runtime
        cross_temp_param = None  # handled in forward() via self.log_cross_temp
    else:
        self.cross_temp = nn.Parameter(torch.ones(num_joints))
        cross_temp_param = self.cross_temp

if use_self_temp:
    self.self_temp = nn.Parameter(torch.ones(num_joints))
    self_temp_param = self.self_temp
```

**Important**: because `temp_log_space=True` requires computing `softplus(log_cross_temp)` at forward time (a dynamic tensor, not a static `nn.Parameter`), the decoder layer's `cross_temp` attribute cannot be pre-set at init time. Instead, pass `None` to `_DecoderLayer` at construction, and override `forward()` in `Pose3dTransformerHead` to call `_temp_scaled_attn` directly (bypassing the decoder layer's conditional), **or** restructure so the decoder layer receives the live tensor each forward call.

**Recommended implementation**: override `Pose3dTransformerHead.forward()` to call the decoder layer differently when `temp_log_space=True`. Specifically:

Replace the decoder layer construction — always pass `cross_temp=None` to `_DecoderLayer` (so it uses standard MHA call internally), and add the temperature logic in `Pose3dTransformerHead.forward()`:

```python
# Always construct decoder layer without cross_temp
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    cross_temp=None,   # temperature applied in head.forward() for log_space case
    self_temp=self_temp_param,
)

assert self.decoder_layer.cross_attn._qkv_same_embed_dim, \
    '_temp_scaled_attn requires _qkv_same_embed_dim=True'
```

Wait — this means Design 002 must override `_DecoderLayer.forward()` or restructure the call. The cleanest implementation is:

**Alternative (recommended)**: modify `_DecoderLayer.forward()` to accept an optional `cross_temp_override` argument, and have `Pose3dTransformerHead.forward()` pass the live `softplus(log_cross_temp)` tensor on each call:

```python
# _DecoderLayer.forward() signature change:
def forward(
    self,
    queries: torch.Tensor,
    spatial_tokens: torch.Tensor,
    cross_temp_override: torch.Tensor | None = None,
) -> torch.Tensor:
    ...
    # Cross-attention
    q = self.norm2(queries)
    effective_cross_temp = cross_temp_override if cross_temp_override is not None else self.cross_temp
    if effective_cross_temp is not None:
        q2 = _temp_scaled_attn(
            self.cross_attn, q, spatial_tokens, spatial_tokens,
            effective_cross_temp, training=self.training)
    else:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
    queries = queries + self.dropout2(q2)
    ...
```

Then in `Pose3dTransformerHead.forward()`:

```python
# Compute effective temperature for cross-attention
cross_temp_override = None
if self.use_cross_temp and self.temp_log_space:
    cross_temp_override = torch.nn.functional.softplus(self.log_cross_temp)

# Decoder
decoded = self.decoder_layer(
    queries, spatial, cross_temp_override=cross_temp_override)
```

When `temp_log_space=False` (Design 001 path), `cross_temp_override=None` and `self.decoder_layer.cross_temp` is used directly — no regression for Design 001 behaviour.

#### 5. `loss()` — add L2 regularisation term

After the existing loss computation, add:

```python
# Temperature L2 regularisation (Design 002 only: temp_reg_weight > 0)
if self.temp_reg_weight > 0 and hasattr(self, 'log_cross_temp'):
    losses['loss/temp_reg/train'] = (
        self.temp_reg_weight * self.log_cross_temp.pow(2).mean())
```

This adds `loss/temp_reg/train` to the losses dict with weight 0.01. MMEngine will include it in the total loss automatically.

#### 6. `_init_head_weights` — no additional changes needed

`nn.Parameter(torch.zeros(num_joints))` is already correctly initialised to zero (softplus(0) ≈ 0.693 at start).

---

### `config.py`

In the `model` dict, under `head=dict(...)`, add four new kwargs:

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
    use_cross_temp=True,
    use_self_temp=False,
    temp_log_space=True,
    temp_reg_weight=0.01,
),
```

All values are bool/float literals. No Python import statements.

---

## Constraints and Invariants to Preserve

1. **Body-only joint loss**: `_BODY = list(range(0, 22))` unchanged.
2. **Pelvis token**: `decoded[:, 0, :]` at index 0 unchanged.
3. **Backward compatibility**: all four new kwargs have defaults so non-specifying configs use baseline behaviour.
4. **AMP dtype cast**: `tau = temperature.clamp(min=1e-6).to(attn.dtype)` — mandatory. `softplus` output is float32; `attn` is float16 under AMP.
5. **`_qkv_same_embed_dim=True`** assertion must be present.
6. **`persistent_workers=False`**: unchanged.
7. **No MMEngine config imports**: all kwargs are literals.
8. **Regularisation loss key naming**: must use `'loss/temp_reg/train'` (with `/train` suffix) to match the MetricsCSVHook naming convention for train-only losses.
9. **`cross_temp_override` default**: `_DecoderLayer.forward()` must default `cross_temp_override=None` to preserve backward compatibility with all existing call sites.
10. **`softplus` import**: `torch.nn.functional.softplus` is available without additional imports since `torch.nn.functional` is already imported via `import torch.nn as nn` in the file's existing imports. Use `torch.nn.functional.softplus(self.log_cross_temp)` — no additional import statement required.
11. **Log-space param name**: the parameter is named `self.log_cross_temp` (not `self.cross_temp`). If `temp_log_space=False`, the parameter is named `self.cross_temp`. The Builder must be careful not to confuse these two cases — they coexist as different attributes in the same class.

---

## Expected Behaviour After Change

- At initialisation: `log_cross_temp` = all-zeros → `softplus(0) ≈ 0.693` → slightly sharper attention than baseline (tau < 1 means amplified logits). The regularisation loss at init is `0.01 * mean(0^2) = 0`.
- During training: `log_cross_temp[0]` (pelvis query) will increase (positive values → larger softplus → tau > 1 → diffuse attention). Distal joint log_temps will decrease (negative values → smaller softplus → tau < 1 → sharp attention). The L2 regularisation resists extreme values.
- The regularisation loss `loss/temp_reg/train` will appear in MMEngine logs and MetricsCSV. It should remain small (< 0.01) throughout training if the temperatures stay near zero.
- Inference: `forward()` calls `softplus(self.log_cross_temp)` even at eval mode — this is correct since the parameter is part of the model and should be used at inference.

---

## Target Metrics

- `composite_val` at stage-1 epoch 20: **< 332** (tighter than Design 001 due to stable parameterisation)
- `mpjpe_body_val` at stage-1 epoch 20: **< 185 mm**
- `mpjpe_pelvis_val` at stage-1 epoch 20: **< 605 mm**
