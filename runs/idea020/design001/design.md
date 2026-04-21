# Design 001 — Per-Query Cross-Attention Temperature Scaling (Minimal, Clamped)

**Design Description:** Add one learnable scalar temperature per joint query (70 params, init=1.0, clamped ≥0.1) applied only to cross-attention logits via a custom MHA forward; self-attention unchanged.

**Starting Point:** `baseline/`

---

## Overview

This is the minimal diagnostic variant. The core algorithm change is to replace the standard `nn.MultiheadAttention` cross-attention call in `_DecoderLayer.forward()` with a custom function (`_temp_scaled_attn`) that exposes logits before softmax and applies per-query temperature scaling. A single `nn.Parameter` of shape `(num_joints,)` initialised to `torch.ones(num_joints)` is added to `Pose3dTransformerHead`. The temperature is passed by reference to `_DecoderLayer`, which applies it in a custom cross-attention forward (`_temp_scaled_attn`) by dividing the dot-product logits by `tau_i` before softmax. Self-attention is untouched (standard `nn.MultiheadAttention` call). The config gains four new bool/float kwargs on the head dict.

---

## Files to Modify

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` is **not modified**.

---

## Detailed Changes

### `pose3d_transformer_head.py`

#### 1. New module-level helper function `_temp_scaled_attn`

Add immediately after the existing `_build_2d_sincos_pos_enc` function (before `class _DecoderLayer`):

```python
def _temp_scaled_attn(
    mha_module: nn.MultiheadAttention,
    query: torch.Tensor,        # (B, Nq, D)
    key: torch.Tensor,          # (B, Ns, D)
    value: torch.Tensor,        # (B, Ns, D)
    temperature: torch.Tensor,  # (Nq,)  — nn.Parameter or derived tensor
    training: bool = True,
) -> torch.Tensor:
    """Cross-attention with per-query temperature scaling of logits.

    Implements:  attn = softmax( (Q @ K^T / sqrt(dh)) / tau )  @ V
    where tau is a per-query scalar (shape (Nq,)).

    The temperature is clamped to >= 0.1 to prevent logit overflow under AMP.
    tau = 1.0 at init → identical to standard scaled dot-product attention.

    Uses mha_module.in_proj_weight / in_proj_bias for Q/K/V projections and
    mha_module.out_proj for the output projection.  Requires
    mha_module._qkv_same_embed_dim == True (verified in head __init__).
    """
    B, Nq, D = query.shape
    _, Ns, _ = key.shape
    Nh = mha_module.num_heads
    dh = D // Nh

    w = mha_module.in_proj_weight   # (3D, D)
    b = mha_module.in_proj_bias     # (3D,) or None

    Q = torch.nn.functional.linear(
        query, w[:D],    b[:D]    if b is not None else None)   # (B, Nq, D)
    K = torch.nn.functional.linear(
        key,   w[D:2*D], b[D:2*D] if b is not None else None)  # (B, Ns, D)
    V = torch.nn.functional.linear(
        value, w[2*D:],  b[2*D:]  if b is not None else None)  # (B, Ns, D)

    # Reshape to multi-head layout
    Q = Q.view(B, Nq, Nh, dh).transpose(1, 2)  # (B, Nh, Nq, dh)
    K = K.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)
    V = V.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)

    scale = dh ** -0.5
    attn = (Q @ K.transpose(-2, -1)) * scale    # (B, Nh, Nq, Ns)

    # Per-query temperature: clamp prevents collapse; cast to attn dtype for AMP
    tau = temperature.clamp(min=0.1).to(attn.dtype).view(1, 1, Nq, 1)
    attn = (attn / tau).softmax(dim=-1)          # (B, Nh, Nq, Ns)

    if training and mha_module.dropout > 0:
        attn = torch.nn.functional.dropout(attn, p=mha_module.dropout)

    out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)  # (B, Nq, D)
    return mha_module.out_proj(out)                                  # (B, Nq, D)
```

#### 2. Modify `_DecoderLayer.__init__`

Change the constructor signature to accept optional temperature parameters:

```python
def __init__(
    self,
    embed_dim: int,
    num_heads: int = 8,
    dropout: float = 0.1,
    cross_temp: 'nn.Parameter | None' = None,
    self_temp: 'nn.Parameter | None' = None,
):
```

At the end of `__init__`, store these as attributes:

```python
# Temperature parameters (None → use standard MHA call)
self.cross_temp = cross_temp
self.self_temp = self_temp
```

#### 3. Modify `_DecoderLayer.forward`

Replace the cross-attention block:

**Before:**
```python
# Cross-attention
q = self.norm2(queries)
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```

**After:**
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

Self-attention block is unchanged for Design 001.

#### 4. Modify `Pose3dTransformerHead.__init__`

Add four new keyword arguments (all with defaults for backward compatibility):

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(...),
    loss_depth: ConfigType = dict(...),
    loss_uv: ConfigType = dict(...),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    use_cross_temp: bool = False,
    use_self_temp: bool = False,
    temp_log_space: bool = False,
    temp_reg_weight: float = 0.0,
    init_cfg: OptConfigType = None,
):
```

Store the new config flags and create temperature parameters (insert after `self.loss_weight_uv = loss_weight_uv`):

```python
self.use_cross_temp = use_cross_temp
self.use_self_temp = use_self_temp
self.temp_log_space = temp_log_space
self.temp_reg_weight = temp_reg_weight

# Learnable temperature parameters
cross_temp_param = None
self_temp_param = None

if use_cross_temp:
    # Design 001: direct parameterisation, clamped to >= 0.1 at use time
    # (temp_log_space=False for this design)
    self.cross_temp = nn.Parameter(torch.ones(num_joints))
    cross_temp_param = self.cross_temp

if use_self_temp:
    self.self_temp = nn.Parameter(torch.ones(num_joints))
    self_temp_param = self.self_temp
```

Replace the decoder layer construction:

**Before:**
```python
# Transformer decoder (1 layer)
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```

**After:**
```python
# Transformer decoder (1 layer)
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    cross_temp=cross_temp_param,
    self_temp=self_temp_param,
)
```

Add assertion for in_proj_weight access (insert after decoder_layer construction):

```python
# Verify standard QKV projection layout (required by _temp_scaled_attn)
assert self.decoder_layer.cross_attn._qkv_same_embed_dim, \
    '_temp_scaled_attn requires _qkv_same_embed_dim=True'
```

#### 5. `_init_head_weights` — no additional changes needed

`nn.Parameter(torch.ones(num_joints))` is already correctly initialised to 1.0 (tau=1.0 → identical to baseline at init).

#### 6. `loss()` — no changes needed for Design 001

`temp_reg_weight=0.0` → no regularisation loss term.

---

### `config.py`

In the `model` dict, under `head=dict(...)`, add four new kwargs as literal values:

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
    temp_log_space=False,
    temp_reg_weight=0.0,
),
```

All other config values are identical to baseline. No Python import statements are introduced.

---

## Constraints and Invariants to Preserve

1. **Body-only joint loss**: the `_BODY = list(range(0, 22))` restriction in `loss()` must remain unchanged.
2. **Pelvis token**: token 0 (`decoded[:, 0, :]`) is the pelvis token — `depth_out` and `uv_out` still read from index 0. No change.
3. **Backward compatibility**: all four new kwargs have defaults (`False`, `False`, `False`, `0.0`) so that any config not specifying them behaves identically to baseline.
4. **AMP dtype cast**: `tau = temperature.clamp(min=0.1).to(attn.dtype)` — the `.to(attn.dtype)` cast is mandatory. `attn` is float16 under AMP; the `nn.Parameter` (cross_temp) is stored as float32. Without the cast, the division would upcast `attn` to float32, breaking AMP efficiency and potentially causing memory issues.
5. **`_qkv_same_embed_dim=True`**: the assertion `assert self.decoder_layer.cross_attn._qkv_same_embed_dim` must be present; this is always True for the baseline architecture but guards against future changes.
6. **`persistent_workers=False`**: already set in baseline config; do not change.
7. **No MMEngine config imports**: `use_cross_temp=True`, `use_self_temp=False`, `temp_log_space=False`, `temp_reg_weight=0.0` are all bool/float literals — no import statements.
8. **`nn.Parameter` passed by reference**: `cross_temp_param = self.cross_temp` stores a reference to the same `nn.Parameter` object. `_DecoderLayer` stores this reference as `self.cross_temp`. Both point to the same tensor; gradients accumulate to the head's `cross_temp` parameter — correct. Do not copy the tensor.
9. **Dropout in `_temp_scaled_attn`**: use `mha_module.dropout` (a float attribute on `nn.MultiheadAttention`) as the dropout probability. Do not add a separate dropout attribute.
10. **`num_joints` for temperature shape**: the temperature parameter shape is `(num_joints,)` = `(70,)`. This matches the query dimension of the decoder — the decoder processes all 70 joint queries simultaneously.

---

## Expected Behaviour After Change

- At initialisation: `cross_temp` = all-ones → `attn / 1.0` → identical to baseline attention. Loss and gradient match baseline at step 0.
- During training: `cross_temp[0]` (pelvis query) is expected to increase toward τ > 1, producing a flatter attention distribution. Distal joint queries (e.g., indices for wrists/ankles in the BEDLAM2 joint ordering) are expected to decrease toward τ < 1, producing sharper attention.
- At inference (`predict()`): `self.training=False` → the dropout branch in `_temp_scaled_attn` is skipped; temperature scaling still applies via the `cross_temp` parameter. No special inference mode needed.
- New parameter count: +70 scalars (cross_temp). Negligible overhead.

---

## Target Metrics

- `composite_val` at stage-1 epoch 20: **< 338** (baseline: ~352)
- `mpjpe_body_val` at stage-1 epoch 20: **< 188 mm** (baseline: ~195.7 mm)
- `mpjpe_pelvis_val` at stage-1 epoch 20: **< 610 mm** (baseline: ~653 mm)
