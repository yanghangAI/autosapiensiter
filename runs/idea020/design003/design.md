# Design 003 — Per-Query Temperature Scaling on Both Self-Attention and Cross-Attention

**Design Description:** Extend Design 001's per-query cross-attention temperature to also scale self-attention logits: each joint query has an independent learned temperature for self-attention (70 params) and cross-attention (70 params), both initialised to 1.0, clamped ≥0.1.

**Starting Point:** `baseline/`

---

## Overview

The core algorithm change extends Design 001's per-query temperature scaling from cross-attention only to both self-attention and cross-attention. Both attention mechanisms in `_DecoderLayer.forward()` are replaced with `_temp_scaled_attn` calls, each receiving an independent learned temperature vector.

This is the highest-expressivity variant. Two independent `nn.Parameter` tensors of shape `(num_joints,)` are added: `self.cross_temp` (controlling cross-attention sharpness) and `self.self_temp` (controlling self-attention sharpness). Both are initialised to `torch.ones(num_joints)` and clamped to ≥0.1 at use time.

Self-attention temperature allows each joint to independently control how much it attends to other joint queries:
- Pelvis query (index 0): large `self_temp_0` → diffuse self-attention → aggregates information from all other joint queries (helpful for global body scale estimation)
- Distal joints (wrists, ankles): small `self_temp_i` → sharp self-attention → focuses on kinematically relevant neighbour queries

The cross-attention temperature behaves identically to Design 001.

Total new parameters: 140 scalars (70 for cross_temp + 70 for self_temp) ≈ 560 bytes.

---

## Files to Modify

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` is **not modified**.

---

## Detailed Changes

### `pose3d_transformer_head.py`

#### 1. New module-level helper function `_temp_scaled_attn`

Identical to Design 001. Add after `_build_2d_sincos_pos_enc`, before `_DecoderLayer`. Exact function body:

```python
def _temp_scaled_attn(
    mha_module: nn.MultiheadAttention,
    query: torch.Tensor,        # (B, Nq, D)
    key: torch.Tensor,          # (B, Ns, D)
    value: torch.Tensor,        # (B, Ns, D)
    temperature: torch.Tensor,  # (Nq,)
    training: bool = True,
) -> torch.Tensor:
    """Cross/self-attention with per-query temperature scaling of logits.

    attn = softmax( (Q @ K^T / sqrt(dh)) / tau ) @ V
    tau shape: (Nq,). Clamped to >= 0.1 to prevent logit overflow under AMP.
    tau = 1.0 at init → identical to standard scaled dot-product attention.
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

    Q = Q.view(B, Nq, Nh, dh).transpose(1, 2)  # (B, Nh, Nq, dh)
    K = K.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)
    V = V.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)

    scale = dh ** -0.5
    attn = (Q @ K.transpose(-2, -1)) * scale    # (B, Nh, Nq, Ns)

    # AMP dtype cast: temperature (float32 Parameter) → attn dtype (float16 under AMP)
    tau = temperature.clamp(min=0.1).to(attn.dtype).view(1, 1, Nq, 1)
    attn = (attn / tau).softmax(dim=-1)

    if training and mha_module.dropout > 0:
        attn = torch.nn.functional.dropout(attn, p=mha_module.dropout)

    out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
    return mha_module.out_proj(out)
```

#### 2. Modify `_DecoderLayer.__init__`

Add `cross_temp` and `self_temp` optional parameters (same as Designs 001/002):

```python
def __init__(
    self,
    embed_dim: int,
    num_heads: int = 8,
    dropout: float = 0.1,
    cross_temp: 'nn.Parameter | None' = None,
    self_temp: 'nn.Parameter | None' = None,
):
    super().__init__()
    # ... existing layer construction unchanged ...
    self.cross_temp = cross_temp
    self.self_temp = self_temp
```

#### 3. Modify `_DecoderLayer.forward`

**Key difference from Design 001**: both self-attention and cross-attention blocks are modified.

Replace self-attention block:

**Before:**
```python
# Self-attention
q = self.norm1(queries)
q2 = self.self_attn(q, q, q)[0]
queries = queries + self.dropout1(q2)
```

**After:**
```python
# Self-attention
q = self.norm1(queries)
if self.self_temp is not None:
    q2 = _temp_scaled_attn(
        self.self_attn, q, q, q,
        self.self_temp, training=self.training)
else:
    q2 = self.self_attn(q, q, q)[0]
queries = queries + self.dropout1(q2)
```

Replace cross-attention block (identical to Design 001):

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

#### 4. Self-attention temperature shape note

For self-attention, `_temp_scaled_attn` is called with `query=q`, `key=q`, `value=q` — all have shape `(B, num_joints, D)`. Both `Nq` and `Ns` equal `num_joints`. The temperature shape `(num_joints,)` is applied to the query dimension (`Nq`), which is correct: each query row in the self-attention matrix gets its own temperature. Verify that `B, Nq, D = query.shape` extracts `Nq = num_joints` correctly — it does, since `q` comes from `self.norm1(queries)` which is `(B, num_joints, D)`.

#### 5. Modify `Pose3dTransformerHead.__init__`

Same four new kwargs as Designs 001/002:

```python
use_cross_temp: bool = False,
use_self_temp: bool = False,
temp_log_space: bool = False,
temp_reg_weight: float = 0.0,
```

**Key difference**: `use_self_temp=True` → create `self.self_temp` parameter:

```python
self.use_cross_temp = use_cross_temp
self.use_self_temp = use_self_temp
self.temp_log_space = temp_log_space
self.temp_reg_weight = temp_reg_weight

cross_temp_param = None
self_temp_param = None

if use_cross_temp:
    # Design 003: direct parameterisation, clamped at use time
    self.cross_temp = nn.Parameter(torch.ones(num_joints))
    cross_temp_param = self.cross_temp

if use_self_temp:
    self.self_temp = nn.Parameter(torch.ones(num_joints))
    self_temp_param = self.self_temp
```

Pass both to the decoder layer:

```python
# Transformer decoder (1 layer)
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    cross_temp=cross_temp_param,
    self_temp=self_temp_param,
)

# Verify standard QKV projection layout (required by _temp_scaled_attn)
assert self.decoder_layer.cross_attn._qkv_same_embed_dim, \
    '_temp_scaled_attn requires _qkv_same_embed_dim=True'
assert self.decoder_layer.self_attn._qkv_same_embed_dim, \
    '_temp_scaled_attn requires _qkv_same_embed_dim=True for self_attn'
```

#### 6. `loss()` — no changes needed for Design 003

`temp_reg_weight=0.0` → no regularisation loss term.

#### 7. `_init_head_weights` — no additional changes needed

Both `nn.Parameter(torch.ones(...))` tensors initialise to 1.0 (tau=1.0 at start → identical to baseline).

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
    use_self_temp=True,
    temp_log_space=False,
    temp_reg_weight=0.0,
),
```

All values are bool/float literals. No Python import statements.

---

## Constraints and Invariants to Preserve

1. **Body-only joint loss**: `_BODY = list(range(0, 22))` unchanged.
2. **Pelvis token**: `decoded[:, 0, :]` at index 0 unchanged.
3. **Backward compatibility**: all four new kwargs have defaults so non-specifying configs use baseline behaviour.
4. **AMP dtype cast**: `tau = temperature.clamp(min=0.1).to(attn.dtype)` — mandatory for both self-attention and cross-attention temperature calls. Both `self.self_attn` and `self.cross_attn` are used inside `_temp_scaled_attn`; the cast must be in the function body (not at the call site).
5. **`_qkv_same_embed_dim=True`** assertions must be present for both `self_attn` and `cross_attn`.
6. **`persistent_workers=False`**: unchanged.
7. **No MMEngine config imports**: all kwargs are literals.
8. **Self-attention temperature dimension**: in self-attention, the attention matrix is `(B, Nh, Nq, Nq)` (square) since key/value = query. The temperature `(Nq,)` is reshaped to `(1, 1, Nq, 1)` and applied across the **query rows** — this is correct. Each row i of the attention matrix (representing joint query i's attention distribution over all joint queries) gets temperature `tau_i`. This is the intended behaviour.
9. **`nn.Parameter` passed by reference**: same as Design 001 — both `cross_temp_param = self.cross_temp` and `self_temp_param = self.self_temp` store references. Do not copy tensors.
10. **Dropout in `_temp_scaled_attn`**: uses `mha_module.dropout` float attribute. For self-attention calls, this is `self.self_attn.dropout`; for cross-attention, `self.cross_attn.dropout`. Both are the same value (0.1) in the baseline config.
11. **Two assertions**: both `cross_attn._qkv_same_embed_dim` and `self_attn._qkv_same_embed_dim` assertions must pass. Both are `True` in the standard `nn.MultiheadAttention` construction used in the baseline.
12. **`num_joints` temperature shape for self-attention**: the temperature shape is `(num_joints,) = (70,)` for both parameters. The self-attention operates on all 70 joint queries simultaneously — the temperature maps 1:1 to each query row.

---

## Expected Behaviour After Change

- At initialisation: both `cross_temp` and `self_temp` = all-ones → both attention mechanisms identical to baseline at step 0.
- During training:
  - `self_temp[0]` (pelvis self-attention): expected to increase → pelvis attends diffusely to all other joint queries (aggregates global body configuration info).
  - `cross_temp[0]` (pelvis cross-attention): expected to increase → pelvis attends diffusely to all 960 spatial tokens (aggregates global scene depth).
  - Distal joint self-temps: expected to decrease → sharper focus on kinematic neighbour queries.
  - Distal joint cross-temps: expected to decrease → sharper spatial focus.
- The two temperature sets (`self_temp`, `cross_temp`) are independent. There is no constraint that they move in the same direction for any query.
- At inference: both temperature parameters are used via `_temp_scaled_attn` in both attention blocks. No special inference mode needed.
- New parameter count: +140 scalars total. Negligible overhead.

---

## Target Metrics

- `composite_val` at stage-1 epoch 20: **< 328** (competitive with best prior 328.14 from idea013/design003)
- `mpjpe_body_val` at stage-1 epoch 20: **< 182 mm**
- `mpjpe_pelvis_val` at stage-1 epoch 20: **< 600 mm**
- `composite_val` at stage-2 epoch 10: **< 225** (targeting near-best prior 224.52 from idea001/design001)
