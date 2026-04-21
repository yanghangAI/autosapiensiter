# Design 002 — Vertical Band Warm-Start Cross-Attention Routing (Design B)

**Design Description:** Same as Design 001 but initialise `cross_attn_bias` with a Gaussian vertical-band prior: body-lower joints biased toward lower spatial rows (+0.5/−0.5), body-upper joints toward upper rows, hand joints at zero.

**Starting Point:** `baseline/`

---

## Overview

Introduce a learnable `nn.Parameter` of shape `(num_joints, num_spatial)` and pass it as `attn_mask` to the cross-attention in `_DecoderLayer.forward`, exactly as in Design 001. The difference is the initialisation: instead of all-zeros, the bias is initialised with a structured anatomical prior based on the vertical position of each joint group within a person-centred crop.

The prior encodes common sense: lower-body joints (hips, knees, ankles, feet) should attend preferentially to the lower spatial rows of the feature grid; upper-body joints (pelvis, spine, shoulders, elbows, wrists) to the upper rows; hand joints (indices 22–69) receive no prior. A smooth Gaussian profile (σ = 5 rows) is used rather than a hard cutoff to prevent suppression of any spatial token and allow gradient-based revision of the prior.

Because the prior is anatomically grounded, the model requires fewer gradient steps to establish coherent spatial routing, which is especially valuable within the 20-epoch training budget.

---

## Algorithm

### Joint Group Definitions

Hardcode the following Python lists in `pose3d_transformer_head.py` (inside `_DecoderLayer.__init__` or a helper):

```python
LOWER_BODY_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
UPPER_BODY_JOINTS = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# HAND_JOINTS = list(range(22, 70))  # receive zero prior
```

### Prior Computation (in `_DecoderLayer.__init__`)

Spatial grid dimensions: `H' = 40`, `W' = 24`, `num_spatial = 960`. Spatial tokens are laid out in row-major order: token index `i = row * W' + col`, where `row ∈ [0, 39]`, `col ∈ [0, 23]`.

Steps:
1. For each spatial token index `s = 0, ..., 959`, compute its row index: `row_s = s // 24` (integer division by `W' = 24`).
2. Compute Gaussian weights centred on `lower_center = 30.0` (rows 20–40 midpoint) and `upper_center = 10.0` (rows 0–20 midpoint), with `sigma = 5.0`:
   ```
   g_lower[s] = exp(-0.5 * ((row_s - 30.0) / 5.0)^2)
   g_upper[s] = exp(-0.5 * ((row_s - 10.0) / 5.0)^2)
   ```
3. Normalise each to [0, 1]: divide by max value (which is 1.0 since peak is at center). Then scale to `[-0.5, +0.5]`: `bias_lower[s] = g_lower[s] - 0.5` and `bias_upper[s] = g_upper[s] - 0.5`. This gives `+0.5` at the preferred spatial row and approaching `−0.5` far from it.
4. Construct `init_bias` of shape `(num_joints, num_spatial)` as `torch.zeros(num_joints, num_spatial)`.
5. For each joint index `j` in `LOWER_BODY_JOINTS`, set `init_bias[j, :] = bias_lower`.
6. For each joint index `j` in `UPPER_BODY_JOINTS`, set `init_bias[j, :] = bias_upper`.
7. Hand joints (22–69) remain zero.
8. Register: `self.cross_attn_bias = nn.Parameter(init_bias)`.

The computation must use only `torch` operations (no numpy, no scipy) to stay within the module's existing imports. Use `torch.arange`, element-wise arithmetic, and `torch.exp`.

### Forward Pass (identical to Design 001)

Assert `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` and pass `attn_mask=self.cross_attn_bias` to cross-attention.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### 1a. `_DecoderLayer.__init__` — new signature and initialisation

**Current signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
```

**New signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70, num_spatial: int = 960,
             cross_attn_bias_init: str = 'zero'):
```

- `cross_attn_bias_init`: `'zero'` (Design 001 behaviour) or `'band_prior'` (this design).

After `self.dropout2 = nn.Dropout(dropout)`, add the following block:

```python
# ── Cross-attention spatial routing bias ──────────────────────────────────
LOWER_BODY_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
UPPER_BODY_JOINTS = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
_H_prime = 40   # 640 / 16
_W_prime = 24   # 384 / 16
# assert _H_prime * _W_prime == num_spatial (implicit)

init_bias = torch.zeros(num_joints, num_spatial)

if cross_attn_bias_init == 'band_prior':
    # Row index for each spatial token
    row_idx = torch.arange(num_spatial, dtype=torch.float32) // _W_prime  # (960,)
    # Gaussian profiles
    sigma = 5.0
    g_lower = torch.exp(-0.5 * ((row_idx - 30.0) / sigma) ** 2)  # peak at row 30
    g_upper = torch.exp(-0.5 * ((row_idx - 10.0) / sigma) ** 2)  # peak at row 10
    bias_lower = g_lower - 0.5   # range ~ [-0.5, +0.5]
    bias_upper = g_upper - 0.5   # range ~ [-0.5, +0.5]
    for j in LOWER_BODY_JOINTS:
        init_bias[j] = bias_lower
    for j in UPPER_BODY_JOINTS:
        init_bias[j] = bias_upper
    # Hand joints (22–69) remain zero

self.cross_attn_bias = nn.Parameter(init_bias)
```

Important: the `torch.arange(...) // _W_prime` operation computes the row index as a float-divided value. Use integer division semantics: `torch.arange(num_spatial, dtype=torch.float32).div(_W_prime, rounding_mode='floor')` to guarantee integer row indices even in float arithmetic.

**Corrected row_idx line:**
```python
row_idx = torch.arange(num_spatial, dtype=torch.float32).div(_W_prime, rounding_mode='floor')
```

#### 1b. `_DecoderLayer.forward` — assert and pass `attn_mask` (identical to Design 001)

```python
# Cross-attention
q = self.norm2(queries)
assert spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1], (
    f'spatial_tokens length {spatial_tokens.shape[1]} != '
    f'cross_attn_bias num_spatial {self.cross_attn_bias.shape[-1]}'
)
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                     attn_mask=self.cross_attn_bias)[0]
queries = queries + self.dropout2(q2)
```

#### 1c. `Pose3dTransformerHead.__init__` — new kwargs and decoder layer construction

Add two new constructor arguments:

```python
num_spatial: int = 960,
cross_routing_type: str = 'none',
```

Map `cross_routing_type` to `cross_attn_bias_init`:

```python
_bias_init_map = {'none': 'zero', 'zero_init': 'zero', 'band_prior': 'band_prior'}
_bias_init = _bias_init_map.get(cross_routing_type, 'zero')
```

Construct decoder layer:

```python
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    num_joints=num_joints,
    num_spatial=num_spatial,
    cross_attn_bias_init=_bias_init,
)
```

Also store: `self.num_spatial = num_spatial`.

The full updated `Pose3dTransformerHead.__init__` signature:
```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_spatial: int = 960,
    cross_routing_type: str = 'none',
    loss_joints: ConfigType = dict(...),
    loss_depth: ConfigType = dict(...),
    loss_uv: ConfigType = dict(...),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    init_cfg: OptConfigType = None,
):
```

---

### 2. `config.py`

Add `num_spatial=960` and `cross_routing_type='band_prior'` to the head kwargs dict:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_spatial=960,
    cross_routing_type='band_prior',
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

Both values are plain literals (integer and string). No Python import statements needed.

---

## Exact Parameter Values

| Parameter | Value |
|-----------|-------|
| `cross_attn_bias` shape | `(70, 960)` |
| `cross_routing_type` in config | `'band_prior'` |
| `num_spatial` | `960` |
| Spatial grid | H'=40, W'=24 |
| Lower-body joint indices | `[1, 2, 4, 5, 7, 8, 10, 11]` |
| Upper-body joint indices | `[0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]` |
| Hand joint indices | `22–69` (zero prior) |
| Lower Gaussian center | row 30.0 (of 0–39) |
| Upper Gaussian center | row 10.0 (of 0–39) |
| Gaussian sigma | 5.0 rows |
| Prior scale | `±0.5` (Gaussian peak at +0.5, far-field approaches −0.5) |
| New parameters | 67,200 |
| All other hyperparameters | unchanged from baseline |

---

## Expected Behaviour

- **At init**: `cross_attn_bias` encodes the anatomical vertical prior — lower-body joints attend preferentially to lower spatial rows, upper-body joints to upper rows, hands unbiased.
- **During training**: the prior acts as a warm start; gradient updates fine-tune the routing toward the actual BEDLAM2 data distribution. Because the prior is compatible with the data distribution (crops are person-centred), few gradient steps are needed to maintain the prior, saving capacity for fine-grained routing.
- **Expected improvement**: faster convergence than Design 001; primary bet for composite_val < 160. Expected −8 to −15 mm body MPJPE improvement and maintenance or improvement of pelvis MPJPE (pelvis token 0 is in the upper-body group with a coherent spatial focus).

---

## Constraints and Invariants the Builder Must Preserve

1. **Gaussian prior values**: bias range is approximately `[−0.5, +0.5]`. Do NOT use hard binary masks (+inf/−inf) which would zero out gradients for suppressed tokens.
2. **Integer row index**: use `.div(_W_prime, rounding_mode='floor')` not `//` on float tensors to guarantee correct integer row indices.
3. **Joint group lists are hardcoded**: do not derive them programmatically from external files or imports.
4. **`cross_attn_bias` is an `nn.Parameter`**: the warm-start values must be copied into the parameter's `.data`, not recomputed at every forward. The `init_bias` tensor created in `__init__` is passed directly to `nn.Parameter(init_bias)`, which copies the data.
5. **`_init_head_weights` must NOT overwrite `cross_attn_bias`**: the structured init must persist. Do not add `cross_attn_bias` to the list of modules initialized in `_init_head_weights`.
6. **Assert at forward**: same shape assertion as Design 001 must be present.
7. **`attn_mask` is additive** (not multiplicative): PyTorch convention. Positive values strengthen attention; negative values suppress it.
8. **No changes to**: self-attention, loss, `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or invariant files.
9. **No Python import statements in `config.py`**: all values are string/integer literals.
10. **No changes to `pelvis_utils.py`**.
11. **`cross_routing_type='none'`** must recover exact baseline behaviour (zero init, same as Design 001 with `'zero_init'`).
