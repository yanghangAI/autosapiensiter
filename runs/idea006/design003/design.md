# Design 003 — Per-Head Attention Bias, Zero Init (Design C)

**Design Description:** Add `num_heads` independent `(70, 70)` learnable attention bias matrices, all zero-initialized, expanded to `(B * num_heads, 70, 70)` and passed as `attn_mask` to query self-attention, giving each head a separate structural specialisation.

**Starting Point:** `baseline/`

---

## Overview

Instead of a single shared bias matrix (Design 001), register `num_heads` independent bias matrices

## Algorithm

The algorithm modification introduces per-head learnable attention biases into the self-attention of `_DecoderLayer`:

1. Register `self.attn_bias = nn.Parameter(torch.zeros(H, J, J))` where `H = num_heads = 8`, `J = num_joints = 70`.
2. At every forward pass, expand `attn_bias` to shape `(B * H, J, J)` by: `self.attn_bias.unsqueeze(0).expand(B, -1, -1, -1).contiguous().reshape(B * H, J, J)`.
3. Pass the expanded tensor as `attn_mask` to `nn.MultiheadAttention.forward`. PyTorch applies slice `[b * H + h]` to batch `b`, head `h`, giving each head an independent structural specialisation.
4. Modified attention: `Attention_h(Q, K, V) = softmax((Q_h K_h^T / sqrt(d_k)) + attn_bias[h]) V_h` for each head `h`.
5. All 8 heads start with zero bias (baseline-identical) and independently evolve their joint-to-joint attention patterns from gradient signal.
6. No other algorithmic changes — loss function, cross-attention, FFN, backbone, and data pipeline are unchanged. of shape `(num_heads, num_joints, num_joints)`, all initialized to zero. At forward time, expand the parameter to `(B * num_heads, num_joints, num_joints)` using `.repeat(B, 1, 1)` (after reshaping) and pass it as `attn_mask`. This allows each of the 8 attention heads to independently learn a different structural specialisation (e.g., one head for kinematic chains, one for symmetry, one for pelvis routing). Parameter cost: `8 × 70 × 70 = 39200` scalars (~157 KB float32).

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### 1a. `_DecoderLayer.__init__` — add per-head `attn_bias` parameter

**New signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70, attn_bias_mode: str = 'none'):
```

Valid values for `attn_bias_mode`:
- `'none'`: no learnable bias (baseline behaviour)
- `'shared'`: single `(num_joints, num_joints)` bias (Design 001/002 mode)
- `'per_head'`: `(num_heads, num_joints, num_joints)` bias (this design)

Store `num_heads` as an attribute:
```python
self.num_heads = num_heads
self.attn_bias_mode = attn_bias_mode
```

After `self.dropout2 = nn.Dropout(dropout)` and before the method ends, add:

```python
if attn_bias_mode == 'per_head':
    self.attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_joints))
elif attn_bias_mode == 'shared':
    self.attn_bias = nn.Parameter(torch.zeros(num_joints, num_joints))
else:
    self.attn_bias = None
```

#### 1b. `_DecoderLayer.forward` — expand per-head bias and pass as `attn_mask`

**New signature:**
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor) -> torch.Tensor:
```

(Signature unchanged; `B` is read from `queries.shape[0]` inside the method.)

**In the self-attention block, replace:**
```python
q2 = self.self_attn(q, q, q)[0]
```

**With:**
```python
if self.attn_bias_mode == 'per_head':
    B = queries.shape[0]
    # self.attn_bias: (num_heads, J, J) → expand to (B * num_heads, J, J)
    _mask = self.attn_bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(
        B * self.num_heads, queries.shape[1], queries.shape[1])
    q2 = self.self_attn(q, q, q, attn_mask=_mask)[0]
elif self.attn_bias_mode == 'shared':
    q2 = self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]
else:
    q2 = self.self_attn(q, q, q)[0]
```

**Important implementation detail for the expand/reshape:**

`self.attn_bias` has shape `(num_heads, J, J)` where `num_heads=8, J=70`.

The expand sequence:
1. `.unsqueeze(0)` → `(1, num_heads, J, J)`
2. `.expand(B, -1, -1, -1)` → `(B, num_heads, J, J)`
3. `.reshape(B * num_heads, J, J)` → `(B * num_heads, J, J)`

PyTorch's `nn.MultiheadAttention` with `batch_first=True` and `attn_mask` of shape `(B * num_heads, T, T)` applies each slice to the corresponding (batch, head) combination. This is the per-head broadcast semantics required.

**Note**: `.expand()` returns a non-contiguous tensor; `.reshape()` will call `.contiguous()` implicitly if needed, or the Builder can explicitly call `.contiguous().reshape(...)` to be safe.

#### 1c. `Pose3dTransformerHead.__init__` — pass `attn_bias_mode` to `_DecoderLayer`

Add `attn_bias_type` parameter to `Pose3dTransformerHead.__init__`:

**New parameter** (add after `loss_weight_uv: float = 1.0`):
```python
attn_bias_type: str = 'none',
```

**Replace:**
```python
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```

**With:**
```python
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    num_joints=num_joints,
    attn_bias_mode=attn_bias_type)
```

(Note: `attn_bias_type` from the head config maps to `attn_bias_mode` in `_DecoderLayer`. The naming mismatch is intentional to keep `_DecoderLayer`'s internal API separate from the MMEngine config key.)

---

### 2. `config.py`

In the `head` dict inside `model`, add `attn_bias_type` as a string literal:

**New head dict:**
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
    attn_bias_type='per_head',
),
```

---

## Exact Parameter Values

| Parameter | Value |
|-----------|-------|
| `attn_bias` shape | `(8, 70, 70)` |
| `attn_bias` init | `torch.zeros(8, 70, 70)` |
| Number of new parameters | 39200 |
| Expanded `attn_mask` shape at forward | `(B * 8, 70, 70)` |
| `attn_bias_type` in config | `'per_head'` (string literal) |
| `num_heads` | 8 (same as baseline) |
| `num_joints` (J) | 70 |

---

## Expected Behaviour

- **At init**: all bias entries are zero → identical to baseline forward pass.
- **During training**: each of the 8 heads independently learns a separate joint-to-joint attention pattern. Expected specialisation: some heads may encode kinematic chains, others body symmetry, others pelvis routing away from body-joint tokens.
- **Convergence**: highest ceiling of the three designs, but may need most epochs to converge from scratch. Within the 20-epoch budget, expected −5 to −15 mm improvement on body MPJPE, with potential pelvis improvement if any head specialises in routing pelvis token 0 away from body interactions.
- **Target**: composite_val < 162 (vs. baseline 169.75).

---

## Constraints and Invariants the Builder Must Preserve

1. **Zero-init**: `attn_bias` must be `nn.Parameter(torch.zeros(num_heads, num_joints, num_joints))`. No other initialisation for this design.
2. **`B` must be read from `queries.shape[0]`** inside `_DecoderLayer.forward`, not passed as an argument. The decoder layer signature does not change.
3. **Contiguity**: after `.expand()`, call `.contiguous().reshape(B * self.num_heads, queries.shape[1], queries.shape[1])` (not `.view()`) to avoid non-contiguous tensor errors. `.reshape()` is equivalent to `.contiguous().view()` when needed.
4. **`attn_mask` dtype**: must match the dtype of the query tensor. Since `attn_bias` is `nn.Parameter` (float32) and the model may run in float32, this is automatically satisfied. If AMP (mixed precision) is enabled, PyTorch will cast the attn_mask internally — do not manually cast.
5. **`attn_bias_mode` string values**: only `'per_head'`, `'shared'`, `'none'` are valid. The `config.py` uses `attn_bias_type='per_head'` which maps to `attn_bias_mode='per_head'` in `_DecoderLayer`. The Builder must ensure the mapping is correct (no typo in the string comparison).
6. **`self.attn_bias = None` for mode `'none'`**: when `attn_bias_mode='none'`, `self.attn_bias` is set to `None` (not an `nn.Parameter`). This means `self.parameters()` will not include it, which is correct — the baseline has no extra parameters.
7. **No changes to loss, metric, data pipeline, backbone, `pelvis_utils.py`, or any invariant files.**
8. **No Python import statements in `config.py`**: `attn_bias_type='per_head'` is a string literal. Fully compliant.
9. **`batch_first=True` is already set** on `self.self_attn`. When `attn_mask` shape is `(B * num_heads, T, T)`, PyTorch distributes slices across heads in order: `attn_mask[b * num_heads + h]` is applied to batch `b`, head `h`. The `.unsqueeze(0).expand(B, ...).reshape(B * num_heads, ...)` interleaving matches this order because `expand` broadcasts the batch dimension *outside* the head dimension, so reshape produces `[b0h0, b0h1, ..., b0h7, b1h0, ..., b(B-1)h7]` — exactly the expected layout.
