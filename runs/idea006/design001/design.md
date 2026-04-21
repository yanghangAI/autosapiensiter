# Design 001 — Shared Learnable Attention Bias, Zero Init (Design A)

**Design Description:** Add a single `(70, 70)` learnable additive attention bias parameter, zero-initialized, passed as `attn_mask` to the query self-attention in `_DecoderLayer`.

**Starting Point:** `baseline/`

---

## Overview

Introduce a learnable `nn.Parameter` of shape `(num_joints, num_joints)` initialized to all zeros and pass it as `attn_mask` to `self.self_attn(...)` in `_DecoderLayer.forward`. Because it is zero-initialized, the model starts training identically to the baseline. Over training, gradient updates push entries positive (joints that should attend to each other) or negative (joints that should be suppressed from attending). This is the minimal-change diagnostic variant.

## Algorithm

The algorithm modification is a single additive attention bias injected into the existing self-attention mechanism of `_DecoderLayer`:

1. Register `self.attn_bias = nn.Parameter(torch.zeros(J, J))` where `J = num_joints = 70`.
2. At every forward pass, pass `attn_mask=self.attn_bias` to `nn.MultiheadAttention.forward`. PyTorch adds this `(J, J)` matrix element-wise to the raw attention logit matrix (before softmax), broadcasting over batch and head dimensions.
3. The modified attention computation is: `Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + attn_bias) V`, where `attn_bias` is the `(J, J)` learnable parameter.
4. Gradients flow through the softmax back into `attn_bias`, pushing entries toward positive (attended) or negative (suppressed) values based on training signal.
5. No other algorithmic changes — loss function, cross-attention, FFN, backbone, and data pipeline are unchanged.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### 1a. `_DecoderLayer.__init__` — add `attn_bias` parameter

**Current signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
```

**New signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70):
```

After `self.dropout2 = nn.Dropout(dropout)` and before the method ends, add:

```python
# Learnable additive bias for self-attention logits (shared across heads and batch)
self.attn_bias = nn.Parameter(torch.zeros(num_joints, num_joints))
```

No other changes to `__init__`.

#### 1b. `_DecoderLayer.forward` — pass `attn_mask`

**Current self-attention call (line 113):**
```python
q2 = self.self_attn(q, q, q)[0]
```

**Replace with:**
```python
q2 = self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]
```

No other changes to `forward`.

#### 1c. `Pose3dTransformerHead.__init__` — pass `num_joints` to `_DecoderLayer`

**Current decoder layer construction (line 185):**
```python
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```

**Replace with:**
```python
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout, num_joints=num_joints)
```

No other changes to `Pose3dTransformerHead`.

---

### 2. `config.py`

No changes required. The head config in `config.py` does not need a new kwarg because `attn_bias` is always registered in this design (the zero-init is unconditional). The baseline config remains fully compatible.

---

## Exact Parameter Values

| Parameter | Value |
|-----------|-------|
| `attn_bias` shape | `(70, 70)` |
| `attn_bias` init | `torch.zeros(70, 70)` |
| `attn_mask` dtype | float32 (same as model) |
| Number of new parameters | 4900 |
| `attn_mask` semantics | additive to attention logits before softmax (PyTorch convention) |

---

## Expected Behaviour

- **At init**: `attn_bias` is all zeros → no change to attention logits → identical forward pass and initial loss to baseline.
- **During training**: gradient flows through the softmax into `attn_bias` entries; pairs of joints with strong correlations will accumulate positive bias, uncorrelated or harmful pairs will accumulate negative bias.
- **Convergence**: expected −5 to −10 mm improvement in body MPJPE by epoch 20. Pelvis MPJPE unchanged or slightly improved.

---

## Constraints and Invariants the Builder Must Preserve

1. **Baseline-identical start**: `attn_bias` must be `nn.Parameter(torch.zeros(...))`. Do NOT use `nn.init.normal_` or any nonzero init in Design 001.
2. **`attn_mask` is additive, not multiplicative**: `nn.MultiheadAttention(attn_mask=...)` adds the mask to attention logits before softmax. Positive → stronger attention; large negative → suppression. This is the PyTorch default; do not set `key_padding_mask` or `is_causal`.
3. **Shape**: `attn_mask` must be exactly `(num_joints, num_joints)` = `(70, 70)`. PyTorch broadcasts this across the batch and head dimensions automatically.
4. **Dtype**: `attn_bias` must be on the same device and dtype as the model. Because it is an `nn.Parameter`, `.to(device)` / `.half()` calls during model setup handle this automatically.
5. **No changes to loss, metric, data pipeline, backbone, `pelvis_utils.py`, or any invariant files.**
6. **No Python import statements in `config.py`** — this design adds no new config keys, so this constraint is trivially satisfied.
7. **`_init_head_weights` must NOT initialize `attn_bias`**: it is already zero-initialized in `nn.Parameter(torch.zeros(...))`. Adding it to `_init_head_weights` would be redundant but not harmful; however, leave it out for clarity.
8. **`batch_first=True` is already set** on `self.self_attn` in the baseline — the `(T_q, T_k)` attn_mask broadcast is correct for this layout.
