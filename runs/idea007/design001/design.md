# Design 001 — Zero-Initialised Learnable Cross-Attention Routing (Design A)

**Design Description:** Add a single `(num_joints, num_spatial)` learnable additive bias parameter, zero-initialized, passed as `attn_mask` to cross-attention in `_DecoderLayer`.

**Starting Point:** `baseline/`

---

## Overview

Introduce a learnable `nn.Parameter` of shape `(num_joints, num_spatial)` initialized to all zeros and pass it as `attn_mask` to `self.cross_attn(...)` in `_DecoderLayer.forward`. Because it is zero-initialized, the model starts training identically to the baseline. Over training, gradient updates push entries positive (joint-spatial pairs that should receive stronger attention) or negative (pairs to suppress). This is the minimal-change diagnostic variant — it tests whether any learned spatial routing of cross-attention is beneficial beyond what the content-based dot-product routing already achieves.

`num_spatial = 960` because the input is 640×384 at 1/16 stride → H' = 40, W' = 24, H'×W' = 960. This is hardcoded as the default but exposed as a constructor kwarg for safety.

---

## Algorithm

1. In `_DecoderLayer.__init__`, register `self.cross_attn_bias = nn.Parameter(torch.zeros(num_joints, num_spatial))`.
2. In `_DecoderLayer.forward`, before calling `self.cross_attn`, assert `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` to catch shape mismatches early.
3. Pass `attn_mask=self.cross_attn_bias` to the cross-attention call. PyTorch adds this `(J, S)` matrix element-wise to the raw attention logit matrix (before softmax), broadcasting over batch and head dimensions automatically.
4. The modified cross-attention computation: `Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + cross_attn_bias) V`, where `cross_attn_bias` is the `(num_joints, num_spatial)` learnable parameter.
5. No other algorithmic changes — self-attention, loss, backbone, data pipeline, and `pelvis_utils.py` are unchanged.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### 1a. `_DecoderLayer.__init__` — new signature and registration

**Current signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
```

**New signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70, num_spatial: int = 960):
```

After `self.dropout2 = nn.Dropout(dropout)` (line 99 in baseline), add:

```python
# Learnable additive bias for cross-attention logits (shared across heads and batch)
self.cross_attn_bias = nn.Parameter(torch.zeros(num_joints, num_spatial))
```

#### 1b. `_DecoderLayer.forward` — assert and pass `attn_mask`

**Current cross-attention block (lines 117–119 in baseline):**
```python
# Cross-attention
q = self.norm2(queries)
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```

**Replace with:**
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

#### 1c. `Pose3dTransformerHead.__init__` — pass `num_spatial` to `_DecoderLayer`

Add `num_spatial: int = 960` as a constructor argument to `Pose3dTransformerHead.__init__`.

**Current decoder layer construction (line 185 in baseline):**
```python
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```

**Replace with:**
```python
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout,
                                   num_joints=num_joints,
                                   num_spatial=num_spatial)
```

Also store the kwarg: add `self.num_spatial = num_spatial` after `self.num_joints = num_joints`.

The full updated `Pose3dTransformerHead.__init__` signature becomes:
```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_spatial: int = 960,
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

Add `num_spatial=960` to the head kwargs dict:

**Current head dict (in `model = dict(...)`):**
```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    ...
),
```

**New head dict:**
```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_spatial=960,
    ...
),
```

`num_spatial=960` is a plain integer literal. No Python imports needed. No other config changes.

---

## Exact Parameter Values

| Parameter | Value |
|-----------|-------|
| `cross_attn_bias` shape | `(70, 960)` |
| `cross_attn_bias` init | `torch.zeros(70, 960)` |
| `num_spatial` default | `960` |
| Number of new parameters | 67,200 |
| `attn_mask` semantics | additive to cross-attention logits before softmax (PyTorch convention) |
| All other hyperparameters | unchanged from baseline |

---

## Expected Behaviour

- **At init**: `cross_attn_bias` is all zeros → identical forward pass to baseline.
- **During training**: gradient flows through softmax into `cross_attn_bias`; joint-spatial pairs with strong correlations accumulate positive bias; irrelevant pairs accumulate negative bias. Each joint query learns a global spatial preference per spatial token.
- **Expected improvement**: −5 to −10 mm body MPJPE by epoch 20. Pelvis MPJPE unchanged or slightly improved.

---

## Constraints and Invariants the Builder Must Preserve

1. **Baseline-identical start**: `cross_attn_bias` must be `nn.Parameter(torch.zeros(num_joints, num_spatial))`. Do NOT use any nonzero init in Design 001.
2. **`attn_mask` is additive**: `nn.MultiheadAttention(attn_mask=...)` adds the mask to cross-attention logits before softmax. Positive → stronger attention; large negative → suppression. Do not set `key_padding_mask` or `is_causal`.
3. **Shape**: `attn_mask` must be exactly `(num_joints, num_spatial)` = `(70, 960)` for the baseline input resolution. PyTorch broadcasts over batch and head dimensions automatically when `batch_first=True`.
4. **Dtype/device**: `cross_attn_bias` is `nn.Parameter`, so `.to(device)` / `.half()` calls during model setup handle placement automatically. No manual casting needed.
5. **Assert at forward**: the assertion `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` must be present to catch stride/resolution mismatches at runtime, not silently propagate wrong shapes.
6. **No changes to**: self-attention, loss, `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or any invariant file.
7. **No Python import statements in `config.py`**: `num_spatial=960` is a plain integer literal; compliant.
8. **`_init_head_weights` must NOT initialize `cross_attn_bias`**: it is already zero-initialized via `nn.Parameter(torch.zeros(...))`. Leave it out of `_init_head_weights`.
9. **`batch_first=True` is set on `self.cross_attn`** in the baseline — the `(T_q, T_k)` = `(num_joints, num_spatial)` shape is correct for this layout.
10. **No changes to `pelvis_utils.py`**.
