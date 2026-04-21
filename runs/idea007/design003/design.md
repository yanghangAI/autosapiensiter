# Design 003 — Per-Head Cross-Attention Routing (Design C)

**Design Description:** Learn `num_heads` independent cross-attention routing bias matrices of shape `(num_joints, num_spatial)`, zero-initialized, expanded to `(B*num_heads, num_joints, num_spatial)` at forward and passed as `attn_mask` to cross-attention.

**Starting Point:** `baseline/`

---

## Overview

This is the richest variant of the spatial routing idea. Instead of a single shared bias matrix, each of the 8 attention heads learns an independent `(num_joints, num_spatial)` routing bias. This allows different heads to develop complementary spatial routing specialisations — one head might route lower-body joints to lower spatial rows, another might route the pelvis query (token 0) to the centre of the crop regardless of body group, and others might learn global or hand-specific routing patterns.

The parameter cost is `8 × 70 × 960 = 537,600` scalars (~2 MB), still negligible relative to the backbone.

`nn.MultiheadAttention` with `batch_first=True` accepts `attn_mask` of shape `(B*num_heads, T_q, T_k)` to apply head-specific masks. The `B` (batch size) dimension must be available in `_DecoderLayer.forward`, which requires a minor refactor: pass `B` as an argument from `Pose3dTransformerHead.forward`.

All biases are zero-initialized (Design A base) — no warm-start in Design C. The per-head structure provides sufficient learning flexibility.

---

## Algorithm

1. In `_DecoderLayer.__init__`, register `self.cross_attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_spatial))`. Shape: `(8, 70, 960)`.
2. In `_DecoderLayer.forward`, accept an additional `B: int` argument for the batch size.
3. Before cross-attention, expand the bias: `bias_expanded = self.cross_attn_bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_heads, num_joints, num_spatial)`. This gives shape `(B*8, 70, 960)`, which PyTorch `nn.MultiheadAttention` interprets as a per-sample-per-head additive mask.
4. Assert `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]`.
5. Pass `attn_mask=bias_expanded` to `self.cross_attn(q, spatial_tokens, spatial_tokens, attn_mask=bias_expanded)`.
6. In `Pose3dTransformerHead.forward`, extract `B` from the feature tensor and pass it to `self.decoder_layer(queries, spatial, B=B)`.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### 1a. `_DecoderLayer.__init__` — store `num_heads` and register per-head bias

**Current signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
```

**New signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70, num_spatial: int = 960):
```

After `self.dropout2 = nn.Dropout(dropout)`, add:

```python
# Store num_heads for bias expansion in forward
self._num_heads = num_heads
# Per-head learnable additive bias for cross-attention logits
self.cross_attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_spatial))
```

Note: `num_heads` is already stored implicitly inside `self.cross_attn` but is not exposed as a public attribute in `nn.MultiheadAttention` in all PyTorch versions. Store it explicitly as `self._num_heads = num_heads` for use in `forward`.

#### 1b. `_DecoderLayer.forward` — accept `B` and expand bias

**Current signature:**
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor) -> torch.Tensor:
```

**New signature:**
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            B: int = 1) -> torch.Tensor:
```

**Current cross-attention block:**
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
# Expand per-head bias to (B * num_heads, num_joints, num_spatial)
bias_expanded = (
    self.cross_attn_bias                     # (num_heads, J, S)
    .unsqueeze(0)                            # (1, num_heads, J, S)
    .expand(B, -1, -1, -1)                  # (B, num_heads, J, S)
    .reshape(B * self._num_heads,
             self.cross_attn_bias.shape[1],
             self.cross_attn_bias.shape[2]) # (B*H, J, S)
)
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                     attn_mask=bias_expanded)[0]
queries = queries + self.dropout2(q2)
```

#### 1c. `Pose3dTransformerHead.__init__` — new kwargs and decoder layer construction

Add `num_spatial: int = 960` and `cross_routing_type: str = 'none'` to constructor. For Design C, `cross_routing_type='per_head'`. The `_DecoderLayer` is constructed as:

```python
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    num_joints=num_joints,
    num_spatial=num_spatial,
)
```

Note: Design C uses the same `_DecoderLayer.__init__` signature as Designs 001/002 for its own `num_joints`/`num_spatial` args. The distinction is that in Design C the bias shape is `(num_heads, num_joints, num_spatial)` — this is selected by the kwarg approach.

**Revised approach for clean single implementation**: to avoid branching in `_DecoderLayer.__init__` between single-bias (Designs 001/002) and per-head-bias (Design 003), add a `per_head_routing: bool = False` kwarg to `_DecoderLayer`:

**Updated `_DecoderLayer.__init__` signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70, num_spatial: int = 960,
             cross_attn_bias_init: str = 'zero',
             per_head_routing: bool = False):
```

When `per_head_routing=True`:
```python
self._per_head = True
self._num_heads = num_heads
self.cross_attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_spatial))
```

When `per_head_routing=False` (Designs 001/002):
```python
self._per_head = False
# existing band_prior or zero init as in Design 002
self.cross_attn_bias = nn.Parameter(init_bias)  # shape (num_joints, num_spatial)
```

In `_DecoderLayer.forward`, branch on `self._per_head`:
```python
if self._per_head:
    bias_expanded = (
        self.cross_attn_bias
        .unsqueeze(0)
        .expand(B, -1, -1, -1)
        .reshape(B * self._num_heads,
                 self.cross_attn_bias.shape[1],
                 self.cross_attn_bias.shape[2])
    )
    q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                         attn_mask=bias_expanded)[0]
else:
    q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                         attn_mask=self.cross_attn_bias)[0]
```

In `Pose3dTransformerHead.__init__`, map `cross_routing_type='per_head'` to `per_head_routing=True`:

```python
_per_head = (cross_routing_type == 'per_head')
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    num_joints=num_joints,
    num_spatial=num_spatial,
    cross_attn_bias_init='zero',
    per_head_routing=_per_head,
)
```

In `Pose3dTransformerHead.forward`, pass `B` to the decoder layer:

**Current decoder call (line 248 in baseline):**
```python
decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)
```

**Replace with:**
```python
decoded = self.decoder_layer(queries, spatial, B=B)  # (B, num_joints, hidden_dim)
```

`B` is already available from `B, C, H, W = feat.shape` earlier in `forward`.

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

Add `num_spatial=960` and `cross_routing_type='per_head'` to the head kwargs dict:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_spatial=960,
    cross_routing_type='per_head',
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

Both values are plain literals. No Python import statements needed.

---

## Exact Parameter Values

| Parameter | Value |
|-----------|-------|
| `cross_attn_bias` shape | `(8, 70, 960)` |
| `cross_attn_bias` init | `torch.zeros(8, 70, 960)` |
| `num_spatial` | `960` |
| `num_heads` | `8` |
| `cross_routing_type` in config | `'per_head'` |
| Expanded `attn_mask` shape at forward | `(B*8, 70, 960)` |
| Number of new parameters | 537,600 (~2 MB) |
| All other hyperparameters | unchanged from baseline |

---

## Expected Behaviour

- **At init**: `cross_attn_bias` is all zeros → identical forward pass to baseline.
- **During training**: each of the 8 heads independently learns a spatial routing bias. Different heads can specialise: e.g., one head reinforces lower-body spatial focus for hip/knee joints, another focuses the pelvis token on the crop centre, etc.
- **Expected improvement**: highest potential of the three designs; each head can contribute complementary routing. May require the full 20 epochs to converge. Expected composite_val < 162. If convergence is slow, Design B (warm-start) is the more reliable bet.

---

## Constraints and Invariants the Builder Must Preserve

1. **Zero init only**: `cross_attn_bias` must be `nn.Parameter(torch.zeros(num_heads, num_joints, num_spatial))`. No warm-start in Design C.
2. **`B` must be extracted from features, not hardcoded**: use `B, C, H, W = feat.shape` already present in `Pose3dTransformerHead.forward`. Pass `B=B` explicitly to `_DecoderLayer.forward`.
3. **`expand` not `repeat`**: use `.expand(B, -1, -1, -1)` to avoid copying data unnecessarily. PyTorch's `nn.MultiheadAttention` reads the expanded view correctly.
4. **Shape of expanded bias**: must be `(B * num_heads, num_joints, num_spatial)` = `(B*8, 70, 960)`. Passing shape `(num_heads, num_joints, num_spatial)` without batch expansion will cause a shape mismatch inside `nn.MultiheadAttention` with `batch_first=True`.
5. **`_per_head` flag and `_num_heads`**: both must be stored as instance attributes in `_DecoderLayer.__init__` for use in `forward`.
6. **Assert at forward**: `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` must be present (check last dim of the per-head bias tensor).
7. **Backward compatibility**: when `cross_routing_type='none'` (baseline), `_per_head=False` and `cross_attn_bias` has shape `(num_joints, num_spatial)` with zero init — identical to Design 001 and the baseline.
8. **`_init_head_weights` must NOT initialize `cross_attn_bias`**: zero init via `nn.Parameter(torch.zeros(...))` is correct; do not add it to `_init_head_weights`.
9. **No changes to**: self-attention, loss, `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or invariant files.
10. **No Python import statements in `config.py`**: all values are string/integer literals.
11. **No changes to `pelvis_utils.py`**.
12. **`B=1` default in `_DecoderLayer.forward`**: the default `B=1` must be present so existing callers that omit `B` do not break. However, `Pose3dTransformerHead.forward` must always pass `B=B` explicitly when `per_head_routing=True` — a wrong `B` would silently produce wrong shapes. The builder should add a comment noting this.
