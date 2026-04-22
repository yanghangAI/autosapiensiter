# Design 002 — Decoupled pelvis decoder, 4-head lightweight, full 70-query joint decoder

**Design Description:** Same as design001 but the dedicated `_PelvisCrossAttnDecoder` uses 4 attention heads instead of 8, testing whether a lighter pelvis localization head converges better for a 2-output regression task.

**Starting Point:** `baseline/`

---

## Overview

Identical to design001 except `pelvis_num_heads=4`. The algorithm change is the same as design001 (dedicated pelvis cross-attn decoder decoupled from joint self-attention), with the sole difference that the pelvis coordinate decoder has 4 attention heads instead of 8. The pelvis coordinate decoder has only 2 output targets (depth + UV), which is a much simpler task than 22-joint body regression. A 4-head cross-attention module may generalize better by using fewer parameters for this low-complexity localization problem, while still providing sufficient capacity to route attention to relevant spatial regions.

The joint decoder remains unchanged (70 queries, 8-head self-attn + cross-attn + FFN). All changes are confined to the dedicated pelvis decoder's attention head count.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Identical to design001 in all respects**, including the `_PelvisCrossAttnDecoder` class definition, the new kwargs in `__init__`, the `_init_head_weights()` changes, and the `forward()` conditional pelvis path.

The only runtime difference from design001 is that `_PelvisCrossAttnDecoder.__init__` is instantiated with `num_heads=4` (passed via `pelvis_num_heads=4` from config). The `nn.MultiheadAttention(embed_dim=256, num_heads=4)` inside `_PelvisCrossAttnDecoder` requires `embed_dim % num_heads == 0`, i.e., `256 % 4 == 0` — satisfied.

The full set of implementation changes is identical to design001:

**Add `_PelvisCrossAttnDecoder` at module level after `_DecoderLayer`:**

```python
class _PelvisCrossAttnDecoder(nn.Module):
    """Lightweight cross-attention decoder for pelvis coordinate queries.

    No self-attention, no FFN — purely cross-attends to spatial tokens.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor,
                spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, 2, embed_dim)
            spatial_tokens: (B, H'*W', embed_dim)
        Returns:
            (B, 2, embed_dim)
        """
        q = self.norm(queries)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
        return queries + q2
```

**New kwargs in `Pose3dTransformerHead.__init__` (after `loss_weight_uv`, before `init_cfg`):**

```python
use_decoupled_pelvis: bool = False,
pelvis_hidden_dim: int = 256,
pelvis_num_heads: int = 8,
num_body_queries: int = 70,
```

Store as instance attributes:
```python
self.use_decoupled_pelvis = use_decoupled_pelvis
self.pelvis_hidden_dim = pelvis_hidden_dim
self.num_body_queries = num_body_queries
```

Conditional module construction after `self.uv_out`:
```python
if use_decoupled_pelvis:
    self.pelvis_coord_queries = nn.Embedding(2, pelvis_hidden_dim)
    self.pelvis_decoder = _PelvisCrossAttnDecoder(
        pelvis_hidden_dim, pelvis_num_heads, dropout)
```

**`_init_head_weights()` addition:**
```python
if self.use_decoupled_pelvis:
    nn.init.trunc_normal_(self.pelvis_coord_queries.weight, std=0.02)
```

**`forward()` conditional pelvis path:**
```python
if self.use_decoupled_pelvis:
    pelvis_qs = self.pelvis_coord_queries.weight.unsqueeze(0).expand(B, -1, -1)
    pelvis_decoded = self.pelvis_decoder(pelvis_qs, spatial)  # (B, 2, pelvis_hidden_dim)
    pelvis_depth = self.depth_out(pelvis_decoded[:, 0, :])    # (B, 1)
    pelvis_uv    = self.uv_out(pelvis_decoded[:, 1, :])       # (B, 2)
else:
    pelvis_token = decoded[:, 0, :]
    pelvis_depth = self.depth_out(pelvis_token)
    pelvis_uv = self.uv_out(pelvis_token)
```

**`loss()` and `predict()` methods:** No changes.

### 2. `config.py`

In `model.head`, set `pelvis_num_heads=4` (the only difference from design001):

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                     loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                    loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                 loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    use_decoupled_pelvis=True,
    pelvis_hidden_dim=256,
    pelvis_num_heads=4,
    num_body_queries=70,
),
```

All other config values are **identical to baseline**.

### 3. `pelvis_utils.py`

No changes.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. `self.num_joints = 70` must remain set in `__init__`.
3. `_PelvisCrossAttnDecoder` must be defined at module level, before `Pose3dTransformerHead`.
4. `embed_dim=256`, `num_heads=4`: `256 % 4 == 0` — valid for `nn.MultiheadAttention`. The Builder must verify this at implementation time (it holds for this design).
5. `pelvis_coord_queries` embedding: size `(2, 256)`. Index 0 = depth query, index 1 = UV query.
6. `pelvis_hidden_dim=256` must equal `hidden_dim=256` — required for `depth_out` and `uv_out` input compatibility.
7. `depth_out` and `uv_out` remain re-initialized by the existing `_init_head_weights()` loop — no separate action needed.
8. `use_decoupled_pelvis=False` default preserved for backward compatibility.
9. `num_body_queries=70` for this design — joint query embedding `nn.Embedding(70, hidden_dim)` unchanged.
10. Joint decoder self-attn and cross-attn unchanged.
11. `_BODY = list(range(0, 22))` — unchanged.
12. MMEngine config: all four new kwargs are bool/int literals. No import statements. Fully compliant.
13. Seed `2026`, batch size `4`, accumulation `8` — do not change.
14. `max_keep_ckpts=1`, `resume=True` — do not change.

---

## Expected Behavior After Change

- Functionally identical to design001 except the pelvis cross-attention uses 4 heads. Each head attends over `256/4=64`-dimensional key/query/value projections.
- With fewer heads, each head has wider per-head dimension (64 vs. 32 in design001). This means each head can represent richer individual spatial patterns — potentially better for the simple 2-output localization task.
- The ~263K parameter overhead from design001 reduces to ~197K (4-head vs. 8-head `nn.MultiheadAttention(256, ·)`).
- Target stage-1: `mpjpe_pelvis_val < 580 mm`, `composite_val < 340`. Design B may outperform Design A if the simpler head avoids overfitting.
- Output shapes: `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)` — identical to baseline.
