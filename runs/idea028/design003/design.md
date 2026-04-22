# Design 003 — Decoupled pelvis decoder + body-only 22-query joint decoder (combined)

**Design Description:** Combine decoupled 8-head pelvis cross-attn decoder (from design001) with 22-query body-only joint decoder (from idea008/design001); joint self-attention is 22×22, pelvis depth/UV from dedicated queries, hand joints zero-padded.

**Starting Point:** `baseline/`

---

## Overview

This design combines two independently validated algorithm changes:

1. **Body-only 22-query joint decoder** (idea008/design001 mechanism): joint self-attention operates on 22×22 query pairs instead of 70×70, removing hand-query contamination from body joint predictions.

2. **Decoupled pelvis coordinate queries** (this idea, design001 mechanism): `depth_out` and `uv_out` read from two dedicated pelvis queries that run an independent cross-attention pass, not from joint token 0.

The result is a joint decoder with a fully clean self-attention space: 22 body queries, no hand contamination, and no pelvis-localization objective. The pelvis decoder has no interaction with joint self-attention whatsoever.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Add `_PelvisCrossAttnDecoder` at module level (after `_DecoderLayer`), identical to design001:**

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

**`Pose3dTransformerHead.__init__` changes:**

Add four new kwargs (after `loss_weight_uv`, before `init_cfg`):

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

Change the joint query embedding to use `num_body_queries` instead of `num_joints`:

**Before:**
```python
self.joint_queries = nn.Embedding(num_joints, hidden_dim)
```

**After:**
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

Note: `self.num_joints = num_joints` (= 70) must still be set (already done in baseline) — needed for `predict()` shape.

After `self.uv_out = nn.Linear(hidden_dim, 2)`, add conditional block:
```python
if use_decoupled_pelvis:
    self.pelvis_coord_queries = nn.Embedding(2, pelvis_hidden_dim)
    self.pelvis_decoder = _PelvisCrossAttnDecoder(
        pelvis_hidden_dim, pelvis_num_heads, dropout)
```

**Updated constructor signature (full, for reference):**

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
    loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                  beta=0.05, loss_weight=1.0),
    loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                               beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    use_decoupled_pelvis: bool = False,
    pelvis_hidden_dim: int = 256,
    pelvis_num_heads: int = 8,
    num_body_queries: int = 70,
    init_cfg: OptConfigType = None,
):
```

**`_init_head_weights()` changes:**

The existing body initializes `self.joint_queries.weight` — this still applies but now the embedding has shape `(22, hidden_dim)` rather than `(70, hidden_dim)`. The initialization call is identical; the shape is handled by the embedding definition.

Add after the existing loop:
```python
if self.use_decoupled_pelvis:
    nn.init.trunc_normal_(self.pelvis_coord_queries.weight, std=0.02)
```

**`forward()` method changes:**

The query broadcast line changes shape due to `num_body_queries=22`:
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
# shape: (B, 22, hidden_dim)  — was (B, 70, hidden_dim)
```

The decoder now operates on 22 queries:
```python
decoded = self.decoder_layer(queries, spatial)  # (B, 22, hidden_dim)
```

Joint output from the 22-query decoder:
```python
body_joints = self.joints_out(decoded)  # (B, 22, 3)
```

Zero-pad hand joints (indices 22–69) after body joints:
```python
pad = torch.zeros(B, self.num_joints - self.num_body_queries, 3,
                  device=body_joints.device, dtype=body_joints.dtype)
joints = torch.cat([body_joints, pad], dim=1)  # (B, 70, 3)
```

The `pad` tensor is constructed with `torch.zeros`, which has `requires_grad=False` by default. No special action required.

Pelvis output — conditional on `use_decoupled_pelvis`:
```python
if self.use_decoupled_pelvis:
    pelvis_qs = self.pelvis_coord_queries.weight.unsqueeze(0).expand(B, -1, -1)
    # pelvis_qs: (B, 2, pelvis_hidden_dim)
    pelvis_decoded = self.pelvis_decoder(pelvis_qs, spatial)  # (B, 2, pelvis_hidden_dim)
    pelvis_depth = self.depth_out(pelvis_decoded[:, 0, :])    # (B, 1)
    pelvis_uv    = self.uv_out(pelvis_decoded[:, 1, :])       # (B, 2)
else:
    pelvis_token = decoded[:, 0, :]
    pelvis_depth = self.depth_out(pelvis_token)
    pelvis_uv = self.uv_out(pelvis_token)
```

For this design, `use_decoupled_pelvis=True` and `num_body_queries=22`, so both the 22-query path and the dedicated pelvis path are active simultaneously.

The returned dict is identical: `{'joints': joints, 'pelvis_depth': pelvis_depth, 'pelvis_uv': pelvis_uv}`.

**`loss()` method:** No changes. Body joint loss uses `_BODY = list(range(0, 22))`, which covers all 22 output joints (the zero-padded hand region at 22–69 is never referenced). Pelvis losses read from `pred['pelvis_depth']` and `pred['pelvis_uv']`.

**`predict()` method:** No changes. `self.num_joints = 70` remains correct.

### 2. `config.py`

In `model.head`, add all four new kwargs with `use_decoupled_pelvis=True` and `num_body_queries=22`:

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
    pelvis_num_heads=8,
    num_body_queries=22,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size 4, accum 8, seed 2026) are **identical to baseline**.

### 3. `pelvis_utils.py`

No changes.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. `self.num_joints = 70` must remain set in `__init__` so `predict()` produces `keypoint_scores` of shape `(1, 70)`.
3. `self.num_body_queries = 22` is stored separately from `self.num_joints = 70`. Both must coexist as instance attributes.
4. `_PelvisCrossAttnDecoder` must be defined at module level, before `Pose3dTransformerHead`.
5. Zero-padding must be done with `torch.zeros(B, self.num_joints - self.num_body_queries, 3, device=..., dtype=...)`. With `num_joints=70` and `num_body_queries=22`, this pads 48 joints. `torch.zeros` returns a tensor with `requires_grad=False` — no special detach needed.
6. The zero-padded `joints` tensor has shape `(B, 70, 3)` before returning from `forward()`.
7. `_BODY = list(range(0, 22))` in `loss()` — unchanged. This covers exactly the 22 active body joint outputs.
8. The joint query embedding `self.joint_queries = nn.Embedding(22, hidden_dim)` — only 22 embeddings, not 70.
9. `pelvis_coord_queries` embedding: size `(2, 256)`. Index 0 = depth query, index 1 = UV query.
10. `pelvis_hidden_dim=256` must equal `hidden_dim=256` for `depth_out` and `uv_out` input compatibility.
11. `depth_out` and `uv_out` are re-initialized by the existing `_init_head_weights()` loop — no additional action needed.
12. `use_decoupled_pelvis=False` default preserved for backward compatibility.
13. The pelvis token `decoded[:, 0, :]` is NOT read for pelvis output in this design (because `use_decoupled_pelvis=True`). The joint decoder's token 0 is a pure body joint query.
14. MMEngine config: all four new kwargs are bool/int literals. No import statements. Fully compliant.
15. Seed `2026`, batch size `4`, accumulation `8` — do not change.
16. `max_keep_ckpts=1`, `resume=True` — do not change.

---

## Expected Behavior After Change

- Joint decoder: self-attention is 22×22 (body only, no hand contamination). Cross-attention: 22 queries × 960 spatial tokens. All 22 queries receive gradients exclusively from body joint regression. Joint token 0 is a pure body joint query.
- Pelvis decoder: 2 queries, single cross-attention pass over 960 spatial tokens, 8 heads. Gradients from depth and UV losses flow exclusively into `pelvis_coord_queries` and `pelvis_decoder`, with no interaction with the joint self-attention graph.
- Output: `joints (B, 70, 3)` — body joints [0:22] predicted, hand joints [22:70] zero; `pelvis_depth (B, 1)`; `pelvis_uv (B, 2)`.
- This design removes both hand-query contamination (idea008 contribution) and pelvis-objective contamination (this idea) from the joint self-attention. Expected to be the strongest variant in this idea.
- Target stage-1: `composite_val < 325` (improving on best prior stage-1 of 323.75 from idea023/design001), `mpjpe_pelvis_val < 580 mm`, `mpjpe_body_val < 185 mm`.
- Target stage-2: `composite_val < 220` (competitive with best stage-2 of 224.52 from idea001/design001).
