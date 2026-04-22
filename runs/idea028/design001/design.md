# Design 001 — Decoupled pelvis decoder, 8-head, full 70-query joint decoder (diagnostic)

**Design Description:** Add dedicated `_PelvisCrossAttnDecoder` (8-head cross-attn, no self-attn, no FFN) with two learned pelvis-coordinate queries; `depth_out` and `uv_out` read from those dedicated outputs rather than joint token 0.

**Starting Point:** `baseline/`

---

## Overview

This is the minimal diagnostic variant. The joint decoder is unchanged (70 queries, self-attn + cross-attn + FFN). The only algorithm change is that `depth_out` and `uv_out` no longer read from `decoded[:, 0, :]`. Instead, two learnable "pelvis coordinate" queries run an independent cross-attention pass over the spatial tokens and produce `(B, 2, hidden_dim)`, from which depth and UV projections read.

Joint token 0 becomes a pure body joint query — it participates in joint self-attention and contributes to body joint regression but no longer carries a conflicting absolute-localization objective.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Add new module `_PelvisCrossAttnDecoder` at module level (after `_DecoderLayer`):**

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
            queries: (B, 2, embed_dim)  — index 0 = depth query, index 1 = UV query
            spatial_tokens: (B, H'*W', embed_dim)
        Returns:
            (B, 2, embed_dim)
        """
        q = self.norm(queries)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
        return queries + q2
```

**`Pose3dTransformerHead.__init__` changes:**

Add four new kwargs after `loss_weight_uv`, before `init_cfg`:

```python
use_decoupled_pelvis: bool = False,
pelvis_hidden_dim: int = 256,
pelvis_num_heads: int = 8,
num_body_queries: int = 70,
```

Store them as instance attributes:
```python
self.use_decoupled_pelvis = use_decoupled_pelvis
self.pelvis_hidden_dim = pelvis_hidden_dim
self.num_body_queries = num_body_queries
```

After the existing `self.uv_out = nn.Linear(hidden_dim, 2)` line, add a conditional block:
```python
if use_decoupled_pelvis:
    self.pelvis_coord_queries = nn.Embedding(2, pelvis_hidden_dim)
    self.pelvis_decoder = _PelvisCrossAttnDecoder(
        pelvis_hidden_dim, pelvis_num_heads, dropout)
```

The existing `self.depth_out` and `self.uv_out` Linear layers are kept as-is in construction. They will now receive input from the dedicated pelvis decoder output rather than joint token 0 — no weight reuse from a different source; `_init_head_weights()` re-initializes them identically.

**`_init_head_weights()` changes:**

After the existing body (initializing `self.joint_queries.weight`, `self.joints_out`, `self.depth_out`, `self.uv_out`), add:

```python
if self.use_decoupled_pelvis:
    nn.init.trunc_normal_(self.pelvis_coord_queries.weight, std=0.02)
```

`self.depth_out` and `self.uv_out` are still initialized by the existing loop — no change needed. The explicit re-initialization ensures clean starts regardless of `use_decoupled_pelvis`.

**`forward()` method changes:**

Keep the existing joint decoder path completely unchanged:
```python
# ... spatial tokens built as before ...
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 70, hidden_dim)
decoded = self.decoder_layer(queries, spatial)  # (B, 70, hidden_dim)
joints = self.joints_out(decoded)  # (B, 70, 3)
```

Replace the pelvis output block:

**Before (baseline):**
```python
pelvis_token = decoded[:, 0, :]
pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

**After (conditional):**
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

The returned dict keys are identical: `{'joints': joints, 'pelvis_depth': pelvis_depth, 'pelvis_uv': pelvis_uv}`.

**`loss()` method:** No changes. The loss reads `pred['pelvis_depth']` and `pred['pelvis_uv']` by key — these are now produced by the dedicated pelvis decoder, but the loss code is transparent to this. The body joint loss still uses `pred['joints'][:, _BODY]` (indices 0–21), unchanged.

**`predict()` method:** No changes.

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

### 2. `config.py`

In `model.head`, add the four new kwargs as bool/int literals. The `use_decoupled_pelvis=True` is the key change:

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
    num_body_queries=70,
),
```

All other config values (optimizer `lr=1e-4`, `weight_decay=0.03`, backbone `lr_mult=0.1`, `clip_grad max_norm=1.0`, LR schedule, data pipeline, hooks, batch size 4, accum 8, seed 2026, SLURM limits) are **identical to baseline**.

### 3. `pelvis_utils.py`

No changes.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. `self.num_joints = 70` must remain set so `predict()` produces correct-shape `keypoint_scores`.
3. `_PelvisCrossAttnDecoder` must be defined at module level, before `Pose3dTransformerHead`, so it is accessible in `__init__`.
4. The `pelvis_coord_queries` embedding is size `(2, pelvis_hidden_dim)`: index 0 = depth query, index 1 = UV query. This ordering must be consistent with the `forward()` indexing.
5. `pelvis_hidden_dim` must equal `hidden_dim` (256) for this design — the `depth_out` and `uv_out` projections take `hidden_dim`-dimensional input. Mismatch would cause a shape error.
6. `depth_out` and `uv_out` are re-initialized by `_init_head_weights()` unconditionally — the existing loop `for m in [self.joints_out, self.depth_out, self.uv_out]` already covers them. No additional initialization code is needed.
7. `use_decoupled_pelvis=True/False` must default to `False` to preserve backward compatibility with all prior designs.
8. `num_body_queries=70` for this design — the joint query embedding is `nn.Embedding(70, hidden_dim)`, unchanged from baseline.
9. The joint decoder (`decoder_layer`) is completely unchanged: self-attn → cross-attn → FFN over 70 queries.
10. Body joint loss indices `_BODY = list(range(0, 22))` — unchanged.
11. MMEngine config: all four new kwargs are bool/int literals — no import statements required. Fully compliant.
12. AMP (`FixedAmpOptimWrapper`) is unchanged.
13. Seed `2026`, batch size `4`, accumulation `8` — do not change.
14. `max_keep_ckpts=1`, `resume=True` checkpoint settings — do not change.

---

## Expected Behavior After Change

- Joint decoder runs on 70 queries with full self-attn + cross-attn + FFN, unchanged from baseline. Gradients for joint token 0 come only from `loss/joints/train` (body joint regression). No pelvis depth/UV gradient flows into the joint decoder.
- Dedicated pelvis decoder: 2 queries, single cross-attention pass over 960 spatial tokens, no self-attention, no FFN. Gradients for `pelvis_coord_queries` and `pelvis_decoder` come only from `loss/depth/train` and `loss/uv/train`.
- `pelvis_decoded[:, 0, :]` feeds `depth_out`; `pelvis_decoded[:, 1, :]` feeds `uv_out`.
- Output shapes: `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)` — identical to baseline.
- Target stage-1: `mpjpe_pelvis_val < 580 mm` (vs. baseline 652 mm), `composite_val < 340`.
- Net new parameters: `pelvis_coord_queries` (2×256 = 512) + `_PelvisCrossAttnDecoder` (~263K). Total overhead vs. baseline: ~263K parameters.
