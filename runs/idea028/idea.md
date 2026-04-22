**Idea Name:** Decoupled Pelvis Coordinate Queries with Axis-Specific Cross-Attention

**Approach:** Replace the baseline practice of reading pelvis depth and UV from body joint query token 0 with two dedicated "pelvis coordinate" queries — one for depth, one for UV — that run a separate, lightweight cross-attention pass over the spatial tokens without ever participating in joint-query self-attention, freeing body joint token 0 from the conflicting objective of encoding both relative body structure and absolute pelvis localization.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Dual-Objective Bottleneck at Token 0

The baseline head regresses all three prediction targets from the decoder output:
```
joints        = joints_out(decoded)          # (B, 70, 3) — token 0 is pelvis
pelvis_depth  = depth_out(decoded[:, 0, :]) # reads from body joint query token 0
pelvis_uv     = uv_out(decoded[:, 0, :])    # reads from body joint query token 0
```

Token 0 participates fully in self-attention with all other 69 joint queries and in cross-attention with the 960 spatial tokens. Through self-attention, it learns to encode the relative pose of all body joints (because it is part of the skeleton topology). Through its output projection, it must simultaneously represent absolute pelvis position in 3D (for depth/UV prediction).

These two objectives are **structurally conflicting**:

1. **Relative pose encoding** (self-attention objective): token 0 accumulates information from all 21 other body queries. The self-attention gradients push the token toward a rich contextual representation of the full skeleton — the pelvis-relative position of every joint.

2. **Absolute pelvis localization** (depth/UV output objective): token 0 must also retain an image-level spatial cue: where in the image the pelvis appears (for UV) and how far away it is (for depth). This requires cross-attention to fire on specific global regions of the spatial feature map, which conflicts with the broad contextual aggregation demanded by self-attention.

This tension is not hypothetical — it is directly observed in the results. Across all 27 prior ideas:
- `mpjpe_body_val` has improved from 195.7 mm (baseline) to as low as 156.6 mm (idea002/design003) — a 39 mm improvement.
- `mpjpe_pelvis_val` has moved from 652.9 mm (baseline) to a best of 322.0 mm (idea003/design002 stage-2), with most designs clustering in the 600–740 mm range at stage-1.
- Nearly every architectural change that improves body MPJPE degrades or has no effect on pelvis MPJPE (e.g., idea001, idea006, idea017, idea022).

The pattern is consistent: improvements to body joint representation destabilize the pelvis token because they push token 0 to encode more body context at the expense of absolute localization.

### Why Prior Approaches Have Not Solved This

| Prior Idea | Mechanism | Limitation |
|---|---|---|
| idea002 | Dedicated pelvis query in the decoder | Pelvis query still runs self-attention with body queries; contamination from body self-attention remains |
| idea004 | Depth-aware positional encoding on spatial tokens | Input-side encoding; does not decouple the pelvis head output from the body decoder |
| idea010 | 2D reprojection consistency loss | Loss-level coupling; token 0 still carries the conflicting dual objective |
| idea014 | Anchor-based depth classification head | Output representation change only; still reads from token 0 which is joint-self-attention-contaminated |
| idea016 | Depth-conditional feature modulation | Scales spatial features by depth; token 0 still participates in full joint self-attention |

No prior idea has addressed the structural root cause: **the pelvis depth and UV heads share an input token that is entangled with the full joint self-attention graph**.

### Proposed Architecture

The key change is to create **two dedicated pelvis-coordinate queries** that are completely independent from the joint decoder:

```
┌─────────────────────────────────────────────────────────────┐
│  Spatial tokens (B, 960, 256)  +  sincos PE                 │
│              ↓                                              │
│  ┌───────────────────────────┐  ┌───────────────────────┐  │
│  │ Joint Decoder (22 or 70)  │  │ Pelvis Coord Decoder  │  │
│  │ self-attn + cross-attn    │  │ cross-attn only       │  │
│  │ → joints (B, 70, 3)       │  │ pelvis_depth_q (B,256)│  │
│  │                           │  │ pelvis_uv_q    (B,256)│  │
│  └───────────────────────────┘  └───────────────────────┘  │
│                                     ↓              ↓        │
│                              depth_out(·)    uv_out(·)      │
│                              → (B,1)         → (B,2)        │
└─────────────────────────────────────────────────────────────┘
```

The joint decoder runs exactly as in the baseline (self-attention + cross-attention). Joint token 0 is now a pure body joint query — it is no longer read by any pelvis output head.

The pelvis coordinate decoder is a **dedicated single cross-attention layer** (no self-attention, no FFN) applied to two learned query vectors:
- `pelvis_depth_query ∈ R^{hidden_dim}`: attends to spatial tokens, then passed to `depth_out`.
- `pelvis_uv_query ∈ R^{hidden_dim}`: attends to spatial tokens, then passed to `uv_out`.

The pelvis cross-attention layer has no self-attention and no FFN — it is strictly a single `nn.MultiheadAttention` with queries `(B, 2, hidden_dim)` and keys/values from the spatial tokens. It is initialized independently and can develop its own spatial routing pattern without being disrupted by joint-query gradients.

### Why This Architecture Is Sound

1. **Gradient decoupling**: The pelvis depth/UV loss gradient no longer flows through the joint self-attention. Token 0 in the joint decoder now receives only body joint supervision, freeing it to encode body structure without sacrificing localization accuracy.

2. **Spatial routing specialization**: The depth cross-attention query can specialize in attending to vertical structures and global depth cues. The UV query can attend to the specific spatial location of the pelvis in the image. With only 2 queries and no self-attention interaction, the cross-attention has low noise and high specificity.

3. **Zero-cost initialization**: The dedicated pelvis queries are initialized to trunc_normal (same as baseline query embeddings). The cross-attention layer is initialized identically to the baseline's cross-attention. The joint token 0 output head weights are re-initialized — `depth_out` and `uv_out` now read from the new dedicated queries. At epoch 0, the network produces outputs of the same statistical scale as baseline. No special warm-start is needed.

4. **Composability**: This architecture is orthogonal to all prior ideas. It can compose with idea008 (body-only 22-query joint decoder), idea023 (heatmap-guided joint query initialization), idea010 (2D reprojection loss — still differentiable through the new pelvis queries), and idea026 (Laplace NLL for joint loss).

5. **Parameter cost**: Two new learned query vectors = `2 × 256 = 512` parameters. The dedicated cross-attention layer = standard `nn.MultiheadAttention(256, 8)` ≈ 263K parameters (same as one of the baseline decoder's cross-attention modules). Net additional parameters: ~263K vs baseline, well within 2080 Ti budget.

### Grounding in Observed Results

- **idea002/design003 stage-2** achieved the best-ever body MPJPE: 156.6 mm (vs baseline 183.7 mm). The dedicated pelvis query in idea002 improved body MPJPE (because it slightly reduced pelvis token 0's self-attention load) but pelvis MPJPE still tracked the body pattern — it didn't break free of the contamination. This idea takes the decoupling further.
- **idea008/design002** dramatically improved `mpjpe_rel_val` (362 mm vs baseline 438 mm) by removing hand query contamination from self-attention. The analogy is exact: just as removing hand queries from the joint decoder improved body predictions, removing pelvis coordinate queries from the joint decoder frees the body queries from competing with pelvis-localization objectives.
- **pelvis MPJPE plateau**: no stage-1 design has achieved pelvis MPJPE below 608 mm (idea023/design001), and the baseline is 652 mm. The improvement so far has come from better body joint regression, not from improved pelvis localization. Decoupling the pelvis head addresses the root cause.

---

## Proposed Variations

### Design A — Dedicated pelvis cross-attention, depth and UV sharing one cross-attention layer

Run joint decoder on all 70 queries (baseline). Add a dedicated `_PelvisCrossAttnDecoder` module that takes `(B, 2, hidden_dim)` queries (one for depth, one for UV) and runs a single cross-attention pass over the spatial tokens. The `depth_out` and `uv_out` projections read from the two dedicated pelvis decoder outputs, not from joint token 0.

Joint decoder: unchanged (70 queries, 1 decoder layer, self-attn + cross-attn + FFN).
Pelvis decoder: `nn.MultiheadAttention(256, 8)` with queries `(B, 2, 256)`, keys/values from spatial tokens. Output: `(B, 2, 256)`. Depth from output[:, 0, :], UV from output[:, 1, :].

Config kwargs: `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=8`.

This is the minimal diagnostic: does decoupling the pelvis head from joint self-attention improve pelvis MPJPE?

### Design B — Decoupled pelvis decoder with 4-head lightweight cross-attention

Same as Design A but the dedicated pelvis decoder uses 4 heads (vs. 8 in the joint decoder). Motivation: the pelvis localization task (2 outputs: depth + UV) is much simpler than joint regression (22 body joints). A lighter-weight attention head may overfit less on the clean BEDLAM2 data and produce smoother convergence.

Config kwargs: `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=4`.

### Design C — Decoupled pelvis decoder + body-only joint decoder (22 queries)

Combines the decoupled pelvis head from Design A/B with the body-only 22-query joint decoder from idea008. Specifically:
- Joint decoder: 22 body queries only (no hand queries, no pelvis pseudo-query). Self-attention is 22×22.
- Pelvis decoder: 2 dedicated queries as in Design A.
- Output: hand joints zero-padded (same as idea008/design001), body joints from 22-query decoder, depth/UV from dedicated pelvis decoder.

This design tests the hypothesis that removing *both* hand contamination (idea008) and pelvis-objective contamination (this idea) from the joint self-attention will yield additive improvements.

Config kwargs: `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=8`, `num_body_queries=22`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. `pelvis_utils.py` is unchanged.

### `pose3d_transformer_head.py`

**1. New module `_PelvisCrossAttnDecoder`**

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
            queries: (B, 2, embed_dim)  — depth query + UV query
            spatial_tokens: (B, H'W', embed_dim)
        Returns:
            (B, 2, embed_dim)
        """
        q = self.norm(queries)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
        return queries + q2
```

**2. `Pose3dTransformerHead.__init__` additions**

New kwargs (all defaulting to baseline behavior):
```python
use_decoupled_pelvis: bool = False   # enable dedicated pelvis coord queries
pelvis_hidden_dim: int = 256         # embed dim for pelvis decoder
pelvis_num_heads: int = 8            # attn heads for pelvis decoder
num_body_queries: int = 70           # 70 = baseline, 22 = body-only (Design C)
```

When `use_decoupled_pelvis=True`:
```python
# Two dedicated pelvis coordinate queries: index 0 = depth, index 1 = UV
self.pelvis_coord_queries = nn.Embedding(2, pelvis_hidden_dim)
nn.init.trunc_normal_(self.pelvis_coord_queries.weight, std=0.02)

# Dedicated cross-attention layer for pelvis queries
self.pelvis_decoder = _PelvisCrossAttnDecoder(
    pelvis_hidden_dim, pelvis_num_heads, dropout)

# depth_out and uv_out still exist; they now read from pelvis_decoder output
# No change to weights needed — re-initialized in _init_head_weights()
```

For Design C (`num_body_queries=22`), the joint query embedding is resized:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

**3. `forward()` additions**

```python
if self.use_decoupled_pelvis:
    # Run pelvis decoder: cross-attn only, no self-attn
    pelvis_qs = self.pelvis_coord_queries.weight.unsqueeze(0).expand(B, -1, -1)
    pelvis_decoded = self.pelvis_decoder(pelvis_qs, spatial)  # (B, 2, hidden_dim)
    pelvis_depth = self.depth_out(pelvis_decoded[:, 0, :])    # (B, 1)
    pelvis_uv    = self.uv_out(pelvis_decoded[:, 1, :])       # (B, 2)
else:
    # Baseline: read from joint token 0
    pelvis_token = decoded[:, 0, :]
    pelvis_depth = self.depth_out(pelvis_token)
    pelvis_uv    = self.uv_out(pelvis_token)
```

For Design C, after the joint decoder produces `(B, 22, 3)` body joints, zero-pad hands:
```python
if self.num_body_queries == 22:
    pad = torch.zeros(B, 48, 3, device=joints.device, dtype=joints.dtype)
    joints = torch.cat([joints, pad], dim=1)  # (B, 70, 3)
```

**4. `loss()` — no changes required**

The loss function reads from `pred['pelvis_depth']` and `pred['pelvis_uv']`, which are now produced by the dedicated pelvis decoder. The body joint loss still uses `pred['joints'][:, _BODY]` (indices 0–21), unchanged. No new loss terms are introduced. All metric logging stays unchanged.

### `config.py`

**Design A:**
```python
use_decoupled_pelvis=True,
pelvis_hidden_dim=256,
pelvis_num_heads=8,
num_body_queries=70,
```

**Design B:**
```python
use_decoupled_pelvis=True,
pelvis_hidden_dim=256,
pelvis_num_heads=4,
num_body_queries=70,
```

**Design C:**
```python
use_decoupled_pelvis=True,
pelvis_hidden_dim=256,
pelvis_num_heads=8,
num_body_queries=22,
```

All values are bool/int literals. No Python import statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — pelvis MPJPE**: removing the dual-objective burden from token 0 should allow the dedicated pelvis queries to fully specialize on absolute localization. Target: `mpjpe_pelvis_val < 580 mm` at stage-1 (vs. baseline 652 mm, best prior 608 mm from idea023). At stage-2, target `< 310 mm` (vs. baseline 365 mm).

- **Secondary gain — body MPJPE**: joint token 0 is now a pure body joint query. Without the conflicting pelvis output objective, all 22 body queries (or 70 in Designs A/B) can focus on relative pose structure. Expected: `mpjpe_body_val < 187 mm` at stage-1.

- **Design A** (8-head dedicated pelvis decoder, full 70-query joint decoder): clean diagnostic. Expected composite_val improvement primarily driven by pelvis and pelvis_abs metrics. Target composite_val < 335.

- **Design B** (4-head lighter pelvis decoder): tests whether a simpler pelvis head (fewer attention heads) converges better for a simple 2-output regression task. Expected performance similar to or better than Design A.

- **Design C** (decoupled pelvis + body-only 22-query decoder): combines two independently validated improvements — pelvis decoupling from this idea and hand-query contamination removal from idea008/design002. The joint decoder now has a fully clean self-attention space: 22 body queries with no hand contamination and no pelvis-objective contamination. Expected to be the strongest design. Target composite_val < 325 at stage-1.

- **Composite target (stage-1)**: Design C: `composite_val < 325`, improving on best prior stage-1 of 323.75 (idea023/design001). Design A/B: `< 340`.
- **Composite target (stage-2)**: `composite_val < 220`, competitive with best stage-2 of 224.52 (idea001/design001).

---

## Risk and Mitigation

- **Token 0 identity change**: in the baseline, `depth_out` and `uv_out` are trained for 20 epochs reading from joint token 0. When using the decoupled head, these projections now read from the pelvis decoder output. Since we re-initialize `depth_out` and `uv_out` fresh, there is no stale weight issue. The Designer should ensure `_init_head_weights` explicitly re-initializes `depth_out` and `uv_out` even when `use_decoupled_pelvis=True`.

- **Joint token 0 now unused for depth/UV**: the baseline's `self._train_mpjpe_abs` computation uses `pred['pelvis_depth']` and `pred['pelvis_uv']`, which come from the new dedicated queries. This is transparent — the downstream metric code is unchanged.

- **Design C hand padding**: zero-padded hand joints (indices 22–69) must use `requires_grad=False` zeros to avoid spurious gradient through the hand auxiliary loss path. Use `torch.zeros(..., device=joints.device, dtype=joints.dtype)` (no autograd by default for `torch.zeros`). Metric code ignores hand joints for composite_val.

- **Memory cost**: the dedicated pelvis cross-attention adds `(B=4, 2, 960)` attention matrix ≈ 7.7K float16 values = 15 KB. The `_PelvisCrossAttnDecoder` has the same parameter count as one multihead attention module ≈ 263K params. Total overhead: negligible.

- **Speed**: the pelvis cross-attention with queries `(B=4, 2, 256)` and keys `(B=4, 960, 256)` is trivially fast — the matrix multiply is `4 × 2 × 256 × 960` ≈ 2M multiply-adds vs the main decoder's `4 × 70 × 256 × 960` ≈ 69M multiply-adds. Net overhead: < 3%.

- **Interaction with idea010 (2D reprojection)**: the 2D reprojection loss assembles `pred_abs = pred_rel + pred_pelvis_3d` and projects through K. `pred_pelvis_3d` is computed from `pred['pelvis_depth']` and `pred['pelvis_uv']` in `pelvis_utils.py`. Since these now come from the dedicated pelvis decoder, the reprojection loss gradient flows into the dedicated pelvis queries — still valid and differentiable.

- **MMEngine config constraint**: all new kwargs are bool/int literals in config.py. No import statements. Fully compliant.
