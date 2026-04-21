**Idea Name:** Body-Focused Multi-Layer Decoder with Linear Hand Recovery

**Approach:** Combine a 22-query body-only transformer decoder (from idea008) with a 2-layer stacked decoder with intermediate supervision (from idea001), so that multi-layer refinement operates exclusively over body queries — eliminating hand-query contamination from self-attention while enabling deeper progressive refinement within the GPU memory budget freed by the reduced query set.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

Two of the best-performing ideas to date operate on orthogonal axes of improvement:

- **idea001/design001** (multi-layer decoder, 2 layers + no aux loss) → stage-2 composite_val **224.52** — the best overall result across all trained designs, primarily through body MPJPE improvement (176.11mm).
- **idea008/design002** (body-focused 22-query decoder + linear hand recovery) → stage-2 composite_val **241.14**, with exceptional relative MPJPE reduction (333.2mm vs. baseline 438.7mm) and absolute MPJPE improvement (533.8mm vs. baseline 833.7mm).

Neither idea has been applied together. idea008's own text explicitly identifies this combination as the "most powerful" next step: *"a 2-layer decoder over 22 body queries would be even more powerful than either alone."* This idea realizes that proposal.

### Why the combination is greater than the sum of its parts

#### 1. Memory synergy enables more decoder depth

The baseline decoder's cross-attention involves Q=(B,70,256) × K/V=(B,960,256). With body-only 22 queries (idea008), Q shrinks to (B,22,256) — a 69% reduction in cross-attention rows. Self-attention shrinks from 70×70=4,900 to 22×22=484 elements — a 90% reduction. These savings together allow stacking 2–3 decoder layers within the same 2080 Ti VRAM budget that the baseline uses for a single layer over 70 queries.

Rough VRAM estimate (AMP, batch=4):
- Baseline: 1 layer × (70×960 cross-attn + 70×70 self-attn) ≈ full budget
- Idea017: 2 layers × (22×960 cross-attn + 22×22 self-attn) ≈ 2 × 31% ≈ 62% of baseline cross/self-attn VRAM → fits comfortably

#### 2. Self-attention contamination elimination amplifies multi-layer benefit

idea001's multi-layer decoder applied to all 70 queries means that in each additional layer, body query self-attention still contaminates with all 48 hand queries. The progressive refinement in idea001 must "fight through" hand-query noise at every layer. With 22-query self-attention, every refinement step involves only body queries attending to each other — the multi-layer refinement signal is clean from the start.

This explains the residual gap between idea001 (composite 224.52) and a hypothetical combined approach: idea001 used 70-query self-attention in all decoder layers, likely degrading the body refinement by hand contamination in deeper layers. Running the same multi-layer refinement over 22 body-only queries removes this noise source at every layer.

#### 3. Pelvis token benefits most from cleaner multi-layer self-attention

Idea001's best design (design001, 2-layer no-aux) improved body MPJPE to 176.11mm but the pelvis only marginally improved (322.82mm vs. baseline 365.54mm at stage-2). The pelvis token (index 0) is one of 70 queries in idea001's self-attention — it must attend through the noise of 48 hand queries in every layer. In the 22-query formulation, token 0 self-attends only to 21 body joint queries; its multi-layer refinement is undiluted by hand query noise.

#### 4. Linear hand recovery makes this fully output-compatible

Rather than zero-padding hands (Design A of idea008), a small linear projection recovers hand joint predictions from the decoded body features. This keeps the output shape at (B,70,3) with no downstream changes, while adding only 22×256×(48×3) = 810K parameters — a negligible cost. The auxiliary hand loss (weight 0.1) prevents the projection from degenerating and provides a light structural anchor for the body decoder.

### Differentiation from prior ideas

| Idea | Mechanism | Gap |
|---|---|---|
| idea001 | Multi-layer decoder, 70 queries | Hand contamination in self-attention every layer; best stage-2 = 224.52 |
| idea008 | 22-query body decoder, single layer | No multi-layer refinement; best stage-2 = 241.14 |
| idea015 | Super-token pooling + stacked layers | Changes token side, not query side; still 70 queries; not yet trained |
| idea016 | FiLM conditioning | Feature modulation; still 70 queries, single layer; not yet trained |

This idea is the only one that combines **query-side reduction** (22 body queries) with **decoder depth increase** (2 layers + optional aux loss). It does not duplicate any prior idea — it is a motivated composition of idea001 and idea008, both empirically validated to help.

### Grounding in observed patterns

- Body MPJPE floor across all ideas: ~155–184mm at stage-2. The two structural changes that most pushed body MPJPE down were: idea001 (multi-layer) to 176.11mm, and idea002/design003 to 156.59mm. Neither used 22-body-only queries with multi-layer decoding.
- Pelvis MPJPE at stage-2 remains 322–378mm across all ideas. The best single pelvis result (idea001/design001: 322.82mm) was paired with the best body MPJPE of idea001. Running those same 2 decoder layers over 22 body queries should replicate or improve pelvis improvement.
- idea008/design002's outstanding relative MPJPE improvement (333.2mm vs. baseline 438.7mm) suggests that removing hand queries from cross-attention benefits joint-relative structure. Stacking layers on top of this clean cross-attention should amplify the effect.

---

## Proposed Variations

### Design A — 22-query body decoder, 2 layers, no intermediate supervision (minimal combination)

Stack 2 decoder layers operating on 22 body queries (same as idea001/design001 but query-reduced). No auxiliary loss. Linear hand recovery with weight 0.1.

This is the most controlled test: idea001/design001 showed that 2 layers without aux loss already beat the baseline significantly. Running this on 22 body queries directly tests whether query reduction + deeper decoding compounds the benefit.

Architecture:
```
joint_queries: nn.Embedding(22, hidden_dim)
decoder_layers: nn.ModuleList([_DecoderLayer(256, 8, 0.1), _DecoderLayer(256, 8, 0.1)])
hand_proj: nn.Linear(22 * hidden_dim, 48 * 3)  # linear hand recovery
```

Forward:
```python
queries = joint_queries.weight.expand(B, -1, -1)  # (B, 22, hidden_dim)
for layer in decoder_layers:
    queries = layer(queries, spatial)
body_joints = joints_out(queries)  # (B, 22, 3)
hand_joints = hand_proj(queries.flatten(1)).reshape(B, 48, 3)
joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)
```

Loss: joint body loss on indices 0–21 (unchanged); auxiliary hand loss weight 0.1.
Config kwargs: `num_body_queries=22`, `num_decoder_layers=2`, `hand_proj_type='linear'`, `hand_aux_loss_weight=0.1`.

### Design B — 22-query body decoder, 2 layers, intermediate supervision on body joints

Same as Design A but add auxiliary joint loss at the output of decoder layer 1 (intermediate), with weight 0.4. This is exactly the idea001/design002 treatment applied to the 22-query setting:

```python
# In loss():
inter_body_joints = joints_out(intermediate_decoded)   # from layer-1 output
losses['loss/joints_aux/train'] = 0.4 * loss_joints_module(
    inter_body_joints[:, _BODY], gt_joints[:, _BODY])
```

Rationale: intermediate supervision at layer 1 forces the first decoder layer to produce a usable pose estimate, preventing gradient vanishing and making the second layer refine from a meaningful starting point. idea001/design002 showed this combination effective at stage-1 (composite 338.78) and pushed design001 to 224.52 at stage-2 — intermediate supervision may further stabilize convergence on the already-cleaner 22-query decoding.

Config kwargs: `num_body_queries=22`, `num_decoder_layers=2`, `hand_proj_type='linear'`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`.

### Design C — 22-query body decoder, 3 layers, intermediate supervision (max-depth variant)

Push to 3 decoder layers (enabled by the 90% self-attention reduction in the 22-query setting). Intermediate body joint losses at layers 1 and 2 (weights 0.4 and 0.6 respectively), full body loss at layer 3 (weight 1.0). Hand linear recovery at final layer.

Memory feasibility:
- 3-layer decoder × (22×960 cross-attn + 22×22 self-attn) ≈ 3 × 31% = 93% of baseline single-layer VRAM for cross/self-attn
- Plus hand projection: negligible
- Net: similar total VRAM to baseline despite 3× decoder depth

This design tests whether the memory headroom from query reduction can be fully spent on decoder depth. idea001's Design C (4 layers on 70 queries) was not trained (only designs 001–003 ran; design003 was 3-layer + aux which reached 408.47 at stage-1, worse than design001). With 22 queries, 3 layers is feasible without the memory risk that 4 layers on 70 queries would carry.

Config kwargs: `num_body_queries=22`, `num_decoder_layers=3`, `hand_proj_type='linear'`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

**`__init__` changes:**

```python
# New constructor kwargs:
#   num_body_queries: int = 70          (22 for this idea)
#   num_decoder_layers: int = 1         (2 or 3 for this idea)
#   hand_aux_loss_weight: float = 0.0   (0.1 for all designs)
#   aux_body_loss_weight: float = 0.0   (0.4 for Designs B/C)

# Query embedding: only body queries
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)

# Decoder stack
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
# Keep self.decoder_layer for single-layer compat (Designer may unify)

# Hand projection (linear): maps (B, num_body_queries * hidden_dim) → (B, 48 * 3)
num_hand_joints = num_joints - num_body_queries  # e.g. 70 - 22 = 48
self.hand_proj = nn.Linear(num_body_queries * hidden_dim, num_hand_joints * 3)
nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
nn.init.zeros_(self.hand_proj.bias)
```

**`forward()` changes:**

```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 22, hidden_dim)

intermediate_outputs = []
for layer in self.decoder_layers:
    queries = layer(queries, spatial)
    intermediate_outputs.append(queries)

body_joints = self.joints_out(queries)                        # (B, 22, 3)
hand_joints = self.hand_proj(
    queries.flatten(1)).reshape(B, self.num_joints - self.num_body_queries, 3)  # (B, 48, 3)
joints = torch.cat([body_joints, hand_joints], dim=1)        # (B, 70, 3)

pelvis_token = queries[:, 0, :]                              # (B, hidden_dim)
pelvis_depth = self.depth_out(pelvis_token)                  # (B, 1)
pelvis_uv = self.uv_out(pelvis_token)                        # (B, 2)
```

**`loss()` changes:**

```python
# Auxiliary intermediate body joint loss (Designs B/C)
if self.aux_body_loss_weight > 0.0:
    for i, inter_decoded in enumerate(intermediate_outputs[:-1]):
        inter_body = self.joints_out(inter_decoded)
        losses[f'loss/joints_aux_{i}/train'] = (
            self.aux_body_loss_weight * self.loss_joints_module(
                inter_body[:, _BODY], gt_joints[:, _BODY]))

# Auxiliary hand loss (all designs)
if self.hand_aux_loss_weight > 0.0:
    losses['loss/hand_aux/train'] = (
        self.hand_aux_loss_weight * self.loss_joints_module(
            pred['joints'][:, 22:], gt_joints[:, 22:]))
```

**`_init_head_weights()`**: add `nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)` and `nn.init.zeros_(self.hand_proj.bias)`.

**`predict()`**: unchanged — reads `pred['joints']` which already has shape (B, 70, 3).

### `config.py`

Add to head kwargs (all literals, no imports):
```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=1024,
    hidden_dim=256,
    num_joints=70,
    num_body_queries=22,          # Design A/B/C: 22
    num_decoder_layers=2,          # Design A/B: 2; Design C: 3
    num_heads=8,
    dropout=0.1,
    hand_aux_loss_weight=0.1,      # all designs
    aux_body_loss_weight=0.0,      # Design A: 0.0; Design B/C: 0.4
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
)
```

All values are int/float/bool literals. No Python import statements. Fully compliant with MMEngine config constraints.

---

## Expected Outcome

- **Primary gain — body MPJPE**: multi-layer progressive refinement over 22 clean body-only queries should push body MPJPE well below the 176mm floor from idea001 and the 184mm floor from idea008/design002. Target: `mpjpe_body_val < 170` at stage-2 (conservative), `< 165` (optimistic).
- **Pelvis MPJPE**: pelvis token (query 0) self-attends only to 21 body joint queries in all decoder layers, with no hand noise. Multi-layer refinement should improve the pelvis depth/UV regression features more than in idea001. Target: `mpjpe_pelvis_val < 320` at stage-2 (improving on idea001's 322.82mm).
- **Relative MPJPE**: following idea008/design002's pattern (333.2mm vs. baseline 438.7mm), the clean body-query cross-attention should preserve this gain. Multi-layer decoding should further improve it. Target: `mpjpe_rel_val < 300` at stage-2.
- **Composite target (stage-1)**: aim for `composite_val < 325` (best prior: 328.14 — idea013/design003).
- **Composite target (stage-2)**: aim for `composite_val < 215` (best prior: 224.52 — idea001/design001).
- **Design A**: cleanest test of the combination. Expected to replicate or beat idea001/design001 stage-2 composite by removing hand contamination. Primary bet for stage-2 gate pass.
- **Design B**: intermediate supervision should improve stage-1 convergence but may add noise to earlier layers. Expected slightly better or equal to Design A.
- **Design C**: 3 decoder layers enabled by memory savings. Higher risk but highest potential if 3 layers add discriminative power. If Design C OOMs (unlikely), reduce to 2 layers.

---

## Risk and Mitigation

- **Memory**: 22 queries reduce cross-attention by 69% and self-attention by 90%. Two decoder layers over 22 queries consume less VRAM than one layer over 70 queries for the attention components. The FFN components scale with query count and layers: 2 × 22 × FFN_params vs. 1 × 70 × FFN_params (FFN hidden_dim=1024). Net: 2×22=44 FFN applications vs. 1×70=70 → still cheaper. Design C (3 layers): 3×22=66 vs. 70 → also cheaper. Memory risk: very low.
- **hand_proj output-shape compatibility**: `bedlam_metric.py` and `BedlamMPJPEMetric` expect `(B, 70, 3)` joint output. The concatenation `cat([body_joints, hand_joints], dim=1)` produces exactly (B, 70, 3). The metric already ignores hand joints for body MPJPE. Shape compatibility: verified.
- **Hand auxiliary loss instability**: the hand projection is a linear layer from 22×256=5632 dims to 48×3=144 dims — a large compression. Weight 0.1 auxiliary loss provides a soft signal without dominating. SoftWeightSmoothL1Loss used (same as body joints) for outlier robustness.
- **Intermediate decoded queries for loss**: `joints_out` is shared across all decoder layers (single Linear(hidden_dim, 3)). Applying it to intermediate outputs (Design B/C) is straightforward and produces (B, 22, 3) intermediate predictions — same as the final output shape.
- **pelvis_token index**: in the baseline, `decoded[:, 0, :]` is used as the pelvis token. In the 22-query formulation, query 0 remains the pelvis token (same as idea008/design002). No change needed.
- **Interaction with idea008 design003 (MLP hand recovery)**: this idea uses linear hand recovery (simpler than idea008's design003 which used a 2-layer MLP). The linear projection is sufficient because the hand output is not evaluated by the metric — it is only a structural regularizer for the body decoder. The simplicity reduces the risk of the hand auxiliary loss dominating or destabilizing early training.
- **Gradient flow through intermediate outputs**: when `aux_body_loss_weight > 0`, gradients flow from the intermediate joint loss back through decoder layer 1 and into the backbone. This is the same gradient pattern as idea001/design002 (which trained successfully). No new gradient flow issues.
- **Forward compatibility with idea010 (reprojection loss)**: this idea is fully compatible with adding a reprojection loss as an additional term in `loss()`. The `recover_pelvis_3d` and `project_joints_to_2d` helpers in `pelvis_utils.py` can be used unchanged. This is left to a future composition idea.
- **MMEngine config constraint**: all kwargs (`num_body_queries`, `num_decoder_layers`, `hand_aux_loss_weight`, `aux_body_loss_weight`) are int/float literals. No Python import statements. Fully compliant.
- **Eval/inference compatibility**: `predict()` calls `self.forward(feats)` which now runs `num_decoder_layers` layers over `num_body_queries` queries, then concatenates hand projections. Output `(B, 70, 3)` shape is preserved. `BedlamMPJPEMetric`, `TrainMPJPEAveragingHook`, and `MetricsCSVHook` see identical interfaces. No downstream changes.
