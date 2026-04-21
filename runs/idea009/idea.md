**Idea Name:** Spatial Token Dropout for Cross-Attention Regularization

**Approach:** During training, randomly mask a fraction of spatial tokens before they are used as keys/values in cross-attention, forcing joint queries to aggregate pose information from diverse feature locations rather than overfitting to a fixed set of dominant spatial anchors.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline cross-attention has joint queries attending to all H'×W' = 960 spatial tokens every step. On BEDLAM2 (synthetic, clean data), the backbone's feature map is highly consistent across training samples: the same anatomical regions always produce the strongest spatial activations. Over training, the joint queries can latch onto a small number of dominant spatial "anchor" tokens — e.g., a particular torso region or a characteristic depth gradient pattern — and stop attending broadly. This creates two problems:

1. **Over-reliance on a few spatial tokens**: if those tokens are ambiguous or shifted (due to NoisyBBoxTransform crop jitter), prediction degrades. The model has not learned to aggregate from surrounding context.

2. **Pelvis localisation fragility**: the pelvis token (query 0) must learn to attend to the correct spatial region to support both root-relative joint decoding and absolute pelvis depth/UV regression. A model that overfits spatial routes for body joints may do so at the expense of the broader spatial context needed for pelvis localisation.

**Spatial token dropout** addresses this by randomly zeroing out (via attention masking) a fraction `p_drop` of spatial tokens during each training forward pass. At inference, all tokens are used. This is analogous to DropPath/DropKey in vision transformers and has been shown to:
- Improve generalisation by acting as a data-augmentation in feature space
- Force the attention mechanism to spread mass across more spatial locations
- Reduce dependence on any single dominant feature region

This is a different regularisation axis from everything tried so far:
- idea001 (more decoder layers): architectural capacity change
- idea002 (dedicated pelvis query): structural decoupling
- idea003 (content-adaptive queries): query initialisation
- idea005 (uncertainty loss weighting): gradient magnitude balancing
- idea006 (self-attention bias): query–query interaction structure
- idea007 (cross-attention gating): query-group routing
- idea008 (body-only decoder): query count reduction

None of these add regularisation to the **spatial token stream**. Spatial token dropout is orthogonal to all of them and can be composed with any.

### Connection to results.csv

`idea001/design001` (2-layer decoder, no aux loss) achieved the best composite_val so far at epoch 14 (162.67 vs baseline 171.12), with body MPJPE dropping by 15 mm but pelvis MPJPE rising by 5 mm. The body improvement with more capacity suggests the model has room to improve body decoding, while the pelvis regression suggests the additional self-attention capacity hurts the pelvis token. Spatial token dropout is a complementary fix: instead of changing attention *structure*, it regularises what the queries are allowed to see, which should benefit generalisation without introducing the pelvis-regression pattern.

---

## Proposed Variations

### Design A — Uniform Spatial Token Dropout (p=0.15)

Drop 15% of spatial tokens uniformly at random during training cross-attention. At inference, use all tokens (standard DropKey behaviour).

**Implementation:**
- In `_DecoderLayer.forward()`, add a `spatial_drop_prob: float = 0.0` argument.
- During `self.training` and `spatial_drop_prob > 0`, generate a random binary mask `mask = torch.rand(B, N_spatial) < spatial_drop_prob`. Set masked positions to `float('-inf')` in the cross-attention key padding mask (`key_padding_mask` argument of `nn.MultiheadAttention`).
- During inference (`not self.training`), pass `key_padding_mask=None`.
- `Pose3dTransformerHead.__init__` accepts `spatial_drop_prob: float = 0.0` and passes it to `_DecoderLayer`.
- `config.py` adds `spatial_drop_prob=0.15` to head kwargs.

This is the lightest variant. Expected to improve generalisation and reduce pelvis regression pattern.

### Design B — Moderate Spatial Token Dropout (p=0.30)

Same mechanism as Design A with higher drop rate 30%. More aggressive regularisation; the model must reconstruct pose from ~672 of 960 tokens. Tests whether the benefit saturates or continues improving.

**Implementation:** identical to Design A, with `spatial_drop_prob=0.30` in config.

### Design C — Structured Spatial Token Dropout with Annealing (p=0.30 → 0.10)

Apply spatial token dropout starting at p=0.30 for the first 10 epochs, annealing linearly to p=0.10 for the remaining 10 epochs. The motivation: high dropout early in training encourages broad spatial exploration (better query initialisation); lower dropout later lets the model refine precise localization on the full token set.

**Implementation:**
- `Pose3dTransformerHead` accepts `spatial_drop_prob_start: float` and `spatial_drop_prob_end: float`.
- Add a `set_drop_prob(p: float)` method that updates `self.spatial_drop_prob`.
- Add a custom hook `SpatialDropAnnealHook` in `pose3d_transformer_head.py` (registered via `HOOKS`) that calls `model.head.set_drop_prob(...)` at the start of each epoch using a linear schedule.
- `config.py` adds the hook to `custom_hooks` and sets the head kwargs `spatial_drop_prob_start=0.30`, `spatial_drop_prob_end=0.10`.
- Since the hook is defined in `pose3d_transformer_head.py`, it gets imported via the existing `custom_imports` entry `'pose3d_transformer_head'`.

This is the most principled variant, but also most complex. Designer should verify hook registration.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`:

**`pose3d_transformer_head.py`:**
1. `_DecoderLayer.__init__`: no change needed (drop prob passed per-call).
2. `_DecoderLayer.forward`: add `spatial_drop_prob: float = 0.0` argument. If training and `spatial_drop_prob > 0`, compute `key_padding_mask` of shape `(B, N_spatial)` as boolean mask (True = masked out). Pass to `self.cross_attn(..., key_padding_mask=key_padding_mask)`.
3. `Pose3dTransformerHead.__init__`: accept `spatial_drop_prob: float = 0.0` (Designs A/B); or `spatial_drop_prob_start`, `spatial_drop_prob_end` (Design C). Store as `self.spatial_drop_prob`.
4. `Pose3dTransformerHead.forward`: pass `self.spatial_drop_prob` to `self.decoder_layer(queries, spatial, spatial_drop_prob=self.spatial_drop_prob)`.
5. (Design C only) `SpatialDropAnnealHook`: lightweight custom hook that updates `self.spatial_drop_prob` each epoch using linear interpolation between start and end values.

**`config.py`:**
- Add `spatial_drop_prob=0.15` (Design A), `=0.30` (Design B), or `spatial_drop_prob_start=0.30, spatial_drop_prob_end=0.10` (Design C) to head kwargs.
- Design C: add `dict(type='SpatialDropAnnealHook', num_epochs=20, start_prob=0.30, end_prob=0.10)` to `custom_hooks`.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

**Key constraint:** `nn.MultiheadAttention` with `batch_first=True` accepts `key_padding_mask` as a boolean tensor of shape `(B, S)` where `True` means "ignore this position." The random mask must be generated fresh each forward call (not registered as a buffer). This is standard PyTorch usage and compatible with 1080 Ti (no memory concern — mask is (B, 960) float/bool).

---

## Expected Outcome

- **Primary gain**: more robust spatial attention → better body MPJPE generalisation, particularly for edge cases where crop jitter places key joints near feature map boundaries.
- **Pelvis**: dropout applies equally to all queries including token 0. Since pelvis needs global spatial context, moderate dropout may actually help the pelvis token learn to integrate from diverse spatial evidence rather than latching onto a narrow set of tokens. Expected to maintain or improve pelvis MPJPE vs baseline.
- **Design A** (p=0.15): minimal disruption, expected composite_val improvement of −5 to −10 mm. Low risk.
- **Design B** (p=0.30): more aggressive regularisation; if the model is currently overfitting spatial routes, this should yield −10 to −15 mm. Moderate risk of underfitting if p is too high.
- **Design C** (annealing): best of both worlds; expected −10 to −18 mm. Highest implementation complexity but well-motivated by training dynamics.
- **Composite target**: aim for composite_val < 160 (vs. baseline 171.12, idea001 best = 162.67).

---

## Risk and Mitigation

- **All tokens masked for a sample**: with p=0.30 and N=960, the expected number of unmasked tokens is 672. Probability of any token being the last unmasked one is negligible. No guard needed.
- **key_padding_mask all-True for any query**: the drop mask is applied uniformly across queries (shared per spatial token per batch element). It is not per-query. So all queries see the same dropped set, ensuring no query is starved. If needed, Designer can resample per query (independent masks), but shared masking is simpler and sufficient.
- **Design C hook registration**: if MMEngine's `HOOKS.register_module` fails because the hook file is imported late, Designer should fall back to registering it inside the `custom_imports` block using `from mmengine.registry import HOOKS`. Alternatively, if hooks must be in a separate file, Design C can be simplified to Design B with a cosine schedule instead of linear annealing (implemented as a fixed value but the Designer can adapt).
- **Interaction with multi-layer decoder (idea001)**: spatial token dropout would compose naturally. The Designer exploring this direction should start with baseline (single-layer) to isolate the signal before combining.
- **MMEngine config constraint**: `spatial_drop_prob` is a float literal. No imports required. Fully compliant.
