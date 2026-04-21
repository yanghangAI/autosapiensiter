**Idea Name:** Joint-Group Spatial Routing via Learned Cross-Attention Gating

**Approach:** Learn a soft spatial routing mask per joint group (body-lower, body-upper, hands) that is added as an additive bias to the cross-attention logits, steering each group of joint queries to attend preferentially to the spatial tokens most relevant to that body region, reducing cross-attention noise from irrelevant spatial positions.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline cross-attention in the decoder allows every joint query to attend uniformly to all H'×W' spatial tokens. For a crop centred on a single person, the feature grid is 40×24 = 960 tokens (at 1/16 resolution for a 640×384 input). However, the spatial locations relevant to the left wrist are quite different from those relevant to the pelvis or the right knee. With no spatial routing prior, each joint query must discover which spatial regions to attend to solely from gradient signal over the dataset — a hard learning problem in 20 epochs.

This matters because:

1. **Body vs. hand separation**: joints 0–21 (body, actively evaluated) share the self-attention with joints 22–69 (hands). Hand joints cross-attend to finger-region spatial tokens; body joints attend to torso and limb tokens. The current uniform cross-attention means hand-specific spatial information (high-frequency texture of fingers) competes with body-level structural information (torso shape, limb silhouettes) in the same attention computation. Soft spatial routing separates these pathways.

2. **Upper vs. lower body**: lower-body joints (hips, knees, ankles) correspond to the bottom half of the crop; upper-body joints (spine, shoulders, elbows, wrists) correspond to the upper half. A routing bias that initialises these groups to attend to their respective spatial half requires near-zero gradient signal to maintain and immediately reduces attention "noise" from spatially distant tokens.

3. **Evidence from prior ideas**:
   - **idea001** (multi-layer decoder): stacking layers improved body MPJPE (−14 mm) but hurt pelvis MPJPE (+6 mm). The likely cause: repeated self-attention across layers gradually routes body structure information away from token 0 (pelvis query). Spatial routing would let body-joint queries focus on body-relevant spatial tokens directly without needing multiple decoder passes.
   - **idea006** (skeleton self-attention bias): modified query-to-query attention to encode kinematic structure. This idea is the cross-attention analogue: modifying query-to-spatial-token attention to encode spatial structure.
   - **idea004** (depth-aware spatial PE): enriched the spatial tokens themselves. This idea instead routes *which* spatial tokens each query group attends to, orthogonal to what information those tokens carry.

### What this idea adds

In `_DecoderLayer.forward()`, the cross-attention call is:

```python
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
```

`nn.MultiheadAttention` accepts an `attn_mask` argument added to cross-attention logits before softmax. We register a learnable parameter `cross_attn_bias` of shape `(num_joints, num_spatial)` initialised to zero, and pass it as the cross-attention `attn_mask`:

```python
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                     attn_mask=self.cross_attn_bias)[0]
```

Initialised to zeros, this recovers the baseline exactly. Over training, gradient updates will push positive entries toward joint-spatial pairs that benefit from stronger attention and negative entries toward pairs to suppress. Because the number of spatial tokens is determined by the feature grid size (H'×W' = 960 for the baseline 640×384 input at 1/16 stride), `cross_attn_bias` has shape `(70, 960)` — approximately 67,200 scalar parameters (~262 KB), negligible relative to the model.

An optional warm-start based on a vertical band prior (body-lower queries attend to lower spatial rows, body-upper queries to upper rows, hands to upper-lateral regions) can accelerate convergence within the 20-epoch budget.

---

## Proposed Variations

### Design A — Zero-initialised learnable cross-attention routing (diagnostic)

Add a single `nn.Parameter` of shape `(num_joints, num_spatial)` initialised to zeros, passed as `attn_mask` to the cross-attention. This is the minimal-change design: identical to the baseline at initialisation, but the parameter can learn any joint-to-spatial routing pattern. No structural prior is imposed; the model discovers the optimal routing from data alone.

This is the diagnostic baseline for the idea: does *any* learned spatial routing for cross-attention help, beyond what the dot-product already learns? Since the dot-product attention already has a learned routing mechanism, the additive bias adds a query-specific *global offset* per spatial token that supplements (rather than replaces) the content-based routing. Expected improvement: −5 to −10 mm body MPJPE from reduced cross-attention noise.

**Implementation**: 5 lines in `pose3d_transformer_head.py`. The spatial dimension `num_spatial = H' * W'` must be computed in `__init__` from the known input resolution (640×384 at 1/16 stride → 40×24 = 960). Hardcode 960 as a default with a constructor kwarg `num_spatial: int = 960`.

### Design B — Vertical band warm-start (body-part prior)

Same as Design A but initialise the `cross_attn_bias` with a structured prior based on the spatial position of each joint group in a typical person crop:

- **Body-lower joints** (indices: hips 1,2; knees 4,5; ankles 7,8; feet 10,11 — approximately indices `[1,2,4,5,7,8,10,11]`): initialise their bias rows to a Gaussian centred on the lower half of the spatial grid (rows 20–40 of the 40-row grid). The bias is `+0.5` in the lower-half tokens and `−0.5` in the upper-half tokens.

- **Body-upper joints** (indices: pelvis 0, spine 3,6,9, neck 12, head 15, shoulders 13,14, elbows 16,17, wrists 18,19, hands 20,21 — approximately indices `[0,3,6,9,12,13,14,15,16,17,18,19,20,21]`): initialise their bias rows to a Gaussian centred on the upper half of the spatial grid (rows 0–20).

- **Hand joints** (indices 22–69): initialise to `0.0` (no prior — the hands are scattered and not evaluated).

The Gaussian width is set to ~10 spatial rows (σ = 5 rows in the 40-row dimension), so the bias is smooth, not a hard cutoff. This warm-start encodes common-sense anatomy without hardcoding exact positions, allowing the learned bias to deviate as needed for the actual BEDLAM2 data distribution (where crops are tightly person-centred).

**Implementation**: the prior can be computed analytically in `__init__` using a nested loop or `torch.arange` operations — no external imports. Hardcode the joint group index lists as Python lists in `pose3d_transformer_head.py`.

### Design C — Per-head cross-attention routing (richest)

Learn `num_heads` independent routing bias matrices, each of shape `(num_joints, num_spatial)`, all initialised to zeros (or with the vertical band prior from Design B). Pass the per-head bias as `(B * num_heads, num_joints, num_spatial)` to cross-attention.

This allows each attention head to develop a different spatial routing specialisation: one head might focus body-lower queries to the lower spatial region, another might route pelvis query (token 0) specifically to the centre of the crop regardless of body part, and a third might learn a global attention pattern for hand queries. Parameter cost: `8 × 70 × 960 ≈ 537,600` scalars (~2 MB) — still small relative to the backbone.

**Implementation note**: as with idea006 Design C, this requires passing `B` into the decoder layer forward to expand `(num_heads, J, S)` → `(B * num_heads, J, S)`. A minor refactor of `_DecoderLayer.forward(queries, spatial_tokens, B=None)` suffices. This is orthogonal to idea006's self-attention bias, and both can be combined if desired.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py`:

1. **`_DecoderLayer.__init__`**: Accept optional `cross_attn_bias_init: Optional[torch.Tensor]` and `num_heads_cross: int = 1`. Register `self.cross_attn_bias = nn.Parameter(cross_attn_bias_init or torch.zeros(num_joints, num_spatial))`.

2. **`_DecoderLayer.forward`**: Pass `attn_mask=self.cross_attn_bias` to `self.cross_attn(q, spatial_tokens, spatial_tokens, ...)`.
   - Design C: expand to `(B * num_heads, J, S)` before passing; requires `B` as an argument.

3. **`Pose3dTransformerHead.__init__`**: Accept `cross_routing_type: str = 'none'` as a config kwarg (`'zero_init'`, `'band_prior'`, `'per_head'`). Construct `_DecoderLayer` with the appropriate `cross_attn_bias_init`. Also accept `num_spatial: int = 960` matching the feature grid at 1/16 resolution for the 640×384 input.

4. **`config.py`**: Add `cross_routing_type` and optionally `num_spatial` to head kwargs as string/int literals. No Python imports needed.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

---

## Expected Outcome

- **Primary gain**: improved body MPJPE by steering body-joint queries to spatially relevant tokens without requiring additional decoder layers.
- **Pelvis**: expected to maintain or improve (unlike idea001). The pelvis query (token 0) is in the body-upper group in Design B's prior, so it receives a coherent spatial focus rather than being pulled by hand-joint cross-attention from distal spatial regions.
- **Design A**: diagnostic — does any learned routing help? Expected −5 to −10 mm body MPJPE at convergence.
- **Design B**: expected to converge faster than A due to anatomically grounded warm-start; may improve both body and pelvis simultaneously. Primary bet.
- **Design C**: richest variant; each head specialises independently. Highest potential but may require more epochs.
- **Composite target**: aim for composite_val < 160 (vs. baseline 169.75, idea001 best design001 = 162.00 at epoch 13).

---

## Risk and Mitigation

- **Fixed `num_spatial` assumption**: the routing bias shape must match the actual spatial token count at runtime. For 640×384 input at 1/16 stride the grid is 40×24 = 960. If the feature stride differs, the shape will mismatch and cause a runtime error. The Designer should assert `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` in `_DecoderLayer.forward` to catch this early.

- **attn_mask convention in PyTorch cross-attention**: `nn.MultiheadAttention` treats `attn_mask` as an additive pre-softmax bias. The Designer should confirm that cross-attention (not just self-attention) accepts `attn_mask` of shape `(T_q, T_k)` = `(num_joints, num_spatial)`. PyTorch ≥1.9 supports this for both self- and cross-attention.

- **Interaction with idea006 (self-attention bias)**: these two biases are independent (self-attn vs. cross-attn). They can be combined in a single design if desired. The Designer may optionally test a combined variant in Design C.

- **Memory**: no additional attention computation. The bias is a constant additive to existing attention logits. Negligible memory overhead (262 KB for Design A/B; ~2 MB for Design C).

- **Training speed**: single element-wise addition to cross-attention logits. No measurable overhead.

- **Warm-start numerics in Design B**: a `±0.5` bias corresponds to a ~60% vs. 40% raw attention split before softmax, which is a gentle prior that does not completely suppress any spatial token. This prevents gradient vanishing in suppressed regions and allows the prior to be overridden if data evidence is strong.

- **MMEngine config constraint**: `cross_routing_type` is a simple string literal. The spatial grid indices and Gaussian bias computation are hardcoded arithmetic in `pose3d_transformer_head.py`. No imports in `config.py` required. Fully compliant with the MMEngine no-imports rule.
