**Idea Name:** Camera-Intrinsic Conditioning of Decoder via FiLM on Normalized K

**Approach:** Inject the per-sample camera intrinsics (fx, fy, cx, cy, crop_h, crop_w) into the transformer decoder via a small MLP that produces FiLM (gamma, beta) affine parameters, which modulate either the joint queries, the spatial tokens, or both — giving the head explicit awareness of projective geometry so that absolute scale and pelvis UV/depth can be reasoned about conditionally on the actual imaging geometry of each sample.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Missing Camera Signal

In the current baseline, the pose head has *no access* to the per-sample camera intrinsic matrix K. Intrinsics are only used:

1. Downstream, in `pelvis_utils.recover_pelvis_3d` to unproject predicted `(pelvis_depth, pelvis_uv)` into absolute 3D for the `mpjpe_abs` metric.
2. In some prior ideas (idea010 reprojection loss, idea022 reprojection-conditioned inter-layer attention) as a **loss-time** or **post-layer** projection operator.

No prior idea has conditioned the *forward computation of the decoder itself* on camera intrinsics. The head therefore treats every sample as if it had the same camera, relying on the backbone features to implicitly encode geometric cues. For BEDLAM2 this is a non-trivial omission:

- Training images are random crops from a large synthetic corpus with heterogeneous virtual cameras — fx, fy, cx, cy vary substantially sample-to-sample.
- The crop operation itself produces a sample-specific K (crop offset shifts cx, cy; crop scale changes fx, fy).
- Pelvis UV is defined in normalized crop pixels, whose meaning in world units depends entirely on (fx, fy) and the depth.

The mapping from normalized image-space features (what the backbone sees) to metric 3D (what we regress) is a geometric transformation whose exact form depends on K. When K is unknown to the regressor, that mapping must be averaged across the training distribution, which is a well-documented cause of systematic depth/scale ambiguity in monocular 3D pose (e.g., CLIFF, SPEC and related camera-aware works in SMPL regression literature).

### Why This Targets the Persistent Pelvis Plateau

Across all 32 prior ideas, `mpjpe_pelvis_val` has been the hardest metric to move:
- Baseline stage-1 pelvis MPJPE: 652.89 mm; best: 608.64 mm (idea023/design001), only −6.8% improvement.
- Meanwhile `mpjpe_body_val` has moved from 195.72 to 183.43 (idea023/design001), a similarly small absolute but structurally different improvement coming from better attention / query init.

Pelvis localization requires absolute-scale reasoning, which is *exactly* the subproblem that camera intrinsics disambiguate. A query that knows fx can learn "this pelvis-region feature activation at this depth maps to X=5m at this crop intrinsic but would map to X=3m at a wider-FoV intrinsic." The same depth-scalar from the head means different world depths under different K — and the head currently has no way to know.

This is distinct from:
- **idea004** (depth-aware spatial positional encoding): uses the *input depth channel*, not the camera intrinsic.
- **idea016** (FiLM from pooled spatial tokens): conditions on the *internal feature summary*, not an external geometric signal. The gamma/beta come from the network's own perception, not from ground-truth projective geometry.
- **idea010 / idea022** (reprojection-based losses / post-hoc attention biases): use K at loss time or between layers, not as a conditioning signal to the computation itself.
- **idea032** (auxiliary depth-map reconstruction): supervises the spatial tokens to retain *sensor* depth, not camera intrinsics.

No prior idea feeds K into the forward pass as a conditioning vector.

### What This Unlocks

A K-conditioned FiLM modulation lets the decoder:

1. **Scale-calibrate the pelvis depth head.** The depth regressor currently outputs a scalar that is implicitly a statistic over the training K distribution. With K as a conditioning input, depth predictions can become proper function `depth = f(features, K)`.
2. **De-bias UV prediction under off-center crops.** When the crop is not centred on the person, cx/cy shift; the head can learn to compensate for the principal-point offset.
3. **Share body-query computation** across K while specializing the pelvis head — since the body joints are root-relative in metres, they are K-invariant, but the pelvis mapping from image coords to world coords is K-dependent.
4. **Act as geometric prior injection.** The FiLM branch explicitly encodes a 6-dimensional (fx, fy, cx, cy, crop_h, crop_w) vector that the head would otherwise need to reconstruct indirectly from spatial features — a clear bandwidth saving.

### Why Now

The pelvis MPJPE plateau has been stable across 32 prior ideas spanning attention biases, query init, loss reweighting, iterative refinement, output parameterization, bin-classification depth heads, UV heatmap heads, and auxiliary depth reconstruction. The remaining untouched axis is **providing the head with information it does not currently have**. Camera intrinsics are such information: already present in `ds.metainfo['K']` and `ds.metainfo['img_shape']`, already used by `recover_pelvis_3d`, but never routed into the decoder itself.

## Rough Mechanism Sketch (details for Designer)

For each sample, extract a 6-dim vector `k = [fx/W_ref, fy/H_ref, cx/W, cy/H, crop_h/H_ref, crop_w/W_ref]` (normalized to be scale-invariant and numerically well-behaved). A small MLP `k → (gamma, beta)` in `R^{2 * hidden_dim}` produces FiLM affine parameters.

Three natural variants for where to inject:

- **Variant A — Query FiLM only.** Apply `q <- gamma * q + beta` to the 70 joint queries before the decoder layer. Cheapest; all queries get the same K-conditioning. Good for testing whether any K signal helps.
- **Variant B — Spatial-token FiLM only.** Apply FiLM to the projected spatial tokens after input_proj and positional encoding. Modulates the key/value source.
- **Variant C — Pelvis-token FiLM at output.** Apply FiLM only to the pelvis token (token 0) *just before* `depth_out` and `uv_out`. Most targeted: the K signal directly modulates the two outputs that are mathematically K-dependent, while leaving body-joint regression (K-invariant in root-relative metres) untouched. This is the design most aligned with the causal structure of the problem.

Zero-init the FiLM's final linear for gamma and beta so the module starts as identity (gamma=1, beta=0), guaranteeing exact baseline recovery at step 0 — a consistent safe-init pattern used across idea011/idea013/idea021 and others.

## Scope & Invariants

- Changes confined to `pose3d_transformer_head.py` (new FiLM module, new forward-path reads of K from `batch_data_samples` metainfo) and `config.py` (bool/str/int/float literals for variant and MLP width).
- `pelvis_utils.py` unchanged.
- `bedlam_metric.py`, `bedlam2_dataset.py`, backbone, `train.py` unchanged — K is already exposed in `metainfo['K']` and `metainfo['img_shape']`.
- Output shapes unchanged; loss signatures unchanged.
- FiLM MLP parameter count is small (~2 * hidden_dim * 32 + 32 * 6 ≈ 16K params with hidden_dim=256).
- Note for Designer: `forward()` does not currently receive `batch_data_samples`; it only receives `feats`. The Designer will need to route K either (a) by augmenting forward to accept an optional K tensor built upstream in `loss()` / `predict()`, or (b) by stashing K onto the head via a pre-forward hook. Option (a) is cleanest and mirrors how `compute_mpjpe_abs` already threads `batch_data_samples` through the loss path.

## Composition with Prior Ideas

Orthogonal to all prior work (K is a new input signal). Natural composition partners:
- **idea002 / idea028** (decoupled pelvis decoder) — apply FiLM only on the pelvis branch.
- **idea014** (binned depth classification) — K conditioning of the bin-prediction softmax gives per-sample adaptive depth ranges.
- **idea031** (UV heatmap head) — K conditioning of the heatmap logits helps align the UV output distribution to actual sensor geometry.

## Success Targets

- Stage-1 composite_val < 325 (to match top tier of idea023 / idea013 design003).
- Stage-1 `mpjpe_pelvis_val` < 600 (breaking the 608 floor set by idea023/design001).
- Stage-1 `mpjpe_abs_val` < 780 (beating idea008/design002 at 747 is a stretch goal).
- Primary outcome of interest: whether `mpjpe_abs_val` improves disproportionately more than relative body MPJPE, which would confirm the K-conditioning mechanism is working as theorized.
