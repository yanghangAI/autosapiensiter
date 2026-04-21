**Idea Name:** Anchor-Based Pelvis Depth via Discretized Classification Head

**Approach:** Replace the single-scalar pelvis-depth regression (one `Linear(hidden_dim, 1)` producing a continuous metre value) with a discretized-bin classification head that predicts a softmax distribution over K pre-defined depth anchors spanning the BEDLAM2 pelvis-depth range, then recovers the continuous depth as the expectation of the distribution (soft-argmax), supervising via cross-entropy on a target soft-label centered at the GT depth.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The persistent pelvis-depth bottleneck

Across all 13 prior ideas (including the three still training), `mpjpe_pelvis_val` has stayed tightly in the 174–185 mm band, while `mpjpe_body_val` has improved from 165 (baseline) → 141 (idea002/design002). The pelvis metric has been the most stubborn component and dominates the 33 % weight in the composite; the best `mpjpe_pelvis_val` ever observed is 174.43 (idea009/design003), a gain of only **1.67 mm** over baseline (176.10). By contrast, body MPJPE has moved by **24 mm** over the same window.

Looking at why all prior pelvis-targeted ideas underwhelmed:

| Prior idea | Pelvis approach | Result |
|---|---|---|
| idea002 (decoupled pelvis query) | Architectural: pelvis query separated from joint queries | Improved composite via body gains but pelvis stayed at 176–184 |
| idea004 (depth-aware PE) | Input-side: depth channel used to modulate spatial PE | Pelvis unchanged; body slightly worse |
| idea005 (uncertainty weighting) | Loss-level: learnable variance balancing | Pelvis unchanged (176–179) |
| idea010 (2D reprojection loss) | Loss-level: geometry-aware 2D consistency | Early training (epoch 1) shows very high pelvis MPJPE; coupling not yet evident |

All attacked the *architecture* upstream of the depth head or the *loss balance* but **none changed what the depth head itself predicts**. The depth head remains `Linear(hidden_dim, 1)` — a direct scalar regression of a continuous metre value — for every idea 000-013.

### The case for discretizing depth

Monocular depth estimation has shown repeatedly (DORN, Fu et al. CVPR 2018; AdaBins, Bhat et al. CVPR 2021; BinsFormer, Li et al. TCSVT 2024) that **classification-over-bins + soft-argmax** outperforms direct regression for depth prediction, especially for noisy / ambiguous inputs. The reasons carry over almost directly to the pelvis-depth task here:

1. **Richer gradient signal per training example.** Direct L1/smoothL1 regression on a scalar gives one gradient per sample (dL/dz_pred = sign(z_pred − z_gt) × β). Classification cross-entropy gives a gradient that is spread across K bins: it *simultaneously* increases the probability of the correct bin while decreasing the probability of all incorrect bins. The network has K-way learning signal per sample instead of a single scalar directional nudge.

2. **Output representation better matches the multi-modal nature of depth ambiguity.** A person standing near vs. far can look nearly identical in a cropped RGB image, which is a major source of 3D-pose depth error. A classification head can maintain probability mass on multiple modes during training and gradually concentrate as more cues are learnt — a direct regression commits to a single value at every forward pass. When the data is genuinely ambiguous (as it is under BEDLAM2 RGB-only; recall the *depth channel* is the backbone-modality signal, but the pelvis depth itself is still recovered only from a single global token), a multimodal output head trains more stably.

3. **Soft-ordinal targets exploit the metric structure of depth.** In vanilla classification, all wrong bins are equally wrong. Soft-ordinal targets (SORD, Diaz & Marathe CVPR 2019) assign higher target probability to bins near the GT depth and lower probability to far bins, so the loss agrees with the ordinal/metric structure of the problem. This is effectively a smooth integral of distance-weighted cross-entropy and is known to be both more stable than ranking-based ordinal losses and better-matched to MPJPE (which is an L2-style metric).

4. **Bounded, well-scaled gradient.** Regression losses on depth are scale-sensitive (a gradient of `sign` × `β` at large depths is the same magnitude as at small depths, even though the expected error scales with depth). Classification gradients are inherently bounded in [0, 1] and are independent of the depth scale, which gives more consistent optimization dynamics across near- and far-depth examples.

5. **Very low parameter cost with better signal-to-noise.** K=64 bins means the head becomes `Linear(hidden_dim, 64)` instead of `Linear(hidden_dim, 1)`. That is +16 k params vs +256 for the baseline — still **under 0.02 %** of the head's parameter count, well within our budget. The depth head sees 63× more gradient surface per sample.

6. **Depth range is fixed and well-known in BEDLAM2.** Synthetic renderings have a known camera-distance distribution (roughly 1–15 m for pelvis). A fixed uniform-in-log-depth bin grid is trivial to define and does not require learning.

### What this is *not*

- Not a loss addition (unlike idea010, idea012). The depth *output head* is what changes, along with the matching loss.
- Not an architectural change upstream of the head (unlike idea001, idea002, idea003, idea004, idea006, idea007, idea008, idea009, idea011). The decoder body, queries, cross-attention, and loss-balance are all unchanged.
- Not an output parameterization for joints (unlike idea013, which reparameterizes 22 body-joint outputs as bone vectors). This idea only touches the pelvis depth output — hand and body joint heads are unchanged.
- The pelvis UV head is unchanged and continues to produce a continuous 2-vector in `[-1, 1]`.

### Why this is orthogonal to every prior idea

- idea001/002/003/004/006/007/008/009/011: change decoder internals / queries / attention / training dynamics. The depth head remains a scalar linear regression. This idea replaces *just* the depth head, so both changes compose cleanly.
- idea005: rebalances task-loss weights (joint/depth/uv). It leaves the depth loss functional form untouched. Replacing depth regression with classification+expectation is orthogonal — the rebalancing logic still works on whatever scalar `loss/depth/train` produces.
- idea010: 2D reprojection loss — operates on the *joints* head and the *recovered absolute pelvis*, which itself depends on the pelvis depth. If pelvis depth is more stable, idea010's geometric consistency signal becomes more informative. These compose cleanly.
- idea012/013: body-joint structural priors. They leave depth untouched. Fully orthogonal.

## Analysis of Baseline Weak Point

The baseline depth head is a single `Linear(256, 1)` applied to token 0 of the decoder. GT pelvis depth is a scalar in [≈1, ≈15] metres, and the loss is `SoftWeightSmoothL1Loss(beta=0.05)`. Consider a training example where the true depth is 4 m:

- If `z_pred = 4.3`, gradient direction is negative; the magnitude depends only on `β` (since |Δ| > β). The network updates the head's weights in a single direction.
- The optimizer never sees "the prediction was *directionally* reasonable but *calibrated* poorly"; it only sees a single scalar error. 
- The head has **zero inductive bias** about the depth distribution. It must learn, from gradient alone, that BEDLAM2 pelvis depths concentrate in ≈2–8 m, even though this is a fixed, known dataset property.

Converting to K bins over a fixed range encodes the depth-range prior into the output head's structure. The softmax naturally normalises over the valid range; unused bin slots decay under cross-entropy pressure without ever emitting spurious predictions outside the valid range.

Empirically, when body MPJPE is improving faster than pelvis MPJPE across 13 ideas, the most likely diagnosis is that the body-feature channels in the ViT are being sharpened by improving decoder changes, while the *pelvis scalar-output mapping* has been at its information-theoretic limit for the baseline scalar regression formulation. A classification head sidesteps the scalar bottleneck.

## Proposed Variations

**Design A — Uniform-log-depth bins + soft-argmax expectation (minimal bin classification)**

- `depth_out` becomes `Linear(hidden_dim, K)` with K = 64 bins.
- Bin centres: `log-uniform` over `[z_min, z_max] = [1.0, 15.0]` metres (so bin centres are `exp(linspace(log(1), log(15), K))`).
- Forward: `logits = depth_out(pelvis_token); probs = softmax(logits); z_pred = sum(probs × bin_centres)` (soft-argmax expectation — fully differentiable).
- Loss: soft-target cross-entropy, where the target distribution is a Gaussian centred at `z_gt` with σ=1.5 × bin_width (log-space). This is SORD-style soft-ordinal targets.
- The existing `SoftWeightSmoothL1Loss(beta=0.05)` on depth is replaced by `depth_ce_loss`. The reconstructed `z_pred` is still used by `recover_pelvis_3d`, so `compute_mpjpe_abs` is unaffected.

**Design B — Design A + auxiliary regression loss on expectation (hybrid, max stability)**

Same as Design A, but add a small auxiliary SmoothL1 on the expectation against the GT depth:
```
L_depth = L_CE(logits, soft_targets) + λ_reg × SmoothL1(z_pred_expected, z_gt)    (λ_reg = 0.3)
```
The two losses are complementary: cross-entropy trains the bin-probability landscape, the regression term guarantees the argmax-recovered scalar is metrically correct. This mirrors the hybrid formulation in BinsFormer and has been shown to stabilize early-training convergence when bins are wide and the classification signal alone is low-resolution.

**Design C — Design B + predicted per-sample depth range (adaptive bins, à la AdaBins)**

Instead of fixed `[z_min, z_max] = [1.0, 15.0]`, predict per-sample bin widths:
- A small `Linear(hidden_dim, K)` on token 0 produces a logits vector, softmaxed + cumulative-summed to produce K+1 cumulative edges in `[z_min, z_max]`.
- Bin centres = midpoints of consecutive edges.
- Classification logits + soft-argmax as in Design B.
- This allows the head to concentrate resolution where the sample's pelvis is likely to be (e.g., tighter bins near 4 m for close subjects, broader bins for far).
- Adds ~16 k params. Still under 0.02 % of head parameters.

Rationale for Design C: AdaBins showed that adaptive bin widths give significant gains over fixed bins in monocular depth estimation. However, the benefit might not materialize in 20 epochs on a single-crop pelvis-depth task where the range variance is lower than full scene depth. Design C is the most ambitious of the three and serves as a diagnostic for whether fixed uniform bins are the bottleneck.

## Implementation Scope

Changes confined to the two allowed files.

### `pose3d_transformer_head.py`

1. `__init__`: accept new kwargs
   - `depth_head_type: str = 'regression'` (values: `'regression'` [baseline], `'classification'` [Design A], `'classification_hybrid'` [Design B], `'classification_adaptive'` [Design C]).
   - `num_depth_bins: int = 64`.
   - `depth_range_min: float = 1.0`, `depth_range_max: float = 15.0`.
   - `depth_soft_label_sigma: float = 1.5` (multiplier on bin width in log-space).
   - `depth_aux_reg_weight: float = 0.0` (Design B/C: 0.3; Design A: 0.0).
2. `__init__`: when classification mode, allocate:
   - `self.depth_out = Linear(hidden_dim, num_depth_bins)` (replaces the scalar head).
   - `self.register_buffer('log_bin_centres', torch.linspace(log(z_min), log(z_max), K))`.
   - For Design C: additional `self.depth_bins_head = Linear(hidden_dim, num_depth_bins)`.
3. `forward()`:
   - In classification mode, compute `probs = softmax(depth_out(pelvis_token))` and `pelvis_depth = sum(probs × exp(log_bin_centres))` (or adaptive centres for Design C). The returned dict still has a key `'pelvis_depth': (B, 1)` — downstream code is unchanged.
   - Additionally expose `'depth_logits'` and `'depth_bin_centres'` (buffer / per-sample) for the loss function.
4. `loss()`:
   - When classification mode, compute soft-target distribution: `target = softmax(-(log_bin_centres − log(z_gt))^2 / (2 σ^2))`.
   - Loss: `(-target × log_softmax(logits)).sum(-1).mean()` (soft cross-entropy, equivalent to KL divergence up to an entropy constant).
   - Design B/C: add `depth_aux_reg_weight × SmoothL1(expected_depth, z_gt)`.
5. `predict()`: unchanged — reads `pred['pelvis_depth']` which is the soft-argmax expectation (metric scalar). `BedlamMPJPEMetric` and `TrainMPJPEAveragingHook` see the same tensor shape and units.

### `config.py`

Four new string/int/float literals:
- `depth_head_type: 'classification' | 'classification_hybrid' | 'classification_adaptive'` (all three Designs use this kwarg with different string values — no Python imports).
- `num_depth_bins: 64` (int literal).
- `depth_range_min: 1.0`, `depth_range_max: 15.0` (float literals).
- `depth_soft_label_sigma: 1.5` (float literal).
- `depth_aux_reg_weight: 0.0 | 0.3` (float literal).

No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or `train.py` wrapper. `recover_pelvis_3d` receives the scalar `pelvis_depth` as before.

## Expected Outcome

- **Primary gain — mpjpe_pelvis_val**: Target `< 170` mm (improving on the 174.43 floor of idea009/design003). The classification-over-bins structure gives the depth head a richer gradient signal and an explicit depth-range prior that a scalar regression lacks.
- **mpjpe_abs**: Expected positive. Absolute MPJPE depends directly on pelvis depth accuracy; even a 5 mm pelvis-depth improvement produces proportional mpjpe_abs gains (since the abs-MPJPE formula is `||pred_pelvis + pred_joints_rr − gt_pelvis − gt_joints_rr||`, and pelvis error propagates to every joint). Target `< 440` mm (beating baseline 454.91; best so far 320 from idea008/design003 but via very different mechanism).
- **mpjpe_body_val**: Expected neutral. The body joints share no weights with the depth head. The only indirect coupling is via the pelvis token 0 being shared between depth and body-via-self-attention — but self-attention is unchanged.
- **Hand MPJPE**: Expected neutral. Hand path fully unchanged.
- **Composite target**: `composite_val < 160`, with upside if the pelvis-depth improvement is large enough that composite drops into the 150s (since pelvis carries 33 % of the composite metric).

## Risk and Mitigation

- **Init-to-zero equivalence**: With `depth_out` trunc-normal initialised, softmax output is near-uniform (= 1/K for all bins), and the expectation is near the geometric mean of bin centres ≈ `exp(mean(log(z_min), log(z_max))) = sqrt(1.0 × 15.0) ≈ 3.87 m`. This is a *reasonable* starting prediction (most BEDLAM2 pelvises are 2–8 m). The expectation will have near-zero variance across samples at init, which is the same failure mode as baseline regression starting from zero bias. Mitigation: none needed; the loss rapidly spreads probability mass toward the correct bin once training starts. If Designer wants the expectation at init to be closer to baseline's learned mean, initialise `depth_out.bias` to a vector where entry i = -(log_bin_centre_i − log(4.0))² (so the softmax peaks near 4 m at init) — this is a minor optimization detail for Designer to tune.
- **Information loss from bin quantization**: With K=64 log-uniform bins over [1, 15] m, the minimum bin width is ≈ 2.4 cm near z=1 m and ≈ 36 cm near z=15 m. The soft-argmax expectation is not quantized; it can take any value in `[z_min, z_max]`. The resolution of the predicted scalar is limited only by how concentrated the probability is, which is at the training objective's discretion. Mitigation: SORD-style soft labels with σ = 1.5 × bin_width provide sub-bin-width resolution by design.
- **CE loss scale vs existing joint/UV losses**: Cross-entropy over 64 bins has magnitude ~ log(64) ≈ 4.16 at init, while SmoothL1 on joints (at typical errors of ~0.1 m with β=0.05) is ~5 × 0.1 = 0.5 at early training. The existing `loss_weight_depth = 1.0` may need to be decreased. Mitigation: keep `loss_weight_depth = 1.0` but recall it's applied after the CE — we may tune this per Design. Designer can use a 0.25–0.5 depth weight. Alternatively, idea005 (uncertainty weighting) composes naturally: the learned log-variance automatically rescales.
- **Loss / metric semantics preserved**: `pred['pelvis_depth']` is still a `(B, 1)` scalar tensor in metres. `compute_mpjpe_abs` sees no change. `BedlamMPJPEMetric` sees no change. `_compute_mpjpe_abs` is the only consumer of pelvis_depth and it treats it as a scalar — perfect.
- **Bin-range correctness**: If BEDLAM2 pelvis depths occasionally fall outside `[1.0, 15.0]`, they'd be clipped to the boundary bins. Sampling BEDLAM2's train set (from `bedlam2_dataset.py` / `bedlam2_transforms.py`) should show depth distribution. Designer should verify the histogram; tentatively `[1.0, 15.0]` is generous. If clamping is concerning, widen to `[0.5, 30.0]`.
- **MMEngine config constraint**: all new kwargs are str/int/float literals. No imports required.
- **Speed / memory**: `Linear(256, 64)` vs `Linear(256, 1)` is 16× more FLOPs on the output head, but the head is a tiny fraction of total compute (backbone is >99 % of GPU time). Additional memory: (B, 64) logits vs (B, 1) scalar = negligible (~ 1 kB per batch). Softmax + expectation is one GPU op. Total runtime overhead: <0.1 % per step.
- **Composition with idea005 (uncertainty weighting)**: learnable log-variance rescales the CE loss; still valid since cross-entropy is a loss, not a scale-sensitive regression.
- **Composition with idea010 (2D reprojection)**: reprojection uses the absolute pelvis, which depends on `pelvis_depth` scalar. Using the soft-argmax expectation preserves differentiability end-to-end; gradients from the reprojection loss flow through the expectation back into the logits. Clean.
- **Interaction with idea002 (decoupled pelvis query)**: Design C (adaptive bins) uses two heads on the pelvis token — this is exactly the kind of head-fan-out idea002 would compose with (dedicated query → per-task heads). Clean composition.
