**Idea Name:** Depth Modality Ablation — Is the Depth Input Actually Needed?
**Approach:** Corrupt the depth channel at the data-preprocessor stage in three controlled ways (zero, Gaussian noise, spatial shuffle) to measure the actual contribution of the depth modality to BEDLAM2 3D pose accuracy.
**Expected Designs:** 3
**Baseline Source:** /work/pi_nwycoff_umass_edu/hang/autosapiens_iter/baseline

## Motivation

All 34 prior ideas assume the depth input is informative; six (idea004, idea014, idea016, idea018, idea032, idea034) explicitly try to *exploit* depth more aggressively. None has tested the more basic question: **does the depth channel actually help on BEDLAM2 at all, and if so, how (as a scale prior, or as a per-pixel spatial signal)?**

If the answer is "depth contributes little," then the entire family of depth-conditioning ideas is mis-targeted, and architectural effort should move to RGB-only mechanisms. If depth dominates, that justifies the depth-targeted directions and explains why pelvis MPJPE has plateaued (depth is doing most of the pelvis localization, leaving little headroom for changes that don't touch it).

This is a **scientific ablation**, not a leaderboard attempt. Its primary product is the *gap* between the corrupted-depth runs and baseline, broken down by metric (especially `mpjpe_pelvis_val` and `mpjpe_abs_val`, which depend most directly on absolute scale).

## Hypotheses

1. **H1 (depth helps absolute localization)** — Pelvis and absolute MPJPE will degrade significantly when depth is corrupted; relative body MPJPE will degrade less. This is the expected baseline outcome under the "depth = scale prior" interpretation.
2. **H2 (spatial alignment matters)** — Design C (shuffled depth, preserves marginal statistics but destroys per-pixel RGB↔depth correspondence) should sit *between* baseline and zero-depth if alignment matters; *equal to* zero-depth if only scale matters; *equal to* baseline if depth is being ignored.
3. **H3 (depth pathway is underused)** — If all three corruption modes give nearly baseline numbers, the backbone is essentially RGB-only in practice and depth-targeted ideas are unlikely to pay off.

## Mechanism (allowed within invariants)

The depth signal enters via `LoadBedlamLabels` → `CropPersonRGBD` → `RGBDPoseDataPreprocessor` → `SapiensBackboneRGBD`. Transforms, the preprocessor file, and the backbone are invariant. However, **adding new classes to `pose3d_transformer_head.py` is allowed** (it is in the experimentable file list and is already registered via `custom_imports`). We register a thin `DepthAblationDataPreprocessor(RGBDPoseDataPreprocessor)` subclass there that:

1. Calls `super().forward(...)` to get the standard preprocessed 4-channel tensor `(B, 4, H, W)` with normalization applied.
2. Replaces channel index 3 (depth) according to `mode`:
   - **zero**: fill with `0.0` (post-normalization mean ≈ 0, so this is "no information").
   - **gauss**: replace with `torch.randn_like(...)` (preserves variance scale, destroys all signal).
   - **shuffle**: take the existing depth channel and apply a per-sample random permutation of pixels (preserves the exact marginal histogram of depth values, destroys all spatial correspondence with RGB).
3. Returns the modified tensor; everything downstream is unchanged.

`config.py` switches `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode=<...>)`. The `mode` is a string literal — config-friendly.

This keeps the input shape, the backbone, the transforms, the dataset, the loss, and the head architecturally identical to baseline. The only difference is the content of one channel.

## Designs (3)

- **Design A — Zero Depth.** `mode='zero'`. Tests pure information removal. Baseline-equivalent computation; backbone simply learns to ignore a constant channel (or fails to compensate if depth was load-bearing).
- **Design B — Gaussian Noise Depth.** `mode='gauss'`. Tests robustness: depth has the right scale but no signal. Distinguishes "backbone uses depth statistics" from "backbone uses depth content." If A and B match closely, the depth conv has effectively been suppressed.
- **Design C — Spatially-Shuffled Depth.** `mode='shuffle'`. Per-sample pixel permutation seeded by sample index for determinism within a forward. Marginal depth-value distribution is exactly preserved — only the *spatial alignment with RGB* is destroyed. The gap between B and C isolates "depth as image-aligned spatial signal" from "depth as bulk statistic."

## Files Touched

- `pose3d_transformer_head.py`: add ~15 lines defining `DepthAblationDataPreprocessor` (subclass, `@MODELS.register_module()`, three `mode` branches).
- `config.py`: change `model.data_preprocessor` dict to use `DepthAblationDataPreprocessor` with the design-specific `mode`. No other edits.
- `pelvis_utils.py`: untouched.

## Expected Outcomes / Success Criteria

This is an ablation, so the deliverable is the comparison table itself, not a target composite. Decision rules for what we learn:

| Observation | Interpretation | Action for future ideas |
|---|---|---|
| All three designs ≈ baseline (within ~5 mm composite) | Depth is being ignored by the trained backbone | De-prioritize all depth-conditioning ideas; focus on RGB-only mechanisms |
| A ≈ B ≪ C ≈ baseline | Only spatial-alignment matters; depth statistics are not used | Prioritize per-pixel depth-fusion ideas (idea018, idea034); skip statistics-based ones |
| A ≈ B ≈ C ≪ baseline | Depth is genuinely load-bearing in both ways | Validates the entire depth-targeted family; doubles down on idea014/032 |
| C between baseline and A/B | Both signal and alignment matter, alignment more so | Prioritize alignment-preserving fusion (idea034) over scale-priors |

## Stage 2 gating

Stage 2 is gated on stage-1 `composite_val` *beating baseline*. For an ablation **we expect the corrupted runs to be no better than baseline** — so stage 2 will not auto-fire, which is correct: spending stage-2 compute on an ablation is wasteful. Stage-1 numbers alone answer the scientific question.

## Composability

Orthogonal to all 34 prior ideas. Particularly informative if combined later with idea008 (body-focused decoder) or idea002 (decoupled pelvis), because those decouple the body and pelvis pathways in ways that may have different sensitivities to depth. Not run as part of this idea — kept as a clean single-axis ablation.

## Risk / Caveats

- The Sapiens 0.3B backbone was pretrained on RGB-only ImageNet-style data; the depth conv at the stem is a small from-scratch addition. It is plausible that 20 stage-1 epochs is not long enough for the depth pathway to be deeply integrated, which would bias the ablation toward "depth doesn't matter much." This caveat must be reported alongside the result.
- BEDLAM2 is synthetic with high-quality ground-truth depth; real-world depth (noisier, with missing values) might give different ablation conclusions. Out of scope here.
