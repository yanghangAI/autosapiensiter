**Idea Name:** Metric 3D Positional Encoding via Depth Unprojection for Spatial Tokens

**Approach:** For each spatial token at grid cell (h, w), unproject the downsampled input depth value at that cell through the per-sample camera intrinsics K into a metric camera-frame 3D coordinate (X, Y, Z) in metres, then embed that 3D coordinate via a small MLP (and/or sinusoidal basis) and add it to the spatial-token positional encoding — giving the decoder a metric, K-correct, per-token 3D geometry signal that the baseline's 2D sinusoidal PE (and even idea004's scalar-depth PE) cannot express.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### What's missing from current spatial-token conditioning

The baseline builds spatial tokens as `input_proj(backbone_feat) + PE_2D(h, w)`. The PE is purely a function of grid index — it tells a cross-attending joint query *where in the feature grid* a token sits, but carries no metric geometry. Two prior ideas are adjacent but not equivalent:

- **idea004 (Depth-Aware Spatial PE)**: adds a learned projection of the *raw scalar depth* value at each grid cell to the PE. This exposes per-pixel depth to the decoder, but it is a **1D signal in sensor units** — it does not know (a) the principal point, (b) the focal length, or (c) how a pixel offset translates to metric X, Y offset. A joint query learning to locate a wrist cannot compute "this token is 1.3 m to the left and 0.8 m in front" — it can only see "this token has depth value 4.7 m".
- **idea033 (Camera-Intrinsic FiLM)**: injects K **globally** via FiLM on queries or pelvis token. It is a single 6-D vector per sample modulating everything uniformly — no per-token geometric structure. A query still cannot read different 3D coords for different spatial tokens; it can only rescale a global intrinsic-aware channel statistic.

**The unexplored axis is combining the two**: per-token depth (a dense per-location signal) **and** per-sample K (a global geometric calibration), fused at the spatial-token level into a metric 3D coordinate `p_{h,w} = unproject(u_{h,w}, v_{h,w}, d_{h,w}, K)` in camera-frame metres. This is the **actual geometric content** of what a joint query would need to regress; providing it as an explicit PE addend converts an implicit inverse-projection subproblem into a lookup.

### Why this targets the pelvis/abs-MPJPE plateau

Across 33 prior ideas:
- `mpjpe_pelvis_val` (stage-1) best = 608.64 (idea023/design001), baseline 652.89 — only −6.8% moved.
- `mpjpe_abs_val` (stage-1) best = 747.25 (idea008/design002), baseline 833.75 — −10.4%.

Pelvis depth/UV is the subproblem where the head must produce **metric** output. The current spatial tokens provide appearance + 2D PE; producing metric world coords from that requires the head to implicitly invert the camera model from training statistics. When K varies sample-to-sample (BEDLAM2 crops = sample-specific K), this implicit inversion is averaged rather than sample-correct. Injecting explicit metric 3D positions into the keys/values collapses the inverse-projection step into the PE, freeing head capacity for the appearance-to-joint mapping itself.

Also complementary to body MPJPE: idea023 (heatmap pooling) showed that giving each body query a joint-specific spatial summary helps significantly. If the spatial tokens themselves carry metric 3D PE, the pooled summary is *already in metric coordinates* — the body-joint regression head no longer has to reconstruct (X, Y, Z) from a 2D-indexed appearance pool.

### Why this is distinct from all prior ideas

| Idea | Per-token depth? | Uses K? | Metric 3D per-token? |
|---|---|---|---|
| idea004 | Yes (scalar) | No | No |
| idea016 | No (FiLM from pooled tokens) | No | No |
| idea018 | Uses predicted relative depth gating | No | No |
| idea022 | Uses predicted 3D joints as bias at later layers | Yes (for reprojection) | No (attention bias, not PE) |
| idea027 | Conv over spatial tokens | No | No |
| idea032 | Depth-map reconstruction (aux loss) | No | No |
| idea033 | K FiLM (global) | Yes (global) | No |
| **idea034 (this)** | **Yes (scalar)** | **Yes (per-token)** | **Yes** |

No prior idea feeds a **per-token camera-frame metric 3D coordinate** into the decoder. It is simultaneously K-aware (unlike idea004), per-token (unlike idea033), and ground-truth geometry at inference time (unlike idea022 which uses predicted 3D).

### Data availability — why this is cheap

Every signal required is already in-batch:
- The raw depth channel of the RGBD input is accessible (idea004 already established a routing path for the downsampled depth map to the head).
- K and `img_shape` are already in `batch_data_samples[i].metainfo` (used by `pelvis_utils.recover_pelvis_3d` and in idea010/idea023/idea033).
- The unprojection arithmetic is the same used in `recover_pelvis_3d`, just applied to the spatial grid instead of a single pelvis UV.

No new dataset fields, no new transforms, no new upstream changes.

## Mechanism Sketch (details for Designer)

At each forward pass, after the backbone produces `feat ∈ R^{B×C×H'×W'}`:

1. Obtain a grid-aligned depth map `D ∈ R^{B×H'×W'}` by bilinear-downsampling the raw input depth channel (same path used in idea004; interpolate to (H', W')).
2. Build per-cell pixel coordinates `(u, v)` on the original crop: `u = (w + 0.5) * crop_w / W'`, `v = (h + 0.5) * crop_h / H'`. Note this uses the per-sample `crop_w`, `crop_h` from `img_shape`.
3. Unproject to camera-frame metres using the same convention as `recover_pelvis_3d`. For BEDLAM2 the convention in `pelvis_utils.py` is `X = d`, `Y = -(u - cx) * X / fx`, `Z = -(v - cy) * X / fy` (Designer MUST verify sign conventions against `pelvis_utils.recover_pelvis_3d` before implementation — they are the load-bearing correctness test). Output: `P ∈ R^{B×H'W'×3}` in metres.
4. Embed `P` into `hidden_dim` via either (a) a 2-layer MLP `(3 → hidden_dim → hidden_dim)` with `GELU`, or (b) a fixed sinusoidal 3D basis followed by a linear projection. Initialize the final linear to **zero** so `PE_3D ≡ 0` at step 0 and baseline is recovered exactly.
5. Add `PE_3D` to the existing 2D sinusoidal PE on the spatial tokens: `spatial = input_proj(feat) + PE_2D + PE_3D`. No other module changes.

Three natural variants for the Designer:

- **Variant A — MLP-embedded metric 3D PE, additive.** 2-layer MLP on `(X, Y, Z)` metres (optionally log-scaled on `X` for scale-invariance), added to PE_2D. Simplest baseline.
- **Variant B — Sinusoidal 3D PE + linear.** Use per-axis sinusoidal bases at several scales (e.g. σ ∈ {0.25m, 1m, 4m, 16m}), concatenate, linear project to hidden_dim (zero-init), add to PE_2D. More expressive for broad depth range.
- **Variant C — Metric 3D PE injected into keys only (not queries / not values).** Apply the 3D PE addend only when the spatial tokens are used as cross-attention *keys*, leaving the values as pure appearance. This decouples routing (geometry-based) from feature aggregation (appearance-based), mirroring the principle behind idea021 (cross-attn spatial bias) but using metric geometry rather than a learned bias.

In all variants: invalid / missing depth (e.g. if the depth channel has inpainted zeros or NaN sentinels) is handled by clamping depth to `[d_min, d_max]` (e.g. `[0.1, 50.0]`) — same clamp used in `recover_pelvis_3d`.

## Scope & Invariants

- Changes confined to `pose3d_transformer_head.py` (PE_3D module, forward-path unprojection, route K/img_shape into forward) and `config.py` (bool/str/float literals for variant, MLP width, depth clamp).
- `pelvis_utils.py`: may add a small helper `unproject_grid_to_metric_3d(D, K, crop_h, crop_w, feat_h, feat_w) -> (B, H'W', 3)` that mirrors `recover_pelvis_3d`'s convention.  This is the only allowed helper addition; `recover_pelvis_3d` itself stays unchanged.
- `bedlam_metric.py`, `bedlam2_dataset.py`, backbone unchanged. Input depth is already part of the RGBD input tensor; downsampling is a forward-time op.
- Output shapes and loss signatures unchanged.
- Zero-init of final linear → exact baseline at step 0. Standard safe-init pattern (ideas 003/011/013/021/023/033).
- MMEngine config constraint: all kwargs are bool/int/float/str literals. No Python imports.
- AMP / fp16 safety: unprojection arithmetic does not overflow for clamped depth; the MLP input is in metres (typically < 50), well within fp16 range.
- Threading K + depth into `forward()`: `forward()` currently accepts only `feats`. Designer should extend it to optionally accept `batch_data_samples` (or pre-extracted tensors `K_tensor`, `depth_map`, `img_shape_tensor`) — matches the pattern used by idea033 and idea023 for threading metadata into the head.

## Composition with Prior Ideas

Orthogonal to every prior idea:
- **idea004** (scalar depth PE): subsumed and strictly generalized — if this idea works, idea004 is a strict subset.
- **idea023** (heatmap query pooling): composable — metric 3D spatial tokens make the pooled per-joint summary metric-3D, strengthening idea023's warm-start.
- **idea033** (K FiLM global): complementary — that idea gives the head a global K vector, this idea gives it per-token metric 3D. If both help, composing them is natural.
- **idea008 / idea028** (body-focused / decoupled pelvis): fully compatible — metric PE is a spatial-token-level change and does not touch queries.
- **idea014** (binned pelvis depth): complementary — metric PE makes the spatial tokens' depth channel explicit, helping the bin-classification head read the correct region.

## Success Targets

- Stage-1 `composite_val` < 325, matching or beating idea013/design003 (328.14) and idea023/design001 (323.75).
- Stage-1 `mpjpe_pelvis_val` < 600 (breaking the 608.64 floor set by idea023/design001).
- Stage-1 `mpjpe_abs_val` < 780 (breaking the 747.25 floor set by idea008/design002 is a stretch goal; 780 is the primary target).
- Stage-2 `composite_val` < 215 (beating idea023/design001's 215.43 is the primary benchmark).
- Primary diagnostic: whether `mpjpe_pelvis_val` and `mpjpe_abs_val` drop disproportionately relative to `mpjpe_body_val`. If yes, the metric-3D-PE mechanism is working as theorized.

## Risk and Mitigation

- **Depth channel quality**: if the input depth map contains NaN/Inf/zero-hole regions from BEDLAM2 rendering, clamp to `[d_min, d_max]` in the unproject helper. The backbone already tolerates the raw depth (it consumes it as channel 4), so in-pipeline clamping at PE-compute time is local and safe.
- **K convention mismatch**: sign/axis conventions for unprojection must match `pelvis_utils.recover_pelvis_3d` exactly. Designer writes a 3-line unit check (pick a single pixel, unproject, reproject through K, verify u/v match to ≤1 px) before committing the forward path.
- **Memory**: extra buffer is `(B, H'W', 3) = (4, 960, 3)` = ~46 KB, plus the MLP activations `(4, 960, 256)` ≈ 4 MB. Negligible on 2080 Ti.
- **Speed**: per-step overhead is dominated by the MLP `(4*960, 256)` ≈ 1 ms. Bilinear depth downsample < 0.1 ms. Net overhead < 2 ms, much smaller than a decoder layer.
- **Over-reliance on depth at inference**: at inference the input depth is also available (RGBD input), so there is no train/test mismatch — this is different from methods that depend on GT depth only at train time.
- **Interaction with fp16 AMP**: `F.interpolate` for depth downsampling under AMP may emit fp16 nan for large-magnitude sentinel values; Designer should cast to fp32 for the unprojection math and cast back.
