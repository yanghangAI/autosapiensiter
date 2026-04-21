**Idea Name:** Depth-Aware Spatial Positional Encoding

**Approach:** Augment the 2D sinusoidal spatial token positional encoding with a learned projection of the raw depth map values, giving each spatial token an explicit per-location depth signal so that cross-attention can reason about absolute scale and depth geometry rather than relying solely on what the backbone implicitly encodes.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder architecture produces spatial tokens by projecting backbone features and adding a fixed 2D sinusoidal positional encoding. This encoding tells each spatial token *where* it is in the 2D feature grid (x, y), but carries no information about *depth* at that location. The backbone (SapiensBackboneRGBD) processes a 4-channel RGBD input and implicitly encodes depth in its feature maps, but there is no direct geometric depth signal injected at the token level.

This matters because:

1. **Cross-attention geometry**: when joint queries cross-attend to spatial tokens, the attention scores are computed from query-key dot products. A joint query learning to find a forearm joint needs to determine not just (x, y) location but also the local depth context to correctly estimate the 3D position. Without explicit depth in the positional signal, the network must reconstruct this from the backbone features alone — a harder implicit learning problem.

2. **Pelvis depth prediction**: the pelvis depth head reads from joint token 0, which must aggregate depth-relevant information during cross-attention. Spatial tokens that carry their own depth value make it much easier for any joint query to find depth-relevant tokens rather than having to infer it from appearance features.

3. **Evidence from prior ideas**: idea001 showed that adding more decoder capacity did not help — suggesting the bottleneck is not the number of processing steps but the *quality of information* in the spatial tokens. idea003 addressed query-side information via content-adaptive initialization. The natural complement is improving the spatial token side — specifically its geometric/depth content.

### What this idea adds

At each forward pass, the raw depth map (already available as channel 4 of the RGBD input, passed through backbone preprocessing) is downsampled to the feature grid resolution (H', W'), flattened to (B, H'*W', 1), projected through a small MLP to `hidden_dim`, and added to the spatial tokens *alongside* the 2D positional encoding. This is a strict superset of the baseline: at initialization the depth projection MLP outputs near-zero, so the model starts from the same effective state as the baseline.

The change is confined to `pose3d_transformer_head.py`. The depth map must be threaded through from `feats` — the backbone's `SapiensBackboneRGBD` already stores the depth input accessible via `feats`. We route the raw depth map as an extra element in the feature tuple (or via a side channel), or alternatively reconstruct it from the batch_data_samples in the head's forward pass.

**Implementation path**: the cleanest approach is to pass the depth map as an additional tensor alongside the backbone features. In `pose3d_transformer_head.py`, `feats[-1]` is currently `(B, C, H', W')`. We can instead pass `feats` as a dict or rely on a convention where `feats[-2]` holds the raw depth at the original resolution and downsample it in the head. Alternatively, we interpolate the depth map to match H'×W' using bilinear interpolation in the forward method, reading the depth from `batch_data_samples` (which contains `img_shape` and the depth path via metainfo). 

The simplest and most self-contained approach: add the depth map as an additional backbone output channel. However, since the backbone is invariant, the cleanest option is to pass the raw RGBD input's depth channel through the feature tuple — checking whether the backbone already exposes `feats` with more than one element.

Actually, the cleanest implementation that keeps changes isolated to `pose3d_transformer_head.py`: accept an optional `depth_map` argument to `forward()` and compute the positional encoding. Since `loss()` and `predict()` both call `self.forward(feats)` and the model assembler calls these with `feats` from the backbone, the depth map can be passed through the `feats` tuple by convention (e.g., `feats = (feature_map, depth_map_resized)`). Since this requires a thin change to how the estimator calls the head, the Designer may also need to subclass `RGBDPose3dEstimator` — which is permissible as a custom module in `config.py`. Alternatively, the depth map can be loaded directly in the head's forward pass from the batch data samples' stored `depth_npy_path`, but that would create a data-loading dependency in the head.

The most implementation-clean approach within the allowed files: **read the depth map from the feature map itself**. The backbone `SapiensBackboneRGBD` processes a 4-channel (RGBD) input; the last feature map `feats[-1]` is `(B, embed_dim, H', W')`. The raw depth at the crop level is not directly available here. However, we can instead downsample the depth channel from the preprocessed input. Since the head's `forward` receives only `feats` (not the raw input), the Designer should consider passing the depth map as `feats[0]` (a raw depth map of shape `(B, 1, H, W)`) alongside `feats[1]` (the backbone feature map `(B, C, H', W')`), with the backbone's feature tuple reordered or augmented. This is a one-line change in `sapiens_rgbd.py` — but that file is in the invariant set.

**Revised clean approach**: store the downsampled depth as a buffer or compute it via bilinear interpolation from the data sample's depth field. The `loss()` method receives `batch_data_samples` and `feats`. We can pool the depth from there. Specifically in `loss()` and `predict()`, load the depth from the batch samples, resize to feature map resolution, and pass it into `forward()` as an extra keyword argument. Since `forward()` in MMPose heads is called by the model assembler (not directly by the user), and the loss/predict wrapper functions already have access to `batch_data_samples`, the head can pass the depth to its own `forward()` — the Designer may need to refactor the internal call pattern (have `loss()` compute the spatial tokens itself, or add an internal `_build_spatial_tokens(feats, depth)` method).

This approach keeps the change fully within `pose3d_transformer_head.py`. The Designer should choose whichever internal refactor is cleanest.

---

## Proposed Variations

**Design A — Scalar depth per spatial token (lightweight)**

Project the raw depth value at each spatial location to `hidden_dim` using a single `nn.Linear(1, hidden_dim)`. Add this depth-derived vector to the spatial tokens alongside the 2D sinusoidal positional encoding:

```
spatial_tokens = input_proj(feat) + 2d_sincos_pos_enc + depth_proj(depth_grid)
```

where `depth_grid` is the bilinearly-downsampled raw depth map at H'×W' resolution, shaped `(B, H'*W', 1)`. This is the minimal-change design: one extra linear layer and one extra addition. Tests whether any explicit depth signal at the token level is useful.

**Design B — Depth sinusoidal encoding (geometric)**

Instead of a learned linear, apply a sinusoidal encoding to the scalar depth value at each token position — analogous to how the 2D positional encoding encodes x and y using sin/cos at multiple frequencies. Build a 1D sinusoidal encoding of the depth value and concatenate (or add) it to the 2D sinusoidal positional encoding:

```
depth_sine = build_1d_sincos_enc(depth_grid, hidden_dim // 2)   # (B, H'*W', hidden_dim//2)
spatial_tokens = input_proj(feat) + concat_and_project([2d_sincos_pos_enc, depth_sine])
```

A small `nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)` projects the concatenated positional signal back to `hidden_dim`. This is parameter-efficient (~0.4 M params) and encodes depth in a frequency decomposition that generalises better to unseen depth ranges.

**Design C — Depth MLP spatial conditioning (richest)**

Use a 2-layer MLP that takes `(x_pos, y_pos, depth)` as a 3-element input (normalised to [−1, 1] and [0, 1] respectively) and outputs a `hidden_dim`-dimensional positional embedding per token:

```
pos_input = cat([norm_x, norm_y, norm_depth], dim=-1)   # (B, H'*W', 3)
spatial_tokens = input_proj(feat) + pos_mlp(pos_input)   # pos_mlp: 3 → 64 → hidden_dim
```

This replaces the fixed 2D sinusoidal positional encoding entirely with a learned positional MLP (Nerf-style). The MLP can compose depth and 2D position jointly, potentially learning that depth affects position perception differently at different image regions (e.g., corners vs. center). Parameter cost: 3×64 + 64×256 ≈ 16 K params — negligible.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py`:

1. **Access to depth map**: refactor `loss()` and `predict()` to extract the bilinearly-downsampled depth map from `batch_data_samples` (which contains `depth_npy_path` and metainfo) and pass it to `forward()` as a keyword argument `depth_map: Optional[torch.Tensor]`. When `depth_map` is None (e.g., test with no depth), fall back to zero padding.

2. **Depth positional signal**:
   - Design A: add `self.depth_proj = nn.Linear(1, hidden_dim)` in `__init__`. In `_build_spatial_tokens`, compute `spatial = spatial + self.depth_proj(depth_grid)`.
   - Design B: add `build_1d_sincos_enc` helper and `self.depth_pos_proj = nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)`.
   - Design C: replace `_build_2d_sincos_pos_enc` with `self.pos_mlp = nn.Sequential(Linear(3,64), GELU(), Linear(64, hidden_dim))` and remove fixed sinusoidal buffer.

3. **`config.py`**: expose `depth_pos_enc_type: 'linear' | 'sinusoidal' | 'mlp'` as a head kwarg.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, invariant files, or data pipeline.

---

## Expected Outcome

- **Primary gain**: improved both body MPJPE and pelvis MPJPE because cross-attention now has explicit depth context at each spatial token, enabling more precise geometric localization.
- **Pelvis depth**: expected meaningful improvement (~−10 to −20 mm) because the depth projection directly encodes the absolute depth at each spatial location, making it easier for joint token 0 (or a dedicated pelvis query, if combined with idea002) to read off the correct absolute depth.
- **Body MPJPE**: moderate improvement (~−5 to −15 mm) from better per-token positional grounding.
- **Composite target**: aim for composite_val < 163 (vs. baseline 176.4 and idea001 best 170.4).
- **Design C** is the primary bet; **Design A** is the cheap ablation; **Design B** provides a middle ground with geometric inductive bias.

---

## Risk and Mitigation

- **Depth map access in forward()**: the head does not natively receive raw data beyond `feats`. The Designer must implement depth extraction from `batch_data_samples` inside `loss()`/`predict()`. This is a slightly unusual pattern but fully contained within `pose3d_transformer_head.py`. The Designer should validate that `depth_npy_path` is accessible in metainfo and that loading+resizing at training time does not bottleneck throughput (it is a single small array per sample).
- **Missing/zero depth at inference**: if depth is unavailable, the model should degrade gracefully to baseline behaviour. Zero-padding with learned projection will produce a near-zero additive signal (by initialization), so the fallback is well-behaved.
- **Memory**: no extra attention operations. Depth positional encoding is a simple addition to spatial tokens. Negligible memory overhead (<1 MB).
- **Training speed**: loading depth NPZ at the positional encoding step adds minor overhead, but the pipeline already loads depth NPZ during data loading (via `LoadBedlamLabels`). The Designer should store the depth map in `batch_data_samples` at load time rather than re-loading in the head forward pass.
- **Interaction with idea002/003**: depth-aware spatial encoding is orthogonal to pelvis query decoupling (idea002) and query conditioning (idea003). If combined, each idea targets a different component and they stack without conflict.
