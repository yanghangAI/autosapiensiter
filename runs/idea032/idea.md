**Idea Name:** Auxiliary Dense Depth-Map Reconstruction from Spatial Tokens

**Approach:** Add a lightweight linear decoder on the projected spatial tokens that predicts the bilinearly downsampled input depth channel at the H'xW' feature grid, supervised by an L1 reconstruction loss — forcing spatial tokens to retain metric-depth information and thereby improving pelvis depth/UV grounding and absolute-pose accuracy without changing any output head or query/decoder architecture.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### What has been tried on the depth pathway

All prior "depth-aware" ideas operate on the *usage* of depth information, not on whether the spatial tokens actually *retain* it:

| Idea | Mechanism | Position in pipeline |
|---|---|---|
| idea004 (depth-aware spatial PE) | Add depth-derived positional encoding to spatial tokens | Input-side modulation |
| idea014 (depth classification head) | Replace scalar depth regression with binned classification | Output head |
| idea016 (depth-conditional FiLM) | Modulate cross-attention features with predicted depth scale | Mid-decoder conditioning |
| idea018 (depth-gated cross-attention) | Gate attention weights by depth-plane consistency | Cross-attention |
| idea031 (2D UV heatmap) | Replace scalar UV regression with spatial heatmap | Output head |

None of these ideas add an **explicit auxiliary supervision signal on the feature map itself**. The RGBD backbone `sapiens_rgbd` is trained end-to-end only through joint/pelvis/UV losses at the output. Whether the spatial tokens `(B, H'W', hidden_dim) = (B, 960, 256)` actually preserve accurate *per-pixel depth* information is not directly enforced — it is only indirectly encouraged through downstream losses that are heavily dominated by body-joint signal (22 joints × 3 coords × SmoothL1 vs. 1 pelvis depth × SmoothL1).

### The core hypothesis

BEDLAM2 provides a clean per-pixel metric depth channel at input time (it is synthetic data with ground-truth depth). This per-pixel depth is arguably the single most accurate training signal available in the data — far denser than the 22 body joint targets. Yet the baseline throws it away after the backbone consumes it: the spatial tokens are never asked to reconstruct it.

If we force the spatial tokens (via a tiny linear head) to reconstruct the bilinearly downsampled input depth map at the H'×W' feature grid, we inject 960 *per-spatial-cell* supervision signals at every training step. This is:

- **Roughly 50× denser** than the body joint signal (960 spatial cells vs. 22 body joints × 3 = 66 supervision points, if we naively compare count).
- **Directly grounded in metric depth**: the reconstruction target is raw metres, same units as `pelvis_depth`. Improving feature-level depth fidelity directly benefits pelvis depth regression (which reads from a cross-attended spatial feature) and UV grounding (which reads from spatial tokens in idea031's heatmap path).
- **Completely architecturally orthogonal** to every prior idea: it adds one `Linear(hidden_dim, 1)` head and one L1 loss term. The forward path of the main model is unchanged. No decoder, query, attention, or output-head modification.
- **Strongly regularising for the backbone** in a small-data regime. With only 100 training images in stage 1, any source of dense, correct, free supervision is valuable.

### How the reconstruction is implemented

The backbone is fed a 4-channel RGBD tensor. The D channel is already available inside `forward()` of the head indirectly — the Designer must extract it from the data pipeline (either from `data_samples[i].metainfo['depth_map']` or from the preprocessor input). The baseline's data pipeline (`bedlam2_transforms.py`) produces a `(1, H, W) = (1, 640, 384)` depth map per sample in metres. Bilinearly downsampling this to `(1, 40, 24)` gives the per-cell target for the feature grid.

The head predicts a `(B, H'W', 1)` depth per spatial cell:
```python
depth_pred = self.aux_depth_head(spatial)  # Linear(hidden_dim, 1)
depth_pred = depth_pred.view(B, H', W')
```

The loss is:
```python
aux_depth_loss = F.smooth_l1_loss(depth_pred, depth_gt_downsampled, beta=0.1)
losses['loss/aux_depth/train'] = lambda_depth * aux_depth_loss
```

**Masking invalid pixels**: some pixels in BEDLAM2 depth maps are at background/sky ranges (very large values). The loss is masked to pixels with `depth_gt > 0.1 m` and `depth_gt < 30 m` — a valid foreground range. The Designer should verify the valid-depth convention.

**Zero-init**: `self.aux_depth_head.weight.zero_(); self.aux_depth_head.bias.zero_()` ensures the auxiliary head produces 0 at start and develops gradient only from its own loss. The main losses are unaffected at step 0.

### Why this is distinct from idea004, idea016, idea018

- **idea004 (depth-aware PE)**: *adds* depth-derived features to spatial tokens (input-side). This idea *supervises* depth-derived predictions *from* spatial tokens (loss-side). Opposite direction.
- **idea016 (depth-conditional FiLM)**: uses a predicted depth scalar to modulate attention. Depth is an intermediate variable, not a supervision target.
- **idea018 (depth-gated cross-attention)**: uses depth consistency as an attention gate. Depth is used, not supervised.
- **All output-head depth ideas (idea014)**: supervise the pelvis depth *output*, not the feature map's depth fidelity.

No prior idea uses dense per-cell depth as an auxiliary feature-map supervision target.

### Grounding in observed results

- **Pelvis depth-derived error dominates mpjpe_pelvis**: pelvis MPJPE = sqrt( (depth_err)² + (uv_err projected to 3D)² ). With pelvis MPJPE at 608–686 mm across ideas and pelvis depths in the range of ~5–15 m, a 1–2 m depth error translates to 1000–2000 mm pelvis error. The depth component is the dominant contributor. Feature-level depth fidelity is the lever.
- **mpjpe_abs floor at 533 mm (idea008 stage 2)**: absolute pose requires accurate pelvis 3D, which is entirely determined by `(pelvis_depth, pelvis_uv, K)`. Any improvement in feature-level depth representation propagates directly to `mpjpe_abs`.
- **Body MPJPE floor at ~156 mm across stage 2**: the backbone has saturated on body-joint signal. Adding fresh, free, dense supervision (depth reconstruction) injects signal into the backbone via a different channel. idea023 already showed that auxiliary 2D heatmap supervision on spatial tokens was the strongest gain so far — this idea applies the same mechanism (dense per-cell supervision of spatial tokens) but with *metric depth* instead of 2D joint locations.

---

## Proposed Variations

### Design A — Simple L1 depth reconstruction, λ=0.1 (diagnostic)

One `Linear(hidden_dim, 1)` head on the projected spatial tokens. Predicts per-cell metric depth. Loss: `SmoothL1(depth_pred, depth_gt_downsampled, beta=0.1)` with per-pixel foreground mask (`0.1 < depth_gt < 30 m`). Weight λ=0.1. Zero-init the head weights and bias.

Config kwargs: `use_aux_depth=True`, `aux_depth_loss_weight=0.1`, `aux_depth_valid_min=0.1`, `aux_depth_valid_max=30.0`, `feat_h=40`, `feat_w=24`.

Design A is the minimal diagnostic: does adding dense auxiliary feature-level depth supervision improve any metric?

### Design B — Log-depth reconstruction with relative depth scaling, λ=0.3

Same as Design A but the reconstruction target and prediction are in log-space: `log(depth + 1)`. This (a) compresses the dynamic range so far-away pixels don't dominate the gradient, (b) matches the log-uniform depth bin convention of idea014, (c) produces a loss that is roughly proportional to *relative* depth error, which is the perceptually meaningful quantity for 3D pose.

The prediction path:
```python
depth_pred_log = self.aux_depth_head(spatial)           # predicts log(depth+1)
depth_gt_log = torch.log1p(depth_gt_downsampled)
aux_depth_loss = F.smooth_l1_loss(depth_pred_log, depth_gt_log, beta=0.1)
```

Weight λ=0.3 (higher than A because log-space loss magnitudes are smaller, typically by ~5–10×). Same masking.

Config kwargs: `use_aux_depth=True`, `aux_depth_loss_weight=0.3`, `aux_depth_log_space=True`, `aux_depth_valid_min=0.1`, `aux_depth_valid_max=30.0`, `feat_h=40`, `feat_w=24`.

### Design C — Depth reconstruction + gradient consistency, λ=0.3

Design B plus an auxiliary smoothness/consistency term: penalise the predicted depth map's first-order spatial gradient difference against the ground-truth depth map's first-order gradient (captures local depth edges — useful for body outline localisation). This is the standard "edge-preserving" depth loss from monocular depth literature.

```python
dx_pred = depth_pred[:, :, 1:] - depth_pred[:, :, :-1]
dy_pred = depth_pred[:, 1:, :] - depth_pred[:, :-1, :]
dx_gt   = depth_gt[:, :, 1:]   - depth_gt[:, :, :-1]
dy_gt   = depth_gt[:, 1:, :]   - depth_gt[:, :-1, :]
grad_loss = (dx_pred - dx_gt).abs().mean() + (dy_pred - dy_gt).abs().mean()
aux_depth_loss = recon_loss + 0.5 * grad_loss
```

Config kwargs: `use_aux_depth=True`, `aux_depth_loss_weight=0.3`, `aux_depth_log_space=True`, `aux_depth_grad_weight=0.5`, `aux_depth_valid_min=0.1`, `aux_depth_valid_max=30.0`, `feat_h=40`, `feat_w=24`.

---

## Implementation Scope

All changes are confined to `config.py` and `pose3d_transformer_head.py`. One small helper in `pelvis_utils.py` extracts and downsamples the input depth map to the feature grid.

### `pelvis_utils.py`

```python
def downsample_depth_map(depth_map: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
    """Bilinearly downsample (B, 1, H, W) depth map to (B, feat_h, feat_w)."""
    return F.interpolate(depth_map, size=(feat_h, feat_w), mode='bilinear', align_corners=False).squeeze(1)
```

### `pose3d_transformer_head.py`

**`__init__` additions:**
```python
use_aux_depth: bool = False
aux_depth_loss_weight: float = 0.1
aux_depth_log_space: bool = False
aux_depth_grad_weight: float = 0.0
aux_depth_valid_min: float = 0.1
aux_depth_valid_max: float = 30.0
feat_h: int = 40
feat_w: int = 24

if use_aux_depth:
    self.aux_depth_head = nn.Linear(hidden_dim, 1)
    nn.init.zeros_(self.aux_depth_head.weight)
    nn.init.zeros_(self.aux_depth_head.bias)
```

**`forward()` additions** (only when `use_aux_depth=True`):
```python
# spatial: (B, H'W', hidden_dim) already available
aux_depth_pred = self.aux_depth_head(spatial).squeeze(-1)  # (B, H'W')
self._aux_depth_pred = aux_depth_pred.view(B, self.feat_h, self.feat_w)
```

**`loss()` additions** (only when `use_aux_depth=True`):

The Designer must extract the input depth map. Two candidate sources:
1. `data_samples[i].metainfo['depth_map']` (if the transforms pack it).
2. `data_samples[i].gt_instances.depth_map` or similar (pipeline-dependent).

If neither is directly available, the Designer may need a minor `bedlam2_dataset.py` change to attach the depth map to `data_sample.metainfo` — but this file is invariant per Architect rules. A cleaner alternative: the depth channel is already in the RGBD input tensor consumed by the backbone. The head can receive the raw input depth via a small change to the head's `forward()` signature — or, more safely, via `data_samples[i].metainfo['raw_depth_map']` if the preprocessor exposes it.

**If the depth map is not accessible from the head's current interface, the Designer must flag this as an infrastructure issue and the Architect/Orchestrator will decide whether to relax the invariants. Most likely the transform `CropPersonRGBD` already stores the crop's depth map; the Designer should verify first.**

Assuming depth is accessible as `data_samples[i].metainfo['depth_map']` of shape `(1, H, W)`:

```python
depth_maps = torch.stack([ds.metainfo['depth_map'] for ds in data_samples], dim=0).to(spatial.device)
depth_gt = downsample_depth_map(depth_maps, self.feat_h, self.feat_w)  # (B, feat_h, feat_w)

valid = (depth_gt > self.aux_depth_valid_min) & (depth_gt < self.aux_depth_valid_max)
if self.aux_depth_log_space:
    pred = self._aux_depth_pred
    target = torch.log1p(depth_gt)
else:
    pred = self._aux_depth_pred
    target = depth_gt

if valid.any():
    recon_loss = F.smooth_l1_loss(pred[valid], target[valid], beta=0.1)
else:
    recon_loss = pred.sum() * 0.0

if self.aux_depth_grad_weight > 0:
    dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
    dx_gt = target[:, :, 1:] - target[:, :, :-1]
    dy_gt = target[:, 1:, :] - target[:, :-1, :]
    grad_loss = (dx_pred - dx_gt).abs().mean() + (dy_pred - dy_gt).abs().mean()
    recon_loss = recon_loss + self.aux_depth_grad_weight * grad_loss

losses['loss/aux_depth/train'] = self.aux_depth_loss_weight * recon_loss
self._aux_depth_pred = None
```

### `config.py`

Design A, B, C kwargs as listed above. All bool/int/float literals. No Python `import` statements required. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — `mpjpe_pelvis_val` and `mpjpe_abs_val`**: dense per-cell metric-depth supervision forces the spatial tokens to encode geometrically accurate depth, which is exactly what the pelvis depth head reads. Target: `mpjpe_pelvis_val < 600` at stage-1 (vs. best prior 608), `< 300` at stage-2 (vs. best prior 308). `mpjpe_abs_val < 780` at stage-1 (vs. best prior 785), `< 500` at stage-2 (vs. best prior 533).
- **Secondary gain — body MPJPE**: backbone regularisation via dense auxiliary supervision helps representation quality broadly. Mild gain expected. Target: `mpjpe_body_val < 183` at stage-1.
- **Secondary gain — composite_val**: via pelvis gain primarily. Target: `composite_val < 322` at stage-1 (vs. best prior 323.75 from idea023).
- **Design A** (simple L1, λ=0.1): diagnostic — does the idea help at all? Should match or beat baseline in all metrics since auxiliary losses with zero-init can only help.
- **Design B** (log-space, λ=0.3): expected best on pelvis metrics due to relative-depth alignment with the perceptual objective.
- **Design C** (log + gradient): best on body MPJPE if depth edges improve body outline localisation; otherwise similar to B.

---

## Risk and Mitigation

- **Output interface preserved**: no output-head modification. All downstream code (`recover_pelvis_3d`, `bedlam_metric.py`, `metrics_csv_hook.py`) sees zero change.
- **Zero-init**: auxiliary head produces 0 at step 0. Training starts identical to baseline; gradient only from the auxiliary loss develops over time.
- **Parameter count delta**: `Linear(256, 1) = 257` new params per head. Negligible.
- **Memory**: one additional `(B, 960)` prediction tensor + one `(B, 40, 24)` GT tensor ≈ 20 KB under FP16. Negligible.
- **Speed**: one `Linear(256, 1)` + one `F.interpolate` per step. <0.5 ms per batch on 2080 Ti. Negligible.
- **Depth-map access infrastructure concern**: the head's `loss()` currently receives `data_samples` (a list of `PoseDataSample`). Whether the depth map is carried in metainfo depends on `bedlam2_transforms.py` (an invariant). The Designer MUST first inspect the current `data_samples[i].metainfo` contents in the baseline. If `depth_map` (or equivalent raw input depth) is NOT present, the Designer MUST flag it to the Orchestrator rather than edit the invariant transforms. A fallback is to route the raw depth through the data preprocessor's `_stacked_depth` attribute or via a new custom hook that attaches depth to metainfo without modifying `bedlam2_transforms.py` — the Designer can propose this.
- **Masking**: the valid-range mask (`0.1 < d < 30 m`) avoids ill-defined gradients on background pixels. BEDLAM2 is synthetic; background can have very large depth values. The Designer MUST verify the valid-depth convention from a few samples.
- **Log-space safety** (Design B/C): `log1p(depth)` is safe for all `depth >= 0`. Predictions in log-space can be arbitrary real numbers; `smooth_l1` is unbounded in either direction. No numerical issues.
- **AMP/FP16 safety**: linear + smooth_l1 + finite interpolation all AMP-friendly. The `F.interpolate` runs in FP16 under AMP without issue.
- **Gradient balance**: λ=0.1 (Design A) is small enough that the auxiliary loss adds a bounded perturbation to the total loss. The joint/pelvis losses have gradient magnitudes of O(1–10); the aux depth loss will be O(0.1–1) after λ. No gradient-scale pathology expected.
- **Interaction with idea004 (depth PE)**: orthogonal. Depth PE adds depth features at the input; this idea supervises depth predictions at the output.
- **Interaction with idea014 (depth classification head)**: composable. idea014 changes the pelvis depth head's output representation; this idea supervises the spatial features that feed into any depth-related head.
- **Interaction with idea018 (depth-gated cross-attn)**: composable. idea018 uses depth at attention time; this idea supervises depth at feature time.
- **Interaction with idea031 (UV heatmap)**: highly composable and synergistic. idea031 requires spatial tokens to encode "where is the pelvis in the image"; this idea requires spatial tokens to encode "how far is each pixel". Both supervision signals land on the same spatial tokens through separate linear heads. A future combined idea is recommended.
- **MMEngine config constraint**: all new kwargs are literals. Fully compliant.
