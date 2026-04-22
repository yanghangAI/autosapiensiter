# Design 001 — Variant A: MLP-embedded Metric 3D PE (additive to spatial tokens, keys+values)

**Design Description:** For each spatial token at grid cell `(h, w)`, unproject the bilinearly-downsampled input depth value `d_{h,w}` through the per-sample crop intrinsics `K` into a camera-frame metric 3D coordinate `(X, Y, Z)` in metres, embed the 3-vector through a 2-layer MLP `(3 → hidden_dim → hidden_dim)` with GELU, and add the result (zero-initialised final Linear) to the existing 2D sinusoidal PE on spatial tokens. Same signal is seen by cross-attention as both keys and values. At step 0, the final Linear is zero so `PE_3D ≡ 0` and the head is bit-for-bit identical to baseline.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `_Metric3DPE` MLP module, `_extract_depth_map` helper (same as idea004), `_build_K_batch` helper, route K and depth through `loss()`/`predict()` into `forward()`, add PE_3D term to the spatial tokens after PE_2D.
2. `pelvis_utils.py` — add one new helper `unproject_grid_to_metric_3d(D, K_batch, crop_hw, feat_h, feat_w) -> (B, H'W', 3)` that mirrors the sign convention of `recover_pelvis_3d`. `recover_pelvis_3d` and `compute_mpjpe_abs` stay unchanged.
3. `config.py` — add five new kwargs to the `head=dict(...)` block: `use_metric_pe_3d=True`, `metric_pe_variant='mlp_additive'`, `metric_pe_mlp_hidden=256`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0`.

All invariant files (`bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `rgbd_pose3d.py`, `train.py`, `tools/train.py`, `infra/*`) are untouched.

---

## Algorithm

### 1. Depth map extraction — reuse idea004 pattern

At the start of `loss()` and `predict()`, extract a per-sample depth map aligned to the backbone feature grid `(feat_h, feat_w)`:

- For each `ds ∈ batch_data_samples`, read `ds.metainfo['depth_npy_path']` and `ds.metainfo['img_shape']` (= `(crop_h, crop_w)`).
- `np.load(path)` → if `NpzFile`, pick key `'depth'` if present else `list(raw.keys())[0]`; if `ndim==3` take `raw[0]`.
- Crop the raw depth to `[:crop_h, :crop_w]` (top-left aligned, identical to idea004).
- If any step fails, fallback to a zeros array of shape `(crop_h, crop_w)` (graceful degradation).
- `torch.from_numpy(depth.astype(np.float32))` → `unsqueeze(0).unsqueeze(0)` → `F.interpolate(..., size=(feat_h, feat_w), mode='bilinear', align_corners=False)`.
- Stack per-sample tensors into `(B, 1, feat_h, feat_w)` on `feats[-1].device`.

This logic goes into a new instance method `_extract_depth_map(self, batch_data_samples, target_h, target_w, device) -> Tensor(B,1,H',W')`. It is literal-equivalent to the idea004 helper and should be written with identical code (reuse for determinism).

### 2. K extraction helper

Add a second helper `_build_K_batch(self, batch_data_samples, device) -> Tuple[Tensor, Tensor]` that returns:

- `K_batch` of shape `(B, 3, 3)` in float32 on `device`, built as:
  ```python
  K_np = np.asarray(ds.metainfo['K'], dtype=np.float32)  # (3,3)
  ```
  stacked over samples.
- `crop_hw_batch` of shape `(B, 2)` in float32, where each row is `[crop_h, crop_w] = img_shape[0], img_shape[1]`. `img_shape` defaults to `(640, 384)` if missing (matches baseline `CropPersonRGBD(out_h=640, out_w=384)`).

### 3. Grid unprojection — new function in `pelvis_utils.py`

Add to `pelvis_utils.py`:

```python
def unproject_grid_to_metric_3d(
    depth_grid: torch.Tensor,   # (B, 1, H', W') float32
    K_batch: torch.Tensor,      # (B, 3, 3) float32
    crop_hw: torch.Tensor,      # (B, 2) float32 — (crop_h, crop_w) per sample
    feat_h: int,
    feat_w: int,
    d_min: float = 0.1,
    d_max: float = 50.0,
) -> torch.Tensor:
    """Unproject a feature-grid depth map to camera-frame metric 3D.

    BEDLAM2 convention (same as ``recover_pelvis_3d``):
        X = d  (forward distance in metres)
        Y = -(u_px - cx) * X / fx
        Z = -(v_px - cy) * X / fy

    Pixel centres on the crop are computed as::
        u_px = (w + 0.5) * crop_w / W'
        v_px = (h + 0.5) * crop_h / H'

    Args:
        depth_grid: (B, 1, H', W') depth in metres, already resized to the
            feature grid.
        K_batch: (B, 3, 3) per-sample crop intrinsics.
        crop_hw: (B, 2) per-sample (crop_h, crop_w) in pixels.
        feat_h, feat_w: feature map spatial dims.
        d_min, d_max: soft clamp bounds applied to depth before
            unprojection.

    Returns:
        (B, H'*W', 3) metric XYZ tensor in metres on the same device/dtype
        as ``depth_grid``.
    """
```

Implementation (fp32 internally; cast back to input dtype at the end):

```python
import torch

def unproject_grid_to_metric_3d(depth_grid, K_batch, crop_hw, feat_h, feat_w,
                                d_min=0.1, d_max=50.0):
    B = depth_grid.shape[0]
    device = depth_grid.device
    out_dtype = depth_grid.dtype

    # Compute pixel centre grid in fp32
    w_idx = torch.arange(feat_w, dtype=torch.float32, device=device)
    h_idx = torch.arange(feat_h, dtype=torch.float32, device=device)
    grid_v, grid_u = torch.meshgrid(h_idx, w_idx, indexing='ij')  # (H', W')

    # Per-sample pixel coords on the crop: (B, H', W')
    crop_h = crop_hw[:, 0].view(B, 1, 1).float()
    crop_w = crop_hw[:, 1].view(B, 1, 1).float()
    u_px = (grid_u.unsqueeze(0) + 0.5) * (crop_w / float(feat_w))
    v_px = (grid_v.unsqueeze(0) + 0.5) * (crop_h / float(feat_h))

    # Intrinsics
    fx = K_batch[:, 0, 0].view(B, 1, 1).float()
    fy = K_batch[:, 1, 1].view(B, 1, 1).float()
    cx = K_batch[:, 0, 2].view(B, 1, 1).float()
    cy = K_batch[:, 1, 2].view(B, 1, 1).float()

    # NaN/Inf-safe clamp — replace non-finite first, then clamp range
    d = depth_grid[:, 0].float()                     # (B, H', W')
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    d = d.clamp(min=d_min, max=d_max)

    X = d
    Y = -(u_px - cx) * X / fx
    Z = -(v_px - cy) * X / fy

    P = torch.stack([X, Y, Z], dim=-1)               # (B, H', W', 3)
    P = P.reshape(B, feat_h * feat_w, 3)
    return P.to(out_dtype)
```

Correctness check (Designer asserts mentally; Builder does not need runtime test): for a pixel at principal point `(cx, cy)` with depth `d`, outputs `(d, 0, 0)`. Reprojecting via baseline's forward projection `u = fx*(-Y/X)+cx, v = fy*(-Z/X)+cy` recovers `(cx, cy)` exactly. Matches the convention in `recover_pelvis_3d` (see `pelvis_utils.py` lines 14–46).

### 4. Metric 3D PE module — new class in `pose3d_transformer_head.py`

Add at module scope (before `Pose3dTransformerHead`):

```python
class _Metric3DPE(nn.Module):
    """Embed per-token metric 3D coordinates (X, Y, Z in metres) → hidden_dim.

    Architecture: Linear(3, mlp_hidden) → GELU → Linear(mlp_hidden, hidden_dim).
    Final Linear is zero-initialised so PE_3D = 0 at step 0 (identity wrt baseline).
    """

    def __init__(self, hidden_dim: int, mlp_hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(3, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_dim)
        self.act = nn.GELU()
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        # p: (B, N, 3) in metres
        return self.fc2(self.act(self.fc1(p)))
```

### 5. Head `__init__` changes

Add to `__init__` signature (keep all existing args/defaults in place; new args go after `loss_weight_uv` and before `init_cfg`):

```python
use_metric_pe_3d: bool = False,
metric_pe_variant: str = 'mlp_additive',  # 'mlp_additive' | 'sinusoidal' | 'keys_only'
metric_pe_mlp_hidden: int = 256,
metric_pe_depth_clamp_min: float = 0.1,
metric_pe_depth_clamp_max: float = 50.0,
```

Inside `__init__`, after existing module constructions, add:

```python
self.use_metric_pe_3d = bool(use_metric_pe_3d)
self.metric_pe_variant = str(metric_pe_variant)
self.metric_pe_depth_clamp_min = float(metric_pe_depth_clamp_min)
self.metric_pe_depth_clamp_max = float(metric_pe_depth_clamp_max)
if self.use_metric_pe_3d:
    assert self.metric_pe_variant in ('mlp_additive', 'sinusoidal', 'keys_only'), \
        f'unknown metric_pe_variant {self.metric_pe_variant}'
    # Variant A of this design: 'mlp_additive'
    self.metric_pe_3d = _Metric3DPE(hidden_dim, mlp_hidden=int(metric_pe_mlp_hidden))
```

When `use_metric_pe_3d=False`, no new module is created; baseline is bit-for-bit preserved.

### 6. `forward()` signature and body

Change the signature to accept optional metric-3D tensor:

```python
def forward(
    self,
    feats: Tuple[torch.Tensor, ...],
    metric_xyz: torch.Tensor | None = None,   # (B, H'*W', 3) in metres, fp32
) -> Dict[str, torch.Tensor]:
```

Inside `forward()`, after `spatial = spatial + pos_enc` and before `queries = ...`:

```python
if self.use_metric_pe_3d and metric_xyz is not None:
    # Cast to spatial dtype (AMP-safe). The MLP is registered as child so autocast applies.
    pe3d = self.metric_pe_3d(metric_xyz.to(spatial.dtype))    # (B, H*W, hidden_dim)
    spatial = spatial + pe3d
```

Nothing else changes in `forward()`. In Variant A the PE_3D is added to the spatial tokens once and therefore appears in both the keys and the values of the subsequent cross-attention (which is the intended "additive" behaviour).

### 7. `loss()` and `predict()` changes

In both methods, build `metric_xyz` before calling `forward`:

```python
feat_h, feat_w = feats[-1].shape[2], feats[-1].shape[3]
metric_xyz = None
if self.use_metric_pe_3d:
    depth_grid = self._extract_depth_map(
        batch_data_samples, feat_h, feat_w, feats[-1].device)     # (B,1,H',W') fp32
    K_batch, crop_hw = self._build_K_batch(
        batch_data_samples, feats[-1].device)                      # (B,3,3), (B,2) fp32
    from pelvis_utils import unproject_grid_to_metric_3d
    metric_xyz = unproject_grid_to_metric_3d(
        depth_grid, K_batch, crop_hw, feat_h, feat_w,
        d_min=self.metric_pe_depth_clamp_min,
        d_max=self.metric_pe_depth_clamp_max,
    )                                                              # (B, H'*W', 3) fp32
pred = self.forward(feats, metric_xyz=metric_xyz)
```

All GT extraction, loss computation, and `_train_mpjpe` / `_train_mpjpe_abs` telemetry below this block remain identical to baseline.

### 8. `config.py` changes

In the `head=dict(...)` block (baseline `config.py` lines 147–162), add five new keys after `loss_weight_uv=1.0,` and before the closing `),`:

```python
        # ── Metric 3D PE (idea034 / Variant A — MLP additive) ──
        use_metric_pe_3d=True,
        metric_pe_variant='mlp_additive',
        metric_pe_mlp_hidden=256,
        metric_pe_depth_clamp_min=0.1,
        metric_pe_depth_clamp_max=50.0,
```

No other changes to `config.py`. Optimizer, LR schedule, data pipeline, batch size, AMP, seed — all unchanged.

---

## Exact Expected Behaviour

- At step 0, `self.metric_pe_3d.fc2` has zero weight and zero bias, so `pe3d = 0` and `spatial = input_proj(feat) + pos_enc`, identical to baseline forward output. Numerical losses at step 0 equal baseline losses to float precision.
- Added parameters: `Linear(3, 256) + Linear(256, 256) = 3*256+256 + 256*256+256 = 1024 + 65792 = 66.8K` (< 0.04% of backbone).
- Per-step overhead: bilinear depth downsample (~0.1 ms), unprojection arithmetic on `(B=4, H'*W'=960)` (~0.1 ms), 2-layer MLP on `(4*960, 3)→(4*960,256)→(4*960,256)` (~1 ms). Total < 2 ms; negligible wrt backbone.
- `K` and `depth` are present in both train and val (baseline includes `depth_npy_path` in `meta_keys` and `depth_required=True` on `LoadBedlamLabels`), so the mechanism is active at both training and evaluation time with no train/test mismatch.
- Metric keys published by evaluator: unchanged (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`).

---

## Constraints / Invariants the Builder Must Preserve

1. **Zero-init of final Linear is load-bearing.** `self.metric_pe_3d.fc2.weight` and `.bias` must both be zero. Do not re-init with trunc_normal. This guarantees baseline equivalence at step 0.
2. **Convention match with `recover_pelvis_3d`.** Sign conventions `Y = -(u-cx)*X/fx`, `Z = -(v-cy)*X/fy` must be identical. Do NOT swap `fy`/`fx` or flip signs.
3. **Pixel centre offset.** Use `(w + 0.5) * crop_w / W'` and `(h + 0.5) * crop_h / H'`; this aligns feature-cell centres with pixel centres and matches `F.interpolate(..., align_corners=False)` semantics used in the depth downsample.
4. **FP32 for unprojection.** Cast depth, K, and the arithmetic to fp32 inside the helper; cast back to `depth_grid.dtype` at the end. This avoids fp16 overflow/NaN under AMP.
5. **NaN/Inf-safe clamp.** Apply `torch.where(isfinite(d), d, 0.0)` before `clamp(d_min, d_max)`. BEDLAM2 depth NPZs can contain sentinel values for sky/background; these must not propagate.
6. **`depth_required=True` stays in config** (already baseline-default). Depth is required at both train and val, matching baseline transforms.
7. **Body-only joint loss (indices 0–21).** Unchanged.
8. **Output dict keys and shapes unchanged:** `joints (B,70,3)`, `pelvis_depth (B,1)`, `pelvis_uv (B,2)`.
9. **`persistent_workers=False`** — unchanged (NPZ mmap FD issue).
10. **MMEngine config constraint:** all `head=dict(...)` kwargs are bool/int/float/str literals. No Python imports inside the config.
11. **Baseline reproducibility.** `use_metric_pe_3d=False` (default) must leave `forward()` branch-inactive and not construct `self.metric_pe_3d` — this design passes `use_metric_pe_3d=True` in config but the off-path must remain a true no-op for future ablations.
12. **`recover_pelvis_3d` and `compute_mpjpe_abs` in `pelvis_utils.py` are NOT modified.** Only `unproject_grid_to_metric_3d` is added.
13. **No change to `forward()` positional-only arg order for existing callers.** The new `metric_xyz` is keyword-only with default `None`.

---

## Edge Cases

- **Missing `depth_npy_path`:** `_extract_depth_map` falls back to a zero depth tensor for that sample; the clamp floor `d_min=0.1` maps the zeros to `0.1`, producing a benign valid metric coord near the camera. (Learning-wise, this is the same "zero depth" edge case as idea004.)
- **Missing `K`:** baseline always provides K (`PackBedlamInputs` includes `'K'` in `meta_keys` at `config.py:173`). Builder does not need defensive fallback code.
- **`img_shape` tuple length:** baseline convention `(crop_h, crop_w)`; use `int(img_shape[0])`, `int(img_shape[1])`. Do not trust lengths > 2.
- **Per-sample variable crop size:** per-sample `crop_h` and `crop_w` must be used; the design correctly threads per-sample values through `crop_hw` tensor. Do not hardcode `640 × 384` in the unprojection arithmetic.
- **AMP autocast around `self.metric_pe_3d`:** handled automatically since it is a registered child module. The helper does its own fp32 cast for the geometric arithmetic.
