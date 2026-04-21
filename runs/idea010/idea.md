**Idea Name:** Auxiliary 2D Reprojection Consistency Loss

**Approach:** Add an auxiliary loss that projects the predicted absolute body joints (assembled from predicted root-relative joints plus predicted pelvis depth/UV via the unprojection pipeline) through the camera intrinsics K back to 2D pixel coordinates, and supervises this projection against the GT 2D projections computed from GT 3D joints and K, creating a geometric consistency signal that couples the joint and pelvis pathways through the camera model.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline head regresses three quantities independently:
- 70 root-relative 3D joint coordinates (body supervision on indices 0–21)
- 1 pelvis depth scalar
- 2 pelvis UV scalars

Each has its own loss with no *cross-term* that enforces geometric consistency between them. This has three concrete consequences:

### 1. No gradient coupling between joint pathway and pelvis pathway

Even in architectures that decouple the pelvis pathway (idea002, design002/003 — composite 154.85–155.21), the only information link between joint-query regression and pelvis-depth/UV regression is *shared features in the backbone*. At the loss level, an error in pelvis depth does not increase the joint loss, and an error in joint XYZ does not increase the pelvis loss. The model has no loss-level incentive to make the full 3D predicted pose *projection-consistent* with the 2D image.

### 2. Pelvis MPJPE has plateaued

Across all 9 prior ideas, `mpjpe_pelvis_val` stays in the 174–185 mm band. Even idea002, which architecturally dedicates capacity to pelvis depth/UV, did not significantly improve it (176–184 mm) — it *enabled* body improvements rather than improving pelvis itself. Pelvis depth is a single-scalar regression from a compressed global feature; without a stronger supervision signal it is very difficult to improve.

### 3. mpjpe_abs is the most degraded metric

Looking at the CSV:
- baseline: `mpjpe_abs = 454.9` mm
- best (idea008/design003): `mpjpe_abs = 320.3` mm

Absolute MPJPE is the sum of root-relative joint error and pelvis reconstruction error. It is by far the weakest metric and is directly controlled by how well the joints + pelvis predictions *geometrically combine*. A 2D reprojection loss is the natural signal for this.

### Why reprojection consistency is well-suited here

BEDLAM2 samples carry the camera intrinsic matrix `K` in `data_sample.metainfo['K']`. Given:
- predicted root-relative joints `pred_joints[b] ∈ R^{22×3}` (body only)
- predicted pelvis absolute position `pred_pelvis[b] ∈ R^{3}` (already computed by `recover_pelvis_3d` in `pelvis_utils.py`)
- GT absolute joints `gt_joints_abs[b] ∈ R^{22×3}`

we can project `pred_joints_abs = pred_pelvis + pred_joints` through K to get 2D pixel coordinates, then compute an L1 loss against `project(gt_joints_abs, K)` normalised to `[-1, 1]` crop coordinates (same convention used for pelvis_uv). This loss:

1. **Couples the pelvis depth/UV and joint pathways**: an error in either pelvis depth OR joint root-relative coordinates shifts the 2D projection, producing a non-zero reprojection loss. The optimiser learns to reduce this by jointly improving both.
2. **Is geometrically grounded**: 2D pixel error has a well-defined meaning in image space, and most of the image evidence is itself 2D (appearance features, silhouette edges). Aligning predictions to 2D is inherently well-supervised.
3. **Is already aligned with the evaluation metric**: `mpjpe_abs` rewards geometrically consistent 3D; reprojection loss is a soft proxy for absolute-pose accuracy via 2D, which the model *can* learn much easier than absolute metres.
4. **Adds zero architecture**: pure loss modification in `pose3d_transformer_head.py` and helper additions in `pelvis_utils.py`. Compatible with every prior idea as a drop-in augmentation.

### Differentiation from prior ideas

| Idea | Mechanism | Difference |
|---|---|---|
| idea001 | Multi-layer decoder | Adds capacity; no cross-task coupling |
| idea002 | Decoupled pelvis query | Architectural decoupling; *opposite direction* (separates pathways) |
| idea003 | Content-adaptive query init | Query warm-start; no loss coupling |
| idea004 | Depth-aware positional encoding | Input side; no loss coupling |
| idea005 | Uncertainty-weighted loss | Rebalances existing 3 losses; **no new loss, no coupling** |
| idea006 | Skeleton self-attention bias | Query self-attention; no loss coupling |
| idea007 | Joint-group spatial routing | Cross-attention; no loss coupling |
| idea008 | Body-only decoder | Query reduction; no loss coupling |
| idea009 | Spatial token dropout | Regularisation; no loss coupling |

This idea is the **first loss-level coupling** between joints, pelvis depth, and pelvis UV. It is strictly orthogonal — the loss can compose with any architectural idea above.

## Analysis of Baseline Weak Point

The current loss summation is:
```
loss_total = 1.0 * L_joints(pred_joints[:, 0:22], gt_joints[:, 0:22])
           + 1.0 * L_depth(pred_depth, gt_depth)
           + 1.0 * L_uv(pred_uv, gt_uv)
```

All three terms are *independently optimisable* — if the model sets `pred_depth = gt_depth` perfectly but `pred_joints` is off by a constant 3D offset, the joint loss increases but the depth loss does not care. There is no feedback loop through the camera geometry.

A reprojection loss is:
```
pred_joints_abs = recover_pelvis_3d(pred_depth, pred_uv, K, H, W) + pred_joints  # (B, 22, 3)
pred_joints_2d = project(pred_joints_abs, K, H, W)                                # (B, 22, 2), normalised to [-1, 1]
gt_joints_2d   = project(gt_joints_abs, K, H, W)                                   # (B, 22, 2) — can be precomputed in loss()
L_reproj = L1(pred_joints_2d, gt_joints_2d)
```

Here, a perturbation of `pred_depth` OR `pred_uv` OR `pred_joints` all affect `pred_joints_2d`, so the gradient of L_reproj flows back into *all three* prediction heads simultaneously.

This directly targets the pelvis+joint coupling gap and is expected to especially help `mpjpe_abs` and `mpjpe_pelvis_val`.

## Proposed Variations

**Design A — Reprojection L1 on body joints (minimal)**

Add a single auxiliary reprojection loss:
```
L_reproj = smooth_l1(pred_body_joints_2d, gt_body_joints_2d)
loss_total = ... baseline ... + λ_reproj * L_reproj
```
with a small `λ_reproj = 0.5` to avoid destabilising the existing losses. Body joints only (22 joints, indices 0–21) to match the evaluation metric. This tests whether the bare coupling signal helps.

**Design B — Reprojection + pelvis-included (stronger coupling)**

Include the pelvis joint explicitly: compute reprojection on all 22 body joints AND include a reprojection consistency term on the pelvis itself (project pred_pelvis through K and compare to the GT pelvis projection). This gives the pelvis pathway a stronger direct supervision signal from image-space error. `λ_reproj = 1.0`. Expected to improve pelvis MPJPE more than Design A.

**Design C — Depth-weighted reprojection (geometry-aware)**

Same as Design B but weight each joint's reprojection term by the predicted 3D distance `||pred_joint_abs||`. Rationale: joints farther from the camera have smaller pixel footprints per millimetre of 3D error, so a naive 2D L1 loss under-weights distant-frame errors that translate to large 3D error. Multiplying by the predicted-depth factor approximately converts pixel error back to mm-scale 3D error:
```
w_i = pred_depth_per_joint_i / fx  (approx 3D-equivalent weight)
L_reproj = sum_i w_i * |pred_2d_i - gt_2d_i|
```
This is the most sophisticated variant; tests whether a geometry-aware reprojection is worth the complexity.

## Implementation Scope

Changes are confined to **two** allowed files:

### `pelvis_utils.py`
Add a new helper `project_joints_to_2d(joints_abs, K, crop_h, crop_w)`:
- Input: `(B, J, 3)` absolute camera-frame joints, `K (3,3)` (numpy or tensor), crop dims.
- Output: `(B, J, 2)` normalised-to-[-1, 1] pixel coordinates using the same convention as `pelvis_uv`:
  `u = fx*(-Y/X) + cx; v = fy*(-Z/X) + cy; then normalise to [-1, 1]`.
- Must handle small-X safely (clamp `X ≥ 0.01`) to avoid NaN.
- Fully differentiable (torch ops only).

### `pose3d_transformer_head.py`
In `loss()`:
1. Compute `pred_pelvis_abs = recover_pelvis_3d(pred_depth, pred_uv, K, H, W)` per-sample (already imported from pelvis_utils).
2. Compute `pred_body_joints_abs = pred_pelvis_abs + pred_joints[:, 0:22]`.
3. Compute `gt_body_joints_abs = gt_pelvis_abs + gt_joints[:, 0:22]` (gt_pelvis comes from `recover_pelvis_3d(gt_depth, gt_uv, K, H, W)`).
4. Project both through `project_joints_to_2d`.
5. Add `loss/reproj/train = λ * smooth_l1(pred_2d, gt_2d)` to the losses dict.
6. Per-sample K loop mirrors the existing `compute_mpjpe_abs` structure — minimal new code.

### `config.py`
- Add `reproj_loss_weight: 0.5 | 1.0 | 1.0` as a head kwarg.
- For Design C, add `reproj_depth_weighted: True` flag.

No changes to `bedlam_metric.py`, backbone, data pipeline, or `train.py` wrapper.

## Expected Outcome

- **Primary gain — pelvis MPJPE**: the reprojection loss directly supervises the pelvis UV and depth via image-space 2D error, which is a strong signal that the current scalar losses lack. Target: `mpjpe_pelvis_val < 170` (vs. baseline 176, best prior 174).
- **Secondary gain — mpjpe_abs**: geometric consistency between joints and pelvis reconstruction should reduce absolute pose error significantly. Target: `mpjpe_abs < 400` (vs. baseline 455).
- **Body MPJPE**: expected neutral to mild positive. The 2D signal provides additional gradient to body-joint regression, which should not hurt and may help convergence on body joints near the image boundary where 2D evidence is strong.
- **Composite target**: aim for `composite_val < 160`, improving on baseline (168.7) with a pathway that specifically targets the pelvis weakness that all prior ideas left mostly untouched.

## Risk and Mitigation

- **Small-X numerical instability**: when predicted depth X is near zero, the projection diverges. Mitigation: clamp `X = max(X, 0.01)` in `project_joints_to_2d` — identical to the clamp already used in `SubtractRootJoint`.
- **Early-training instability from 2D loss before depth is learned**: at init, `pred_depth ≈ 0` and projections may be wild. Mitigation: (a) `recover_pelvis_3d` already guards against this, (b) use `smooth_l1` (not `L1`) for soft clipping of large errors, (c) start with `λ_reproj = 0.5` in Design A. Designer may explore warmup via optim iterations.
- **K availability in loss**: K is in `data_sample.metainfo['K']`, which is always present after `CropPersonRGBD`. The `compute_mpjpe_abs` function already extracts it per-sample — the same pattern is used in the new loss.
- **Gradient interaction with idea002's decoupling**: the reprojection loss couples the pelvis and joint pathways at the loss level. If combined with a decoupled-pelvis architecture, the result is architectural separation + loss-level consistency, which is actually the ideal combination (specialist pathways + consistency constraint). Orthogonal composition is safe.
- **Memory / speed**: adds one small projection computation per forward. Per-sample K loop is identical to `compute_mpjpe_abs`; negligible overhead on 1080 Ti.
- **MMEngine config constraint**: `reproj_loss_weight` is a float literal; `reproj_depth_weighted` is a bool literal. No imports required.
- **Interaction with Evaluation**: `bedlam_metric.py` is invariant — reprojection loss is training-only. No change to val metric.
