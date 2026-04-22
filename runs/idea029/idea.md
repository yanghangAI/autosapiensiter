**Idea Name:** End-to-End Absolute Body Joint Supervision via 3D Absolute Consistency Loss

**Approach:** Add a direct smooth-L1 loss on the predicted absolute 3D body joint positions (computed by adding the predicted root-relative joints to the unprojected pelvis 3D position), creating a joint end-to-end gradient signal that flows simultaneously through both the relative joint head and the pelvis depth/UV heads — coupling the two prediction pathways that are currently supervised only by independent losses.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Decoupled-Gradient Bottleneck

The baseline trains the head with three independent loss terms:

```
L_joints   = SoftWeightSmoothL1(pred_rel_joints[:, 0:22], gt_rel_joints[:, 0:22])
L_depth    = SmoothL1(pred_depth, gt_depth)
L_uv       = SmoothL1(pred_uv, gt_uv)
```

These three losses are **structurally independent**: the gradient flowing to the relative-joint head never accounts for how that prediction will be combined with the predicted pelvis position, and the gradient flowing to the depth/UV head never accounts for how the relative body pose will shift the absolute joint positions.

The **absolute body joint position** is computed at evaluation time as:

```
joints_abs[i] = joints_rel[i] + unproject(pred_depth, pred_uv, K)
```

where `unproject(...)` converts the predicted pelvis depth and normalised UV to a 3D camera-frame position. The evaluation metric `mpjpe_abs_val` measures the error of this compound prediction. Yet **no training loss directly supervises this compound quantity**. The network must implicitly learn, through independent per-pathway losses, that its relative joints and its pelvis localization will combine correctly in absolute space.

This mismatch between training (independent pathway losses) and evaluation (joint absolute metric) is a fundamental source of the observed absolute MPJPE ceiling (baseline 833mm, best prior 533mm from idea008/design002).

### What Happens Without Absolute Coupling

Consider a failure mode that the current losses cannot penalise: the network predicts the pelvis depth correctly (`L_depth ≈ 0`), the pelvis UV correctly (`L_uv ≈ 0`), and the relative joints correctly (`L_joints ≈ 0`), but makes a **correlated error** where the relative joints are systematically biased away from the pelvis in exactly the direction that the pelvis localization error already compensates. This type of anti-correlated bias can be nearly invisible to the independent losses but catastrophic for `mpjpe_abs`.

More concretely: when the network is uncertain about pelvis depth (BEDLAM2 has varied person-to-camera distances), it may learn to bias relative joints in the X-direction (forward/depth) to partially compensate. The independent loss on relative joints tolerates this bias if it is absorbed by the depth prediction. But the absolute combination will see both biases and produce large absolute errors.

An **absolute body joint loss** directly penalises the combined absolute position, eliminating any such compensatory bias and coupling the gradient from the absolute evaluation metric directly back to both prediction heads during training.

### How the Absolute Loss Works

For each training sample, compute the predicted absolute joint positions:

```python
pred_pelvis_3d = recover_pelvis_3d(pred_depth[i:i+1], pred_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
pred_abs_joints = pred_joints[i, :22] + pred_pelvis_3d   # (22, 3), body joints only
gt_abs_joints   = gt_joints[i, :22]   + gt_pelvis_3d     # (22, 3)
```

The absolute body joint loss is the mean smooth-L1 over 22 body joints:

```python
L_abs = SmoothL1(pred_abs_joints, gt_abs_joints).mean()
```

This loss has **gradients flowing to**:
1. `pred_joints[i, :22]` (relative joint head) — direct gradient
2. `pred_depth[i]` and `pred_uv[i]` (pelvis heads) — via the chain rule through `recover_pelvis_3d`

The gradient through `recover_pelvis_3d` is:

```
d(L_abs) / d(pred_depth) = sum_j [ d(L_abs) / d(pred_pelvis_3d_X) ] * d(pelvis_X) / d(depth)
                          = sum_j [ residual_X_j ]
```

Since `pelvis_X = pred_depth` (X=forward convention), the depth gradient is the sum of residuals in the forward direction across all 22 joints. This is a **much richer signal** than the standalone `L_depth` which only measures the error at the pelvis centroid — the absolute loss simultaneously tells the depth head how far off it is for every body joint, not just the pelvis root.

### Why This is Different from All Prior Ideas

| Prior Idea | Mechanism | Difference from idea029 |
|---|---|---|
| idea010 (2D reprojection) | Projects predicted abs joints to 2D and computes reprojection error | idea010 operates in 2D image space; idea029 operates in full 3D camera-frame space. 2D reprojection is depth-invariant for small depth errors; 3D absolute loss is not. idea010 improved stage-2 body MPJPE to 168.8mm (best ever) but did not address absolute errors which require 3D supervision |
| idea005 (uncertainty weighting) | Reweights 3 task-level losses | Operates on existing per-task losses; no coupling between relative and absolute pathways |
| L_depth + L_uv + L_joints (baseline) | Independent per-pathway losses | No coupling; cannot penalise anti-correlated bias between relative and absolute predictions |
| idea014 (anchor-based depth) | Changes pelvis depth representation | Output representation change; no coupling loss |
| idea016 (depth-conditional modulation) | Scales spatial features by depth signal | Input-side depth integration; no coupling loss |

**idea029 is the first idea to add a loss term on `joints_rel + pelvis_3d`** — the exact quantity that `mpjpe_abs_val` measures. Every prior loss operates in either relative space (joint loss) or absolute-centroid space (depth/UV losses), but never on the compound absolute prediction.

### Grounding in Observed Results

**The absolute MPJPE is a large, underexploited improvement channel:**

- Baseline `mpjpe_abs_val` = 833mm at stage-1. Best prior = 533mm (idea008/design002 stage-2). This is the most room for improvement of any tracked metric.
- Composite formula is `0.67 * mpjpe_body_val + 0.33 * mpjpe_pelvis_val`. The composite metric does NOT directly include `mpjpe_abs` — but `mpjpe_pelvis_val` IS directly about the pelvis localization quality, which is the main driver of absolute errors. `mpjpe_abs = mpjpe_rel + pelvis_offset_error` approximately.
- **idea010/design002 stage-2** achieved the best body MPJPE (168.8mm) using 2D reprojection. The 2D reprojection loss provided indirect gradient coupling between relative joints and pelvis predictions. A direct 3D absolute loss is strictly more informative: it captures the full 3D position error rather than the 2D projection.
- **Pelvis MPJPE plateau**: at stage-1, pelvis MPJPE ranges 608–740mm across 27 ideas. No idea has substantially broken this floor. The pelvis MPJPE (`mpjpe_pelvis_val`) measures how well the pelvis is localized in 3D — exactly what the absolute joint loss reinforces. By providing body joint supervision in absolute space, the depth/UV heads receive gradient from 22 sources (one per body joint) rather than just one (the pelvis centroid), making the signal much denser.
- **idea023** (heatmap-guided query init, best stage-1) reached composite 323.75 with body MPJPE 183.4mm and pelvis MPJPE 608.6mm. The body pathway is near its current floor. The pelvis pathway, by contrast, still has ~150-200mm of improvement potential based on the stage-2 gap (365mm at baseline stage-2 vs 308mm best). An absolute coupling loss targets this pelvis gap more directly than any prior idea.

---

## Proposed Variations

### Design A — Absolute body joint loss, uniform weight λ=0.5 (diagnostic)

Add a new loss term `L_abs` computed via a per-sample loop (same structure as `compute_mpjpe_abs` in `pelvis_utils.py`) using `recover_pelvis_3d` for each sample. Apply smooth-L1 with the same `beta=0.05` as the joint loss. Weight: `λ_abs = 0.5`.

The absolute loss is computed over **body joints only** (indices 0–21), consistent with the primary training signal. Hand joints (22–69) are excluded because hand absolute positions have negligible effect on composite_val and hand positions are not well-supervised in stage-1 (body-only loss scope).

Config kwargs: `abs_joint_loss_weight=0.5`, `abs_joint_indices=22` (how many body joints to include, 0–21).

This is the minimal diagnostic: does direct 3D absolute supervision improve composite_val and pelvis MPJPE?

### Design B — Absolute body joint loss with depth-axis upweighting, λ=0.5

Same architecture as Design A, but the smooth-L1 loss over the 3D coordinate axes is **asymmetrically weighted**: the X-axis (forward/depth, the hardest axis to predict in camera-frame) gets weight 2.0, while Y and Z get weight 1.0.

Rationale: in BEDLAM2's coordinate system (X=forward=depth), the X-coordinate of each joint is `joint_rel_X + pelvis_depth`. Depth uncertainty is the dominant source of absolute error — the network can localise Y and Z reasonably well from image appearance, but X requires accurate depth estimation. Upweighting the X-axis residual provides stronger gradient signal for the depth pathway.

```python
axis_weights = torch.tensor([2.0, 1.0, 1.0], device=pred_abs.device)  # (3,) X, Y, Z
abs_residual = smooth_l1(pred_abs, gt_abs)  # (B, 22, 3) element-wise
weighted_loss = (abs_residual * axis_weights.view(1, 1, 3)).mean()
```

Config kwargs: `abs_joint_loss_weight=0.5`, `abs_joint_axis_weights=[2.0, 1.0, 1.0]`, `abs_joint_indices=22`.

### Design C — Absolute body joint loss with adaptive weight decoupling via stop-gradient

Same as Design A (uniform weight, λ=0.5), but the gradient through the **pelvis branch** and the **relative joint branch** is weighted differently using selective stop-gradient:

- Through the **relative joint head**: gradient flows freely (full weight 1.0). This helps the model learn relative joint positions that are consistent with absolute space.
- Through the **pelvis depth/UV heads**: gradient is scaled by 0.5 (half weight). This prevents the absolute loss from over-riding the direct depth/UV supervision (`L_depth`, `L_uv`) which already provide clean per-target signals.

Implementation: in the forward pass, when computing the absolute body joint loss, use `pred_pelvis_3d_for_abs = recover_pelvis_3d(pred_depth.detach() * (1 - alpha) + pred_depth * alpha, ...)` where `alpha=0.5`. A cleaner approach: compute `pred_abs_joints` twice — once with `pred_pelvis_3d` as-is for gradient through both branches, once with `pred_pelvis_3d.detach()` for gradient through the relative joint head only. The total loss is:

```python
# Full gradient version (through both branches)
pred_abs_full   = pred_rel_joints + pred_pelvis_3d           # gradient to both
# Detached pelvis version (gradient to rel joints only)
pred_abs_detach = pred_rel_joints + pred_pelvis_3d.detach()  # gradient to rel joints only

abs_loss = 0.5 * smooth_l1(pred_abs_full, gt_abs).mean() + \
           0.5 * smooth_l1(pred_abs_detach, gt_abs).mean()
```

This effectively gives the relative joint branch a gradient of 1.0 and the pelvis branch a gradient of 0.5 from the absolute loss, while the direct depth/UV losses continue at full strength. The decoupling prevents the depth/UV heads from being dominated by the indirect absolute signal when the relative joint prediction is already good.

Config kwargs: `abs_joint_loss_weight=0.5`, `abs_joint_pelvis_grad_scale=0.5`, `abs_joint_indices=22`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. A helper function is added to **`pelvis_utils.py`**.

### `pelvis_utils.py`

Add one new helper function `recover_pelvis_3d_batched_list` that encapsulates the per-sample loop for the absolute joint loss computation. This mirrors the structure of `compute_mpjpe_abs` but returns tensors with gradients instead of a scalar MPJPE value:

```python
def recover_abs_joints_batched(
    pred_joints_rel: torch.Tensor,  # (B, J, 3)
    gt_joints_rel: torch.Tensor,    # (B, J, 3)
    pred_depth: torch.Tensor,       # (B, 1)
    gt_depth: torch.Tensor,         # (B, 1)
    pred_uv: torch.Tensor,          # (B, 2)
    gt_uv: torch.Tensor,            # (B, 2)
    batch_data_samples: Sequence,
    num_body_joints: int = 22,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute predicted and GT absolute joint positions (with gradients).

    Returns:
        pred_abs: (B, num_body_joints, 3) predicted absolute body joints.
        gt_abs:   (B, num_body_joints, 3) GT absolute body joints.
    """
    B = pred_joints_rel.size(0)
    pred_abs_list = []
    gt_abs_list = []
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h, crop_w = int(img_shape[0]), int(img_shape[1])

        pred_pelvis = recover_pelvis_3d(
            pred_depth[i:i+1], pred_uv[i:i+1], K, crop_h, crop_w)   # (1, 3)
        gt_pelvis = recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h, crop_w)       # (1, 3)

        pred_abs_list.append(pred_joints_rel[i, :num_body_joints] + pred_pelvis)
        gt_abs_list.append(gt_joints_rel[i, :num_body_joints] + gt_pelvis)

    return torch.stack(pred_abs_list), torch.stack(gt_abs_list)  # (B, J, 3) each
```

**Designer note**: this function is nearly identical to `compute_mpjpe_abs` except it returns the raw tensors (preserving gradients) instead of a scalar MPJPE. The key difference is that no `.norm()` or `* 1000.0` is applied — we need the gradient to flow back through the joint coordinates. The Designer should import this helper at the top of `pose3d_transformer_head.py` alongside the existing `from pelvis_utils import compute_mpjpe_abs`.

### `pose3d_transformer_head.py`

**1. Import addition**

```python
from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs
from pelvis_utils import recover_abs_joints_batched as _recover_abs_joints_batched
```

**2. `__init__` additions**

New kwargs (all defaulting to baseline behaviour):

```python
abs_joint_loss_weight: float = 0.0          # 0.0 = baseline (no absolute loss)
abs_joint_indices: int = 22                  # how many body joints to include
abs_joint_axis_weights: list = None          # per-axis weights (Design B); None = uniform
abs_joint_pelvis_grad_scale: float = 1.0    # pelvis branch gradient scale (Design C); 1.0 = full grad
```

Storage in `__init__`:
```python
self.abs_joint_loss_weight = abs_joint_loss_weight
self.abs_joint_indices = abs_joint_indices
self.abs_joint_pelvis_grad_scale = abs_joint_pelvis_grad_scale

if abs_joint_axis_weights is not None:
    w = torch.tensor(abs_joint_axis_weights, dtype=torch.float32)  # (3,)
    self.register_buffer('abs_axis_weights', w)
else:
    self.abs_axis_weights = None
```

**3. `loss()` additions** (after the existing joint/depth/UV losses)

```python
# ── Absolute Body Joint Consistency Loss ────────────────────────────────
if self.abs_joint_loss_weight > 0.0:
    # Get GT depth and UV from batch_data_samples (same as existing loss code)
    # pred['joints'] is (B, 70, 3) root-relative; pred['pelvis_depth'] is (B,1); pred['pelvis_uv'] is (B,2)

    # Optionally scale the gradient through the pelvis branch (Design C)
    if self.abs_joint_pelvis_grad_scale < 1.0:
        alpha = self.abs_joint_pelvis_grad_scale
        # Full gradient version: gradient flows to both relative joints and pelvis
        pred_abs_full, gt_abs = _recover_abs_joints_batched(
            pred['joints'], gt_joints,
            pred['pelvis_depth'], gt_depth,
            pred['pelvis_uv'], gt_uv,
            batch_data_samples, self.abs_joint_indices)
        # Detached pelvis version: gradient flows to relative joints only
        pred_abs_det, _ = _recover_abs_joints_batched(
            pred['joints'], gt_joints,
            pred['pelvis_depth'].detach(), gt_depth,
            pred['pelvis_uv'].detach(), gt_uv,
            batch_data_samples, self.abs_joint_indices)
        pred_abs = alpha * pred_abs_full + (1.0 - alpha) * pred_abs_det
    else:
        pred_abs, gt_abs = _recover_abs_joints_batched(
            pred['joints'], gt_joints,
            pred['pelvis_depth'], gt_depth,
            pred['pelvis_uv'], gt_uv,
            batch_data_samples, self.abs_joint_indices)

    # Compute smooth-L1 over absolute joint positions
    beta_abs = 0.05
    abs_diff = (pred_abs - gt_abs).abs()
    abs_loss_raw = torch.where(
        abs_diff < beta_abs,
        0.5 * abs_diff ** 2 / beta_abs,
        abs_diff - 0.5 * beta_abs,
    )  # (B, abs_joint_indices, 3)

    # Per-axis weighting (Design B)
    if self.abs_axis_weights is not None:
        abs_loss_raw = abs_loss_raw * self.abs_axis_weights.view(1, 1, 3)

    losses['loss/abs_joints/train'] = self.abs_joint_loss_weight * abs_loss_raw.mean()
```

**Designer note on gt_depth / gt_uv extraction**: the existing `loss()` method already extracts `gt_depth` and `gt_uv` from `batch_data_samples` for the standalone depth/UV losses. The Designer should reuse those already-extracted tensors for the absolute loss computation, rather than re-extracting from `batch_data_samples`. Check the baseline `loss()` code to confirm how `gt_depth` and `gt_uv` are assembled (they may be stacked from per-sample metainfo in a loop similar to `compute_mpjpe_abs`). The Designer must pass these same tensors to `_recover_abs_joints_batched`.

### `config.py`

**Design A:**
```python
abs_joint_loss_weight=0.5,
abs_joint_indices=22,
```

**Design B:**
```python
abs_joint_loss_weight=0.5,
abs_joint_indices=22,
abs_joint_axis_weights=[2.0, 1.0, 1.0],
```

**Design C:**
```python
abs_joint_loss_weight=0.5,
abs_joint_indices=22,
abs_joint_pelvis_grad_scale=0.5,
```

All values are float/int/list-of-float literals. No Python import statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — mpjpe_pelvis_val**: the absolute joint loss provides 22× denser gradient signal to the depth/UV heads (one residual per body joint vs one residual for the pelvis centroid). The depth head receives gradient that measures how much pelvis depth error propagates into each body joint's absolute position — far more informative than the scalar depth loss. Target: `mpjpe_pelvis_val < 580mm` at stage-1 (vs. baseline 652mm, best prior 608mm from idea023/design001). At stage-2: `< 310mm` (vs. baseline 365mm, best prior 322mm from idea003/design002).

- **Secondary gain — mpjpe_body_val**: the absolute loss simultaneously supervises the relative joint head in absolute space, providing additional gradient for joints near the pelvis where relative and absolute errors are correlated. Target: `mpjpe_body_val < 185mm` at stage-1.

- **Tertiary gain — mpjpe_abs_val**: the absolute MPJPE metric directly measures what the absolute loss trains. Target: `mpjpe_abs_val < 750mm` at stage-1 (vs. baseline 833mm), `< 550mm` at stage-2 (vs. baseline 628mm).

- **Design A** (uniform weight, λ=0.5): minimal diagnostic. Expected composite_val < 335 at stage-1 driven by pelvis MPJPE improvement.

- **Design B** (depth-axis upweighted): the 2.0× upweighting on the X-axis amplifies gradient signal specifically for depth errors, which are the dominant source of absolute MPJPE. Expected to outperform Design A on `mpjpe_pelvis_val`. Expected composite_val < 330.

- **Design C** (pelvis-grad scaled 0.5): prevents the absolute loss from over-competing with the clean direct depth/UV losses. The relative joint branch benefits more from the absolute signal than the pelvis branch (which is already well-supervised by `L_depth` and `L_uv`). Expected most stable training. Expected composite_val < 330.

- **Composite target (stage-1)**: `composite_val < 330`, improving on best prior stage-1 of 323.75 (idea023/design001) in the best case. Primary gain through pelvis MPJPE reduction.
- **Composite target (stage-2)**: `composite_val < 222`, competitive with best prior stage-2 of 224.52 (idea001/design001).

---

## Risk and Mitigation

- **Double-counting gradient with L_depth and L_uv**: the standalone depth and UV losses already provide direct gradient to the depth/UV heads. The absolute joint loss adds a second, indirect gradient path for the same parameters. If both are too strong, the depth/UV parameters receive conflicting gradients. Mitigation: Design C's `abs_joint_pelvis_grad_scale=0.5` halves the absolute-loss gradient to the pelvis branch, keeping the combined gradient manageable. Design A/B use `λ_abs=0.5` which at 22-joint averaging produces a per-parameter gradient contribution similar in scale to the existing `L_depth` + `L_uv` terms.

- **Gradient scale analysis**: the absolute loss has 22 joints × 3 coords = 66 residual terms per sample. The standalone joint loss also has 22 × 3 = 66 terms. At `λ_abs=0.5`, the absolute loss contributes 50% of the per-parameter gradient of the joint loss to the relative joint head, and 50% × (gradient through recover_pelvis_3d w.r.t. depth/UV) to the pelvis head. The pelvis branch gradient from the absolute loss is approximately `λ_abs × 22 × mean_abs_err / pelvis_depth_scale` — at pelvis depth ≈ 3-5m (BEDLAM2 range), this is well-controlled.

- **`recover_pelvis_3d` differentiability**: the function uses only arithmetic operations (multiplication, division, addition) on tensors — fully differentiable through PyTorch autograd. The only non-differentiable step is reading `K`, `crop_h`, `crop_w` from numpy/Python scalars, which is standard practice in the existing codebase (same pattern as `compute_mpjpe_abs`). No special treatment needed.

- **Per-sample loop overhead**: the implementation loops over `B=4` samples to handle per-sample `K` and `img_shape`. This mirrors the existing `compute_mpjpe_abs` loop in the baseline `loss()`. The additional overhead is identical to the training-time abs MPJPE computation already performed in the baseline — approximately 0.5ms per batch, negligible.

- **AMP / float16 safety**: `recover_pelvis_3d` performs divisions by `fx`, `fy` (scalars from K, dtype float32 from numpy). Mixed-precision autocast will upcast the computation to float32 for the division. The gradient through the absolute loss is computed in float16 in the main autocast region, but since the loss is a `.mean()` of smooth-L1 values (bounded by the loss scale), there is no overflow risk. Designer should verify that `abs_diff` values are in the expected range (0–10m) by inspecting early training loss curves.

- **Interaction with idea010 (2D reprojection loss)**: idea010 adds a 2D reprojection loss that also couples relative joints with pelvis predictions. The absolute 3D loss (idea029) and the 2D reprojection loss (idea010) are complementary: the 2D loss provides image-plane coupling while the 3D loss provides full camera-frame coupling. They can be combined in a future idea. In isolation (this idea only), the 3D absolute loss is strictly more informative than the 2D reprojection loss because it captures depth errors that are invisible to the 2D projection.

- **Interaction with idea028 (decoupled pelvis queries)**: if the pelvis head is decoupled (idea028), the gradient path from the absolute joint loss to the pelvis head changes: it flows into the dedicated pelvis queries rather than into joint token 0. This is fully compatible — the `pred['pelvis_depth']` and `pred['pelvis_uv']` tensors are the same regardless of whether they come from the baseline token 0 or from dedicated pelvis queries. The Designer working on a combined design should be aware of this.

- **Interaction with idea013 (kinematic bone vectors)**: the absolute joint loss operates on the kinematically-recovered absolute joint positions (`joints_rel + pelvis_3d`). If the relative joints come from bone-vector forward kinematics, the gradient flows back through the kinematic chain (all ancestor bones receive gradient). This makes the combination of idea013 + idea029 particularly powerful: the kinematic structure enforces internal consistency, while the absolute loss enforces global localization consistency. A future combined idea is recommended if both ideas show individual gains.

- **Boundary cases for near-camera subjects**: for subjects very close to the camera (pelvis depth < 1m), the absolute body joint positions have large absolute values, and the smooth-L1 loss activates the linear regime more frequently. At `beta=0.05m`, any absolute joint error > 5cm activates linear regime — which is always true in early training. This is consistent with the joint loss (`beta=0.05` in the baseline). No special handling needed.

- **MMEngine config constraint**: all new kwargs in config.py are float/int/list-of-float literals. No Python import statements. Fully compliant.

- **Memory**: the new tensors `pred_abs` and `gt_abs` are `(B=4, 22, 3)` = 264 float16 values ≈ 528 bytes. The `abs_axis_weights` buffer is 3 floats ≈ 12 bytes. Total additional memory: negligible on the 2080 Ti.

- **Validation**: the composite metric `composite_val = 0.67 * mpjpe_body + 0.33 * mpjpe_pelvis` does not directly include `mpjpe_abs`. However, pelvis MPJPE (`mpjpe_pelvis_val`) measures the pelvis localization quality that is the main driver of absolute errors. Improving pelvis MPJPE is the primary route to improving composite_val beyond the current 323mm floor. The absolute joint loss provides a richer gradient signal for the pelvis pathway than any prior idea.
