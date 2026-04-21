# Design 001 — Upper-triangular Pairwise L1 Distance-Matrix Loss (minimal, lambda=0.5)

**Design Description:** Add a single auxiliary L1 loss on the upper-triangular pairwise Euclidean distance matrix of the 22 predicted body joints (231 pairs) vs the corresponding GT distance matrix, with scalar weight `dist_loss_weight=0.5` added on top of the unchanged baseline losses. Distances are computed with `torch.cdist(pred_body, pred_body, p=2)`; no new learnable parameters; training-only; `predict()` and eval untouched.

**Starting Point:** `baseline/`

---

## Overview

The baseline loss supervises the 22 body joints with per-joint Smooth-L1 on root-relative 3D coordinates. It says nothing about **inter-joint relationships** (bone lengths, cross-body distances, limb proportions). This design adds a single auxiliary term

```
L_dist = mean_{i<j} | ||pred_body[i] - pred_body[j]||_2 - ||gt_body[i] - gt_body[j]||_2 |
losses['loss/dist_matrix/train'] = dist_loss_weight * L_dist
```

where the body joints are indices `0..21` (matching `_BODY = list(range(0, 22))` in baseline `loss()`). Only the 231 upper-triangular pairs `(i, j)` with `i < j` are used (no redundancy and no diagonal). The auxiliary loss is added to the `losses` dict with key `'loss/dist_matrix/train'` and weight `0.5`. All existing loss terms (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`) are unchanged in value and weight.

The mechanism is translation-invariant (pure distances), root-invariant (distances between *any* pair — including pairs that involve joint 0, the pelvis root of the root-relative coordinates), and fully differentiable. The extra compute per batch is a `torch.cdist` on `(B, 22, 3)` plus a constant-index gather — well under 1 ms on 1080 Ti.

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are identical to the baseline.

---

## Files to Change

1. `pose3d_transformer_head.py` — accept three new kwargs (`dist_loss_weight`, `dist_loss_mode`, `dist_loss_eps`) in `Pose3dTransformerHead.__init__`, store them; append the auxiliary distance-matrix loss term in `loss()`; no changes to `forward()` or `predict()`.
2. `config.py` — add `dist_loss_weight=0.5`, `dist_loss_mode='abs'`, `dist_loss_eps=1e-3` inside the `head=dict(...)` block.
3. `pelvis_utils.py` — **no change**.

No new imports are introduced beyond those already present (`torch`, `torch.nn`). `torch.cdist` and `torch.triu_indices` are used but both are in the base `torch` namespace.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `Pose3dTransformerHead.__init__` — new parameters

Add three kwargs to the `__init__` signature, placed immediately after `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`:

```python
dist_loss_weight: float = 0.0,
dist_loss_mode: str = 'abs',
dist_loss_eps: float = 1e-3,
```

Store them as attributes immediately after the existing `self.loss_weight_uv = loss_weight_uv` line (or equivalent spot near the top of `__init__`):

```python
self.dist_loss_weight = dist_loss_weight
self.dist_loss_mode = dist_loss_mode
self.dist_loss_eps = dist_loss_eps
```

Validate `dist_loss_mode` once in `__init__` with a simple assert (fail-fast on typos):

```python
assert dist_loss_mode in ('abs', 'bone_weighted', 'log'), (
    f"dist_loss_mode must be one of 'abs' | 'bone_weighted' | 'log', "
    f"got {dist_loss_mode!r}")
```

Constraints:
- Default `dist_loss_weight=0.0` preserves baseline behaviour exactly (loss term is identically zero). Design 001 sets `0.5` via config.
- Default `dist_loss_mode='abs'` matches Design 001. (Design 002 will set `'bone_weighted'`, Design 003 will set `'log'`.)
- Default `dist_loss_eps=1e-3` is used only by mode `'log'`; in Design 001 it is unused but must still be stored (so Designs 002 and 003 can reuse the same signature).
- NO new `nn.Module` / `nn.Parameter` / `register_buffer` calls in Design 001. Weights are a Python float only.

#### 2. `_init_head_weights` — unchanged

No change. No new learnable parameters are introduced in Design 001.

#### 3. `forward()` — unchanged

No change. The distance-matrix loss operates on the existing `pred['joints']` tensor in `loss()` only.

#### 4. `loss()` — append auxiliary distance-matrix term

Inside `Pose3dTransformerHead.loss`, AFTER the existing three loss assignments

```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
    pred['pelvis_depth'], gt_depth)
losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
    pred['pelvis_uv'], gt_uv)
```

and BEFORE the `with torch.no_grad():` MPJPE block, insert the following block:

```python
# Auxiliary pairwise distance-matrix loss on body joints (indices 0-21).
if self.dist_loss_weight > 0.0:
    pred_body = pred['joints'][:, _BODY]      # (B, 22, 3)
    gt_body = gt_joints[:, _BODY]              # (B, 22, 3)

    D_pred = torch.cdist(pred_body, pred_body, p=2)  # (B, 22, 22)
    D_gt = torch.cdist(gt_body, gt_body, p=2)        # (B, 22, 22)

    # Upper-triangular indices for 22 joints: 22*21/2 = 231 pairs.
    iu = torch.triu_indices(22, 22, offset=1, device=pred_body.device)
    d_pred = D_pred[:, iu[0], iu[1]]          # (B, 231)
    d_gt = D_gt[:, iu[0], iu[1]]              # (B, 231)

    if self.dist_loss_mode == 'abs':
        # Design 001: plain absolute-distance L1.
        L_dist = (d_pred - d_gt).abs().mean()
    elif self.dist_loss_mode == 'bone_weighted':
        # Design 002 path — see design002.md. Not exercised in Design 001.
        w = self.bone_weights.to(d_pred.device)  # (231,)
        L_dist = (w * (d_pred - d_gt).abs()).mean()
    else:  # 'log'
        # Design 003 path — see design003.md. Not exercised in Design 001.
        eps = self.dist_loss_eps
        L_dist = (torch.log(d_pred + eps) - torch.log(d_gt + eps)).abs().mean()

    losses['loss/dist_matrix/train'] = self.dist_loss_weight * L_dist
```

Constraints:
- The three-branch `if/elif/else` on `self.dist_loss_mode` MUST be written exactly as above. Designs 002 and 003 re-use the same head file skeleton with different config values; the branches `'bone_weighted'` and `'log'` are unreachable in Design 001 but must compile/type-check and MUST NOT raise a `NameError` or `AttributeError`. For Design 001, `self.bone_weights` is NOT created (Design 002 only), so the `'bone_weighted'` branch is unreachable — this is fine because `dist_loss_mode='abs'` is set in `config.py`.
- Key name MUST be `'loss/dist_matrix/train'` (this is the naming convention for all train losses; MMEngine will aggregate it automatically into the epoch loss log).
- The multiply by `self.dist_loss_weight` MUST be applied AFTER computing the raw mean. The Smooth-L1 baseline losses already embed their `loss_weight` inside the module; the distance-matrix loss does not use an MMEngine `MODELS` loss module (it's a direct tensor op), so the weight must be applied explicitly here.
- The `if self.dist_loss_weight > 0.0:` guard means `dist_loss_weight=0.0` (the default) reproduces the baseline loss dict exactly (three keys, no `loss/dist_matrix/train`).
- `torch.cdist` on `(B, 22, 3)` is numerically stable for non-coincident joints. The diagonal `i==i` is NOT included via the `offset=1` argument to `torch.triu_indices`, so zero-distance diagonal entries never appear in the loss. This avoids the well-known `torch.cdist` gradient-at-zero issue.
- `torch.triu_indices(22, 22, offset=1, device=pred_body.device)` returns a `(2, 231)` long tensor on the same device as `pred_body`, avoiding any host↔device transfer inside the forward pass. It is recomputed on every call (cheap, 231 entries); DO NOT try to cache it as a buffer in Design 001 (keeps the diff minimal).
- `d_pred` and `d_gt` are both shape `(B, 231)`; `(d_pred - d_gt).abs().mean()` averages over both batch and pair dims, producing a single scalar. Do NOT sum — use `.mean()` — so that the loss scale does not depend on batch size.

Keep the `with torch.no_grad():` block (train-time MPJPE recording) UNCHANGED. It reads `pred['joints']`, which is unchanged by this design.

#### 5. `predict()` — unchanged

No change. The distance-matrix loss is training-only and does not touch `predict()`.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`, add the three new kwargs at the end (after `loss_weight_uv=1.0,` and before the closing `),`):

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    dist_loss_weight=0.5,
    dist_loss_mode='abs',
    dist_loss_eps=1e-3,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights, `custom_imports` list, dataloaders, evaluators) are identical to the baseline.

---

## Exact Config Values (unchanged from baseline except three head kwargs)

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.03 |
| backbone lr_mult | 0.1 |
| clip_grad max_norm | 1.0 |
| accumulative_counts | 8 (effective batch 32) |
| LR schedule | LinearLR (epoch 0-3, start_factor=0.333) + CosineAnnealingLR (epoch 3-20, eta_min=0), both convert_to_iter_based=True |
| seed | 2026 |
| batch_size | 4 |
| num_workers | 4 |
| persistent_workers | False |
| hidden_dim | 256 |
| num_heads | 8 |
| dropout | 0.1 |
| loss_joints loss_weight | 1.0 |
| loss_depth loss_weight | 1.0 (× loss_weight_depth=1.0) |
| loss_uv loss_weight | 1.0 (× loss_weight_uv=1.0) |
| **dist_loss_weight** | **0.5 (new)** |
| **dist_loss_mode** | **'abs' (new)** |
| **dist_loss_eps** | **1e-3 (new, unused in Design 001)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change (NPZ mmap FD issue).
2. Loss restricted to body joints 0-21 only (`_BODY = list(range(0, 22))`). This applies to BOTH the existing `loss/joints/train` term AND the new `loss/dist_matrix/train` term.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `dist_loss_weight=0.5`, `dist_loss_mode='abs'`, `dist_loss_eps=1e-3` are float/str/float literals.
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package). Do NOT add any new relative imports.
6. `dist_loss_weight` default MUST be `0.0` (so omitting it reproduces baseline behaviour exactly).
7. `dist_loss_mode` default MUST be `'abs'` and must be validated via an assert in `__init__` that rejects values other than `'abs'`, `'bone_weighted'`, `'log'`.
8. The new loss term MUST appear in the `losses` dict with the exact key `'loss/dist_matrix/train'`. Any other key (e.g., `'dist_matrix_loss'`, `'loss_dist'`) will NOT be picked up correctly by `MetricsCSVHook` / `TrainMPJPEAveragingHook` naming conventions.
9. The `torch.cdist` call MUST use `p=2` (Euclidean distance). Using `p=1` or any other norm changes the semantics.
10. `torch.triu_indices(22, 22, offset=1, ...)` MUST use `offset=1`. `offset=0` would include the diagonal (zero distances) and destabilise the gradient.
11. The loss averages over all 231 pairs AND all batch samples using `.mean()` — do NOT use `.sum()`.
12. `forward()` MUST NOT be modified. The distance-matrix term is computed inside `loss()` using only `pred['joints']` and `gt_joints`, both of which are already available in the current baseline `loss()` body.
13. `predict()` MUST NOT be modified. The distance-matrix loss is training-only; inference is unchanged.
14. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, and `BedlamMPJPEMetric` are untouched — they see `pred['joints']` with shape `(B, 70, 3)` just as in the baseline.
15. No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
16. No changes to `pelvis_utils.py`.
17. The head `__init__` signature MUST remain backward compatible. The three new kwargs MUST be keyword-only (they have defaults, so adding them after existing kwargs is safe) and MUST NOT reorder any existing kwargs.
18. No extra learnable parameters. Parameter count is bit-identical to baseline.
19. Memory / throughput: the new `cdist` + gather adds <1% wall-time on 1080 Ti per step. No OOM risk (the two `(B, 22, 22)` float32 tensors are ~7.7 KB per batch).

---

## Expected Behaviour After Change

- `forward()` is identical in compute and output to the baseline. Per-step wall-time overhead from the new loss term: <1 ms on 1080 Ti (negligible vs. the >200 ms backbone forward).
- Training emits FOUR loss scalars per step: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, **`loss/dist_matrix/train`** (new). `MetricsCSVHook` should automatically pick up the new key into its per-epoch log without modification (it enumerates all keys in the aggregated losses dict).
- The auxiliary loss starts moderately positive (untrained joints have random distance-matrix error on the order of 0.3–1.0 m) and decreases monotonically as training progresses. If it does not decrease after epoch 5, the loss is not propagating correctly and the Builder should add a breakpoint to confirm `D_pred.requires_grad == True` and `L_dist.grad_fn is not None`.
- At init (epoch 0 step 0), the three existing losses have their baseline values; the new `loss/dist_matrix/train` is a finite positive scalar (not NaN, not Inf). If NaN, the Builder should verify `pred_body` does not contain NaN (it shouldn't; baseline runs cleanly from step 0) and that `torch.cdist` is not receiving a zero-length tensor.
- Validation metrics (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`) are computed by the unchanged `BedlamMPJPEMetric` on `pred['joints']` — no change to evaluation at all.
- `MetricsCSVHook` writes the same CSV columns as before, plus the new `loss_dist_matrix_train` column (auto-derived from the loss key by the hook's existing naming convention).
- Extra parameter count: **0**.
- Expected result vs. baseline: `mpjpe_body_val` improves (target < 150 mm; baseline is ~165 mm, best prior is 140.96 mm). `mpjpe_pelvis_val` neutral (pelvis pathway untouched). `mpjpe_abs_val` mild positive (indirect benefit from more structurally consistent body). `composite_val` target < 160.
- At inference the behaviour is bit-identical to the baseline (loss branch never executes in `predict()` mode).

---

## Rationale Summary (why λ=0.5, why 'abs', why upper-triangular)

- **λ = 0.5** — in the range [0.1, 1.0] standard for distance-matrix losses in 3D regression. Starting at the midpoint allows future tuning up/down without landing at an extreme. The baseline `loss/joints/train` mean value is approximately 0.08–0.12 m early in training; the new `loss/dist_matrix/train` raw value (before the 0.5 multiplier) is approximately 0.2–0.3 m early in training, so 0.5× yields an additive contribution of the same order of magnitude as the primary coordinate loss — a meaningful signal without dominating.
- **'abs' mode** — simplest scalar mean of |Δdistance|. No per-pair weighting, no log transform. This is the minimal-hyperparameter variant for measuring whether the pairwise signal helps at all. Designs 002 and 003 progressively add structure-aware weighting (bone emphasis) and scale-invariance (log), respectively.
- **Upper-triangular (i<j)** — avoids double-counting every pair twice (the full `(22, 22)` matrix is symmetric with zero diagonal; summing over all 484 cells double-counts the 231 real pairs and adds 22 trivial zeros). Using the upper-triangular variant is mathematically equivalent (up to a factor of 2) and simpler. The `torch.triu_indices(22, 22, offset=1)` call returns the exact 231 index pairs needed.

---

## Risk and Mitigation Specific to Design 001

- **Gradient at coincident joints**: `torch.cdist` with `p=2` has undefined gradient at exactly zero distance (subgradient choice of 0). Mitigation: the diagonal is excluded via `offset=1`; distinct joints in the GT have non-zero pairwise distances (body joints are never coincident in real poses), so predicted distances asymptotically settle to non-zero. Empirically no issue in practice on similar setups.
- **Scale mismatch with baseline losses**: absolute pairwise distances are in [0.05, 1.5] m, similar scale to the per-joint coordinate loss (~0.1 m). Scalar weight 0.5 keeps the auxiliary term from dominating.
- **MMEngine config constraint**: `dist_loss_weight` is a float literal (`0.5`); `dist_loss_mode` is a string literal (`'abs'`); `dist_loss_eps` is a float literal (`1e-3`). No imports required.
- **Eval/inference compatibility**: the loss is training-only. `predict()` is unchanged. `bedlam_metric.py` is unchanged.
- **Interaction with existing ideas**: orthogonal to every prior idea. The distance-matrix loss operates only on `pred['joints'][:, 0:22]`.
- **Pelvis joint at index 0**: after `SubtractRootJoint`, `pred_body[:, 0, :]` is approximately zero (root-relative). Pairs involving joint 0 have distance equal to `||joint_i - 0|| = ||joint_i||`, which is a useful radial signal (distance from pelvis). Not a risk; explicitly included.
- **Device placement**: `iu = torch.triu_indices(22, 22, offset=1, device=pred_body.device)` guarantees the index tensor is on the correct device. Omitting `device=` causes a CPU→GPU copy every step on CUDA; this is a correctness bug-free performance trap and MUST be avoided.
