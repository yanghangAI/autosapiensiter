# Design 002 — Bone-Vector Head + Auxiliary Bone-Length Loss (explicit bone-length prior)

**Design Description:** Same bone-vector (kinematic-chain) output head as Design 001 (22 body joints reparameterized as cumulative sums along the SMPL-X parent chain with `1/sqrt(21)` weight scale-init), plus an auxiliary L1 loss on the *magnitudes* of the 21 predicted bone vectors vs. the magnitudes of the 21 GT bone vectors, with `bone_length_loss_weight=0.3`. The direction of each bone remains free; only the magnitude (bone length) is explicitly regularised by the auxiliary term.

**Starting Point:** `baseline/`

---

## Overview

Design 001 introduces the kinematic-chain output parameterization but applies no explicit bone-length prior — all supervision comes from the per-joint Smooth-L1 loss on recovered root-relative coordinates. Because BEDLAM2 uses a small number of rigged SMPL-X skeletons with tightly concentrated bone lengths (nearly subject-independent within the dataset), an **explicit scalar bone-length prior** can provide a low-variance, easy-to-extract signal that accelerates learning of the bone-length distribution.

Design 002 adds a single auxiliary term on top of Design 001:

```
L_bone_len = mean over i=1..21 of | ||pred_bone_i||₂ - ||gt_bone_i||₂ |
losses['loss/bone_length/train'] = bone_length_loss_weight * L_bone_len    # weight = 0.3
```

where `pred_bone_i = pred_body_rr[child_i] - pred_body_rr[parent_i]` and `gt_bone_i = gt_body[child_i] - gt_body[parent_i]`. The direction of each bone is NOT penalised by this term — only its Euclidean norm (length). The primary joint-coordinate loss continues to supervise direction.

All architecture (backbone, decoder layer, queries, pelvis depth/UV, joints_out single-head), optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are identical to the baseline — except for the three head kwargs flipped from Design 001 and the addition of `bone_length_loss_weight=0.3`.

---

## BEDLAM2 / SMPL-X 22-Joint Body Skeleton (hardcoded)

Identical to Design 001. The parent list `BONE_PARENTS_SMPLX_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]` is used for both (a) forward kinematics in `forward()` and (b) computation of GT and predicted bone vectors in the auxiliary loss.

The 21 bones (child, parent) pairs used by the auxiliary loss:

```
(1, 0), (2, 0), (3, 0),
(4, 1), (5, 2), (6, 3),
(7, 4), (8, 5), (9, 6),
(10, 7), (11, 8),
(12, 9), (13, 9), (14, 9),
(15, 12),
(16, 13), (17, 14),
(18, 16), (19, 17),
(20, 18), (21, 19)
```

Bone `i ∈ {1..21}` is always identified by its **child index** — i.e., bone 4 is the left-hip→left-knee bone. This is the same convention as the `bone_parents` list (index into `bone_parents` gives the parent of that joint).

---

## Files to Change

1. `pose3d_transformer_head.py` — identical signature, buffer registration, `_init_head_weights` scaling, `forward()` kinematic recovery, and `_forward_kinematics` method as Design 001. The ONLY extra behaviour vs. Design 001 is that the `bone_length_loss_weight` path in `loss()` is now active (because `bone_length_loss_weight=0.3 > 0.0` in this design's config).
2. `config.py` — same structure as Design 001 with `bone_length_loss_weight=0.3` instead of `0.0`.
3. `pelvis_utils.py` — **no change**.

No new imports are introduced beyond those in Design 001 (`torch`, `torch.nn`, `math` — all already present).

---

## Algorithm Changes

### `pose3d_transformer_head.py`

All edits from Design 001 apply verbatim (new `__init__` kwargs, scale-init, `_forward_kinematics`, `forward()` kinematic recovery). **Design 002 does not re-describe them — see design001.md for the detailed spec of those shared components.** The only design-specific change is that the bone-length loss block inside `loss()` is now exercised (the `bone_length_loss_weight > 0.0` guard is True).

For completeness, the exact bone-length loss block that MUST appear in `loss()` — AFTER the three existing loss assignments and BEFORE the `with torch.no_grad():` MPJPE block — is repeated here:

```python
# Auxiliary bone-length loss on body joints (indices 1..21 = 21 bones).
if self.kinematic_parametrization and self.bone_length_loss_weight > 0.0:
    # Indices into the 22-body tensor.
    # child_idx: [1, 2, ..., 21]; parent_idx: bone_parents[1..21]
    device = pred['joints'].device
    child_idx = torch.arange(1, 22, device=device)          # (21,)
    parent_idx = self.bone_parents[1:22].to(device)          # (21,) long

    gt_body = gt_joints[:, _BODY]                            # (B, 22, 3)
    gt_bones = gt_body[:, child_idx, :] - gt_body[:, parent_idx, :]      # (B, 21, 3)

    pred_body = pred['joints'][:, _BODY]                     # (B, 22, 3)
    pred_bones = pred_body[:, child_idx, :] - pred_body[:, parent_idx, :]  # (B, 21, 3)

    # L1 on magnitudes only.
    gt_bone_len = gt_bones.norm(dim=-1)                      # (B, 21)
    pred_bone_len = pred_bones.norm(dim=-1)                  # (B, 21)

    L_bone_len = (pred_bone_len - gt_bone_len).abs().mean()
    losses['loss/bone_length/train'] = self.bone_length_loss_weight * L_bone_len
```

Constraints:
- `child_idx` is `torch.arange(1, 22, device=device)` — length 21, values `1..21`. The device MUST match `pred['joints'].device` to avoid implicit CPU→GPU transfers inside the fancy-index operation.
- `parent_idx = self.bone_parents[1:22]` — slice out the 21 parent indices, skipping the root sentinel at index 0 (which is `-1` and would cause an out-of-bounds index if used).
- The `.to(device)` on `parent_idx` is defensive — `self.bone_parents` is a registered buffer that already moves with `model.to(device)`, so in the normal training path the `.to(device)` is a no-op. Keep it for CPU-only unit-test safety.
- `pred_bones` is computed from the **recovered** joint coordinates (the output of `forward_kinematics`), NOT from the raw bone-vec head output. Mathematically these are equivalent (because `body_rr[child] - body_rr[parent] = bone_vec[child]` by construction under the cumulative-sum recovery), but computing from recovered coords keeps the loss code independent of any refactor to the kinematic recovery path (e.g., if Design 003's per-limb heads produce `bone_vecs` via a different code path, the loss still reads from `pred['joints']` and works uniformly).
- Use `.norm(dim=-1)` (Euclidean / L2 norm over the last dim) to compute the 3-vector magnitude. Do NOT use `.norm(dim=-1, keepdim=True)` — we want `(B, 21)` not `(B, 21, 1)`.
- `.abs().mean()` produces a scalar, averaging over both batch and bone dims.
- Multiply by `self.bone_length_loss_weight` explicitly (there is no separate `nn.Module` loss object for this term).
- Key name MUST be `'loss/bone_length/train'`. This matches the project's loss-key naming convention (`loss/<name>/<split>`) so `MetricsCSVHook` auto-records it.

The `with torch.no_grad():` block is **UNCHANGED** from baseline. `self._train_mpjpe` reads `pred['joints'][:, _BODY]` just as before — the recovered coordinates.

#### Edge case: the guard

The guard `if self.kinematic_parametrization and self.bone_length_loss_weight > 0.0:` means that:
- Design 001 (`bone_length_loss_weight=0.0`) SKIPS this block — `loss/bone_length/train` is NOT added to the loss dict.
- Design 002 (`bone_length_loss_weight=0.3`) EXECUTES this block — `loss/bone_length/train` IS added.
- Baseline (`kinematic_parametrization=False`) SKIPS this block — regardless of `bone_length_loss_weight` (defensive: bone_length loss only makes sense with kinematic parametrization).

### `predict()` — unchanged

No change.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`, same kwargs as Design 001 but with `bone_length_loss_weight=0.3`:

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
    kinematic_parametrization=True,
    bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    bone_length_loss_weight=0.3,
    per_limb_heads=False,
    limb_index=None,
),
```

All other config values are identical to the baseline.

---

## Exact Config Values (unchanged from baseline except five head kwargs)

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
| **kinematic_parametrization** | **True (new)** |
| **bone_parents** | **[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] (new)** |
| **bone_length_loss_weight** | **0.3 (new)** |
| **per_limb_heads** | **False (new; disabled in Design 002)** |
| **limb_index** | **None (new; disabled in Design 002)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

All constraints 1–24 from Design 001 apply verbatim. Additional constraints specific to Design 002:

25. `bone_length_loss_weight=0.3` MUST be stored as `self.bone_length_loss_weight` and referenced in the `loss()` guard.
26. The auxiliary loss key MUST be exactly `'loss/bone_length/train'`. Any other key (e.g., `'loss/bone_len/train'`, `'loss_bone_length'`) will NOT be picked up correctly by `MetricsCSVHook` / `TrainMPJPEAveragingHook` naming conventions.
27. The auxiliary loss MUST use `(pred_bone_len - gt_bone_len).abs().mean()` — plain L1 on magnitudes. Do NOT use `mse_loss`, `smooth_l1`, or weighted variants; the simplest L1 is intentional to match the Design description precisely.
28. `child_idx` and `parent_idx` MUST be computed via `torch.arange` and slice of `self.bone_parents`, respectively, using `device=pred['joints'].device` to avoid host↔device transfers every step.
29. The auxiliary loss computation MUST read from `pred['joints']` (recovered coordinates), NOT from any intermediate `bone_vecs` tensor. This keeps the block self-contained and uniform across Designs 001/002/003.
30. The bone-length loss is training-only. `predict()` is NOT modified.
31. Parameter count is **bit-identical** to the baseline. No new learnable parameters are added by the bone-length term. (The computation is a tensor op with no new nn.Module.)

---

## Expected Behaviour After Change

- `forward()` behaviour is identical to Design 001.
- Training emits FOUR loss scalars per step: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/bone_length/train` (new).
- At init (epoch 0 step 0), `loss/bone_length/train` is a finite positive scalar. Raw magnitude: GT bone lengths are in `[0.05, 0.55]` m (typical SMPL-X); predicted bone-vec magnitudes at init are on order of `sqrt(3) * 0.02 * 1/sqrt(21)` ≈ 0.0076 m under the scale-init; the difference `|pred - gt|` averages around 0.2 m. With the `0.3` weight, the initial contribution to the total loss is ~0.06 — meaningful but not dominating (compare to joint loss ~0.1 m).
- Over training, `loss/bone_length/train` should decrease monotonically. By epoch 5, it should be < 0.02 m (i.e., predicted bone lengths within 2 cm of GT on average).
- Validation metrics use the unchanged `BedlamMPJPEMetric` on `pred['joints']`.
- Extra parameter count: **0**. Extra non-learnable buffer: `bone_parents` (22 int64 = 176 bytes, same as Design 001). Extra per-step compute: one L1 over `(B, 21)` scalars = sub-microsecond.
- Expected result vs. Design 001: small positive delta on `mpjpe_body_val` (target: ≤ Design 001's body MPJPE, ideally 1–3 mm better). Neutral elsewhere. The bone-length prior is a weak but cheap regulariser that should not hurt. If it **does** hurt (e.g., Design 002 produces worse body MPJPE than Design 001), that result itself is informative: it would mean the primary coordinate loss already absorbs bone-length information and the auxiliary term acts as a constraint overfitting against it. The scalar `0.3` is the moderate starting point; a future design could tune up/down.
- At inference the behaviour is bit-identical to Design 001.

---

## Rationale Summary

- **Why weight = 0.3?** The bone-length L1 raw value is ~0.05 m late in training (when predicted bones approach GT lengths). Joint L1 is on order ~0.05–0.10 m late in training. A weight of 0.3 gives the auxiliary an effective contribution of ~0.015 m — roughly 15% of the primary loss — enough to have a measurable pull without dominating. The range `[0.1, 0.5]` is standard for this kind of prior; 0.3 is the midpoint.
- **Why L1 on magnitudes only (not 3-vec L1)?** A 3-vector L1 on bone vectors is almost exactly equivalent to the existing per-joint Smooth-L1 on recovered coordinates (up to the Smooth-L1 vs. L1 difference and the parent-child pairing), so it would be redundant. The magnitude-only variant is **orthogonal information**: the direction is already supervised by the primary loss, while the magnitude adds a **scalar prior** that the primary loss only indirectly constrains (because the primary loss couples magnitude + direction).
- **Why not L2 on magnitudes?** L1 is robust to occasional outlier poses (e.g., a rare crouched frame where SubtractRootJoint happens to produce a jittery relative position). L2 would over-penalise such outliers and bias toward the mean bone length too strongly.
- **Why on body bones only?** Hands are not subject to the kinematic parametrization (no hand forward-kinematics in this design), and hand joints are not even supervised by the primary `loss/joints/train` term (which restricts to `_BODY`). An auxiliary bone-length term for hands would add noise without a corresponding supervised signal.

---

## Risk and Mitigation Specific to Design 002

- **Bone-length over-regularisation**: with weight 0.3, the auxiliary could pull predictions toward the *average* bone length across the training batch, dampening legitimate subject-to-subject bone-length variation. Mitigation: BEDLAM2 uses a small number of SMPL-X skeletons with narrow bone-length distribution (synthetic, rigged), so this is low-risk in practice. If Design 002 underperforms Design 001, a follow-up design can drop the weight to 0.1 or remove the term entirely.
- **Double-counting information**: the primary loss already penalises per-joint coordinate errors, which implicitly penalises bone-length errors. The bone-length auxiliary is only informative to the extent that the primary loss under-weights magnitude vs. direction. This is a small, controlled addition; the 0.3 weight prevents dominance.
- **Per-iteration device transfer of `parent_idx`**: `self.bone_parents` is a registered CUDA-resident buffer after `model.to(device)`, so `self.bone_parents[1:22]` is already on GPU; the `.to(device)` call is a no-op. Kept defensively for CPU-only tests.
- **Magnitude at zero**: if two joints ever coincide in a prediction (`pred_bones[..., i] = 0`), `.norm(dim=-1)` returns 0 with gradient 0. The |0 - gt_len| term still has a valid gradient for `gt_len > 0`. No NaN risk.
- **Predict path**: unchanged — auxiliary loss is training-only.
- **MMEngine config constraint**: all new kwargs are plain Python literals.
- **Interaction with Design 001**: Design 002 reuses exactly the Design 001 code path with `bone_length_loss_weight` flipped from `0.0` to `0.3`. A single head-file implementation serves both.
- **Interaction with Design 003**: Design 003 uses `per_limb_heads=True` in addition to the kinematic parametrization. Its `loss()` block is identical to Designs 001/002; the per-limb heads modify only `__init__` and `forward()` (how bone_vecs are produced), not the loss semantics. Design 003 sets `bone_length_loss_weight=0.0` (no bone-length auxiliary) to isolate the effect of the per-limb architectural change.
- **Memory / speed**: auxiliary adds one `(B, 21, 3)` tensor subtraction, two `.norm(dim=-1)` calls, and a single `.abs().mean()`. Total <50 μs per step on 1080 Ti — negligible.
