# Design 002 — Bone-Length-Weighted Pairwise Distance-Matrix Loss (structure-emphasised)

**Design Description:** Same upper-triangular pairwise L1 distance-matrix auxiliary loss as Design 001 (231 body-joint pairs, `dist_loss_weight=0.5`), but multiplied element-wise by a fixed 231-dim bone-weight vector that up-weights the 21 SMPL-X kinematic parent-child edges by factor 2.0 and keeps non-adjacent pairs at weight 1.0. The weight vector is built in `__init__` from a hardcoded SMPL-X parent list and stored as a non-persistent buffer.

**Starting Point:** `baseline/`

---

## Overview

Design 001 treats every one of the 231 upper-triangular body-joint pairs equally. Anatomically, however, not all pairs carry the same structural information:

- **Kinematic (parent-child) bone edges** are direct anatomical constraints — shoulder↔elbow, elbow↔wrist, hip↔knee, knee↔ankle, neck↔head, spine segments, etc. An error here means a literally wrong bone length, which no root translation, rotation, or limb rearrangement can fix.
- **Non-adjacent pairs** (e.g., left-foot ↔ right-hand, head ↔ left-ankle) are much less strongly constrained in natural poses and tolerate larger deviation.

Design 002 encodes this anatomical bias as a fixed `(231,)` weight vector `w` with

```
w[p] = 2.0   if pair p is a (parent, child) edge in the SMPL-X 22-joint body skeleton
w[p] = 1.0   otherwise
```

and the auxiliary loss becomes

```
L_dist = mean_{i<j} w[pair(i,j)] * | ||pred_body[i] - pred_body[j]|| - ||gt_body[i] - gt_body[j]|| |
losses['loss/dist_matrix/train'] = dist_loss_weight * L_dist      # dist_loss_weight = 0.5
```

The scalar weight is unchanged (`0.5`). Only the per-pair weighting changes.

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are identical to the baseline.

---

## BEDLAM2 / SMPL-X 22-Joint Body Skeleton (hardcoded)

The 22 body joints (indices 0–21) follow the standard SMPL-X kinematic tree:

| idx | name           | parent idx |
|-----|----------------|------------|
| 0   | pelvis         | -1 (root)  |
| 1   | left_hip       | 0          |
| 2   | right_hip      | 0          |
| 3   | spine1         | 0          |
| 4   | left_knee      | 1          |
| 5   | right_knee     | 2          |
| 6   | spine2         | 3          |
| 7   | left_ankle     | 4          |
| 8   | right_ankle    | 5          |
| 9   | spine3         | 6          |
| 10  | left_foot      | 7          |
| 11  | right_foot     | 8          |
| 12  | neck           | 9          |
| 13  | left_collar    | 9          |
| 14  | right_collar   | 9          |
| 15  | head           | 12         |
| 16  | left_shoulder  | 13         |
| 17  | right_shoulder | 14         |
| 18  | left_elbow     | 16         |
| 19  | right_elbow    | 17         |
| 20  | left_wrist     | 18         |
| 21  | right_wrist    | 19         |

As a Python list of parent indices (length 22):

```python
BONE_PARENTS_SMPLX_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
```

This yields **21 kinematic bone edges**: one edge per entry where parent is non-negative (i.e., entries at indices 1..21, each connecting child `i` to parent `BONE_PARENTS_SMPLX_22[i]`).

The 21 bone edges as `(parent, child)` pairs (where `i < j` canonicalisation is applied by `min/max`):

```
(0,1), (0,2), (0,3),
(1,4), (2,5), (3,6),
(4,7), (5,8), (6,9),
(7,10), (8,11),
(9,12), (9,13), (9,14),
(12,15),
(13,16), (14,17),
(16,18), (17,19),
(18,20), (19,21)
```

All 21 edges satisfy `parent < child` already, so no `min/max` swap is needed in this specific parent list. The Builder MUST still apply `min/max` defensively (see §4 below) to make the code robust to future parent-list edits.

---

## Files to Change

1. `pose3d_transformer_head.py` — accept the same three new kwargs as Design 001 plus one additional kwarg `bone_parents` (a list of 22 ints); in `__init__`, compute a `(231,)` bone-weight tensor from `bone_parents` and register it as a non-persistent buffer `self.bone_weights`; in `loss()`, take the `'bone_weighted'` branch.
2. `config.py` — add `dist_loss_weight=0.5`, `dist_loss_mode='bone_weighted'`, `dist_loss_eps=1e-3`, and `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]` in the `head=dict(...)` block.
3. `pelvis_utils.py` — **no change**.

No new imports are introduced beyond those already present (`torch`, `torch.nn`).

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `Pose3dTransformerHead.__init__` — new parameters

Add FOUR kwargs to the `__init__` signature, placed immediately after `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`:

```python
dist_loss_weight: float = 0.0,
dist_loss_mode: str = 'abs',
dist_loss_eps: float = 1e-3,
bone_parents: list = None,
```

Store the first three as attributes (identical to Design 001) and validate `dist_loss_mode`:

```python
self.dist_loss_weight = dist_loss_weight
self.dist_loss_mode = dist_loss_mode
self.dist_loss_eps = dist_loss_eps

assert dist_loss_mode in ('abs', 'bone_weighted', 'log'), (
    f"dist_loss_mode must be one of 'abs' | 'bone_weighted' | 'log', "
    f"got {dist_loss_mode!r}")
```

Build the bone-weight buffer **conditionally** on `dist_loss_mode == 'bone_weighted'`. Place this block AFTER the store of `self.dist_loss_eps`:

```python
if dist_loss_mode == 'bone_weighted':
    assert bone_parents is not None and len(bone_parents) == 22, (
        f"dist_loss_mode='bone_weighted' requires bone_parents "
        f"(len-22 list of int), got {bone_parents!r}")

    # Upper-triangular (i<j) pair order for 22 joints: 231 pairs.
    # Build a (22, 22) is_bone mask, then gather upper-tri entries in the
    # same order as torch.triu_indices(22, 22, offset=1).
    is_bone = torch.zeros(22, 22, dtype=torch.bool)
    for child, parent in enumerate(bone_parents):
        if parent < 0:
            continue  # root has no parent edge
        i = min(child, parent)
        j = max(child, parent)
        is_bone[i, j] = True
        is_bone[j, i] = True  # symmetric; the gather uses i<j so only (i,j) is read

    iu = torch.triu_indices(22, 22, offset=1)          # (2, 231)
    is_bone_pairs = is_bone[iu[0], iu[1]]               # (231,) bool

    # Weight: 2.0 for bone edges, 1.0 otherwise.
    bone_weights = torch.where(
        is_bone_pairs,
        torch.full((231,), 2.0, dtype=torch.float32),
        torch.full((231,), 1.0, dtype=torch.float32),
    )
    self.register_buffer('bone_weights', bone_weights, persistent=False)
else:
    # Sentinel so attribute access is well-defined when mode != 'bone_weighted'.
    self.bone_weights = None
```

Constraints:
- The `bone_parents` kwarg default is `None`. When `dist_loss_mode != 'bone_weighted'`, `bone_parents` is IGNORED (it can be omitted from the config). Design 002 supplies a 22-entry list.
- `bone_weights` MUST be registered as a non-persistent buffer (`persistent=False`) — it is derived from config, not learned; no need to be saved/restored with the checkpoint.
- The buffer ordering MUST match the order `torch.triu_indices(22, 22, offset=1)` produces at `loss()` time. Using the same function in both places guarantees identical ordering. DO NOT reconstruct the ordering manually.
- Exactly 21 of the 231 entries should be `2.0`; the remaining 210 should be `1.0`. The Builder can add a one-line sanity assertion after construction: `assert bone_weights.sum().item() == 21 * 2.0 + 210 * 1.0 == 252.0` (optional; helpful during debugging).
- `is_bone[i, j] = is_bone[j, i] = True` is symmetric on purpose so the gather via upper-tri indexing is robust to any child→parent orientation convention.

#### 2. `_init_head_weights` — unchanged

No change. No new learnable parameters are introduced.

#### 3. `forward()` — unchanged

No change. The distance-matrix loss operates on `pred['joints']` in `loss()` only.

#### 4. `loss()` — append auxiliary distance-matrix term (bone_weighted branch)

Identical to Design 001's `loss()` block. Exactly the same three-branch `if/elif/else` on `self.dist_loss_mode`:

```python
# Auxiliary pairwise distance-matrix loss on body joints (indices 0-21).
if self.dist_loss_weight > 0.0:
    pred_body = pred['joints'][:, _BODY]      # (B, 22, 3)
    gt_body = gt_joints[:, _BODY]              # (B, 22, 3)

    D_pred = torch.cdist(pred_body, pred_body, p=2)  # (B, 22, 22)
    D_gt = torch.cdist(gt_body, gt_body, p=2)        # (B, 22, 22)

    iu = torch.triu_indices(22, 22, offset=1, device=pred_body.device)
    d_pred = D_pred[:, iu[0], iu[1]]          # (B, 231)
    d_gt = D_gt[:, iu[0], iu[1]]              # (B, 231)

    if self.dist_loss_mode == 'abs':
        L_dist = (d_pred - d_gt).abs().mean()
    elif self.dist_loss_mode == 'bone_weighted':
        # Design 002: up-weight parent-child bone pairs by 2.0.
        w = self.bone_weights.to(d_pred.device)   # (231,), buffer on correct device already
        L_dist = (w * (d_pred - d_gt).abs()).mean()
    else:  # 'log'
        eps = self.dist_loss_eps
        L_dist = (torch.log(d_pred + eps) - torch.log(d_gt + eps)).abs().mean()

    losses['loss/dist_matrix/train'] = self.dist_loss_weight * L_dist
```

In Design 002 the taken branch is `'bone_weighted'`.

Constraints:
- `w * (d_pred - d_gt).abs()` relies on PyTorch broadcasting: `w` is `(231,)` and `(d_pred - d_gt).abs()` is `(B, 231)`; the result is `(B, 231)`. `.mean()` then averages over both dims, giving a single scalar.
- Note that because 21 pairs have weight 2.0 and 210 have weight 1.0, the **mean** weight across the 231 entries is `(21 * 2.0 + 210 * 1.0) / 231 = 252 / 231 ≈ 1.091`. That is, Design 002's raw `L_dist` is ~1.09× Design 001's raw `L_dist` on a uniformly-random pose. This is a small, deliberate ~9% scale bump on the auxiliary contribution; the scalar `dist_loss_weight=0.5` is kept unchanged so the total pairwise contribution is only ~9% larger than Design 001.
- The `.to(d_pred.device)` call on `self.bone_weights` is defensive: buffers registered in `__init__` move with `model.to(device)`, so under normal mmengine training this `.to` is a no-op. Keep it anyway to protect against CPU-only unit tests.
- DO NOT normalise `w` to have mean 1.0. The raw `w ∈ {1.0, 2.0}` makes the intended interpretation transparent ("bone pairs count twice"), and the slight scale bump is within the tolerance of the scalar weight tuning.
- Key name remains `'loss/dist_matrix/train'` (same as Design 001).

Keep the `with torch.no_grad():` block UNCHANGED.

#### 5. `predict()` — unchanged

No change.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`, add the FOUR new kwargs at the end (after `loss_weight_uv=1.0,`):

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
    dist_loss_mode='bone_weighted',
    dist_loss_eps=1e-3,
    bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
),
```

`bone_parents` is a plain Python list of 22 `int` literals — fully MMEngine-config compliant (no imports required). All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights, `custom_imports` list, dataloaders, evaluators) are identical to the baseline.

---

## Exact Config Values (unchanged from baseline except four head kwargs)

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
| **dist_loss_mode** | **'bone_weighted' (new)** |
| **dist_loss_eps** | **1e-3 (new, unused in mode 'bone_weighted')** |
| **bone_parents** | **[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] (new)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only (`_BODY = list(range(0, 22))`). The bone-weight vector is defined over the 22 body joints only; hand/face joints (22-69) are not involved.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `bone_parents` is a list of 22 int literals (ints only, including the `-1` for the root's parent).
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package). No new imports needed.
6. `dist_loss_weight` default MUST be `0.0` (so omitting it reproduces baseline behaviour exactly).
7. `dist_loss_mode` default MUST be `'abs'` and must be validated in `__init__`.
8. `bone_parents` default MUST be `None`. When `dist_loss_mode == 'bone_weighted'`, a non-None 22-entry list must be provided; otherwise an `AssertionError` with a clear message MUST be raised.
9. The new loss term MUST appear with key `'loss/dist_matrix/train'`.
10. `torch.cdist` MUST use `p=2`. `torch.triu_indices` MUST use `offset=1`.
11. The buffer `self.bone_weights` MUST be registered via `self.register_buffer('bone_weights', ..., persistent=False)` so it moves with `model.to(device)` but is NOT saved into checkpoints (derived from config on each construction).
12. The bone-weight mask ordering MUST be derived from the same `torch.triu_indices(22, 22, offset=1)` call used in `loss()`. Do not enumerate pairs by hand or use a different ordering.
13. `forward()` MUST NOT be modified.
14. `predict()` MUST NOT be modified.
15. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, and `BedlamMPJPEMetric` are untouched.
16. No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
17. No changes to `pelvis_utils.py`.
18. The head `__init__` signature MUST remain backward compatible. New kwargs are keyword-only with defaults and MUST NOT reorder any existing kwargs.
19. No extra *learnable* parameters. The `bone_weights` buffer is fixed (not a `nn.Parameter`). Parameter count is bit-identical to baseline.
20. Exactly 21 of the 231 upper-tri entries should end up with weight 2.0 (one per SMPL-X bone). The Builder can verify via `(bone_weights == 2.0).sum().item() == 21`.

---

## Expected Behaviour After Change

- `forward()` is identical in compute and output to the baseline.
- Training emits FOUR loss scalars: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/dist_matrix/train` (new).
- The auxiliary loss starts positive, similar magnitude to Design 001 (~9% higher due to the re-weighting; see §4 note), and decreases monotonically through training.
- At init, `loss/dist_matrix/train` is a finite positive scalar (not NaN).
- The 21 bone-edge pairs receive 2× gradient compared to the other 210 pairs, so mid-training the head is biased toward getting the 21 most structurally important distances (bone lengths) right first.
- Validation metrics are unchanged in code path — computed by the unchanged `BedlamMPJPEMetric` on `pred['joints']`.
- Extra parameter count: **0**. Extra non-learnable buffer: `(231,)` float32 = 924 bytes.
- Expected result vs. baseline: on top of the general Design 001 gain, bone-length errors specifically should tighten. Likely cleanest win on `mpjpe_body_val`.
- At inference the behaviour is bit-identical to the baseline.

---

## Rationale Summary

- **Why weight = 2.0 on bones?** A factor of 2 is the conservative midpoint of the common range [1.5, 4.0] in structurally-weighted pose losses. Large enough to have measurable effect (each of 21 bones contributes 2× the gradient of a random pair) but small enough that 210 non-bone pairs still contribute meaningful cross-body-structure signal (not ignored). A higher multiplier (e.g., 5×) would effectively ignore non-adjacent pairs and regress to a pure bone-length loss, losing the cross-body signal that was part of Design 001's motivation.
- **Why hardcode parent list rather than learn bone weights?** Anatomical bones are a fixed, known prior. Learning them would require per-pair scalars (231 new params) and would risk the model zeroing them out without enough structural signal. A fixed prior is strictly cheaper and expresses the intent cleanly.
- **Why only 22 body joints?** Hand joints (22–69) are masked out of the joint loss by the `_BODY` slice (baseline convention) and have no GT metric in `BedlamMPJPEMetric`; their predictions are noisy and including them in the distance-matrix loss would add unsupervised gradient to them. The body-only scope keeps the auxiliary signal aligned with the supervised part of the output.

---

## Risk and Mitigation Specific to Design 002

- **Bone parent list correctness**: the 22-entry list above is the standard SMPL-X kinematic tree. The Builder MUST copy it verbatim into `config.py` (see §Config Changes). Any off-by-one or swapped index breaks bone identification. After model construction, the one-line invariant `(model.head.bone_weights == 2.0).sum().item() == 21` is a quick correctness check the Builder can include in a local debug run before SLURM submission.
- **Interaction with Design 001**: Designs 001 and 002 use the same three-branch `loss()` body. Mode selection is by string. If a future design wants to sweep bone weights (e.g., try 1.5 or 3.0), it can use a new kwarg rather than editing existing code.
- **MMEngine config constraint**: `bone_parents` is a list of ints; `dist_loss_mode` is a string; `dist_loss_weight`, `dist_loss_eps` are floats. All are plain Python literals — fully MMEngine-config-compliant.
- **Memory / speed**: a `(231,)` float32 buffer is negligible. The per-step overhead is a single `231` broadcast multiply, sub-microsecond.
- **Per-head / per-layer broadcasting**: none — `w` is 1-D `(231,)` and broadcasts cleanly against `(B, 231)`.
- **Device placement of buffer**: `register_buffer(..., persistent=False)` ensures the buffer is moved by `model.to(device)`. The `.to(d_pred.device)` at loss-time is defensive and cheap.
- **Interaction with existing ideas**: orthogonal to every prior idea; composes cleanly.
- **No NaN risk**: `torch.cdist` with distinct joints is safe; diagonal excluded via `offset=1`; no log or division by zero.
