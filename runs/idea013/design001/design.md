# Design 001 — Bone-Vector (Kinematic-Chain) Output Head (minimal)

**Design Description:** Reparameterize the 22 body joints at the output head so that `joints_out` emits 22 *bone-translation vectors* (with the root forced to zero) and the root-relative body joint positions are recovered by forward kinematics (cumulative sum along the SMPL-X parent chain). The hand joints (indices 22..69), pelvis depth, and pelvis UV pathways are unchanged. The output head weight is scale-initialised by `1/sqrt(21) ≈ 0.21821789` so the recovered-joint variance at init matches the baseline's direct-regression variance.

**Starting Point:** `baseline/`

---

## Overview

The baseline head regresses each body joint's root-relative 3D coordinate independently via a single shared `joints_out: Linear(hidden_dim, 3)` applied per token. Under this design, the same `Linear(hidden_dim, 3)` is retained but its per-token output is re-interpreted for body joints only (indices 0..21):

- Output index 0 (pelvis / root): forced to zero by construction (the root bone is not predicted).
- Output index i (i=1..21): `bone_vec[i] ∈ R^3` — the 3D translation from joint `parent[i]` to joint `i`.

Absolute root-relative body joint positions are recovered by forward kinematics (cumulative sum along the parent chain):

```
body_rr[0]   = 0
body_rr[i]   = body_rr[parent[i]] + bone_vec[i]      for i = 1..21
```

Hand joints (indices 22..69) are unchanged — still direct root-relative coordinate regression out of the same `joints_out` head.

The loss is unchanged in form: `smooth_l1(pred['joints'][:, 0:22], gt_joints[:, 0:22])` plus the unchanged depth/UV losses. Because the forward-kinematics transform is a differentiable, bijective reparameterization, gradient is well-defined and the supervised signal on each body joint flows back through every ancestor bone vector on its chain to the root.

All architecture (backbone, decoder layer, queries, pelvis depth/UV), optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are identical to the baseline.

---

## BEDLAM2 / SMPL-X 22-Joint Body Skeleton (hardcoded)

The 22 body joints follow the standard SMPL-X kinematic tree (identical to idea012/design002):

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

Notation invariant (guaranteed by construction of this list):
- `parent[0] = -1` (root has no parent).
- For every `i ∈ {1,…,21}`, `parent[i] < i`. This is the **critical** property that makes a single in-order `for i in range(1, 22)` forward-kinematics loop correct: when we process child `i`, `body_rr[parent[i]]` has already been written.

The Builder MUST add a defensive assertion in `__init__` that verifies this topological property:

```python
for child in range(1, 22):
    assert bone_parents[child] < child, (
        f"bone_parents[{child}]={bone_parents[child]} must satisfy "
        f"bone_parents[child] < child for the in-order forward-kinematics "
        f"loop to be correct.")
```

---

## Files to Change

1. `pose3d_transformer_head.py` — add the new kwargs, store `bone_parents` as a non-persistent int buffer, scale-init the `joints_out` weight when kinematic mode is enabled, and apply `forward_kinematics` to the body slice of `joints` inside `forward()` before returning.
2. `config.py` — add the new head kwargs.
3. `pelvis_utils.py` — **no change**.

No new imports are introduced beyond those already present (`torch`, `torch.nn`, `math`). The `math.sqrt` for the init scale can reuse the already-imported `math` module.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `Pose3dTransformerHead.__init__` — new parameters

Add FOUR kwargs to the `__init__` signature, placed immediately after `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`:

```python
kinematic_parametrization: bool = False,
bone_parents: list = None,
bone_length_loss_weight: float = 0.0,
per_limb_heads: bool = False,
```

Also accept `limb_index: list = None` as the LAST new kwarg (used only by Design 003; must be accepted with default `None` in Designs 001 and 002 so the signature is shared across all three designs of idea013):

```python
kinematic_parametrization: bool = False,
bone_parents: list = None,
bone_length_loss_weight: float = 0.0,
per_limb_heads: bool = False,
limb_index: list = None,
```

Store them as attributes. Place the block immediately after the existing `self.loss_weight_uv = loss_weight_uv` line:

```python
self.kinematic_parametrization = kinematic_parametrization
self.bone_length_loss_weight = bone_length_loss_weight
self.per_limb_heads = per_limb_heads
```

When `kinematic_parametrization` is True, validate and register `bone_parents` as a non-persistent long-tensor buffer; also validate the topological ordering `parent[child] < child`:

```python
if kinematic_parametrization:
    assert bone_parents is not None and len(bone_parents) == 22, (
        f"kinematic_parametrization=True requires bone_parents "
        f"(len-22 list of int), got {bone_parents!r}")
    assert bone_parents[0] == -1, (
        f"bone_parents[0] must be -1 (root), got {bone_parents[0]}")
    for child in range(1, 22):
        assert 0 <= bone_parents[child] < child, (
            f"bone_parents[{child}]={bone_parents[child]} must satisfy "
            f"0 <= p < child for a valid topologically-ordered kinematic tree.")
    self.register_buffer(
        'bone_parents',
        torch.tensor(bone_parents, dtype=torch.long),
        persistent=False)
else:
    self.bone_parents = None
```

Design 001 sets `kinematic_parametrization=True`, `bone_parents=[-1, 0, ..., 19]`, `bone_length_loss_weight=0.0`, `per_limb_heads=False`, `limb_index=None`.

Constraints:
- `bone_parents` is registered as a **non-persistent** buffer (`persistent=False`): derived from config; no need to save/restore with checkpoint.
- `dtype=torch.long` is required because the buffer is used as an index tensor in `forward_kinematics`.
- `bone_parents[0]` is `-1`, which is a **sentinel**; it must NEVER be used as an actual index. The forward-kinematics loop (§3 below) starts at child index `1` and skips the root.
- `per_limb_heads=True` is a Design 003 option only; Design 001 keeps the default `False`.
- `limb_index=None` is the default; Design 001 does not use it.

#### 2. `_init_head_weights` — scale-init the body portion of joints_out

Currently `_init_head_weights` initialises `self.joints_out.weight` with trunc-normal `std=0.02`. Under kinematic parametrization, cumulative sum of 21 approximately-i.i.d. `bone_vec` predictions produces a recovered-joint variance of `21 * Var(bone_vec)`. To keep the initial recovered-joint magnitude equal to baseline's, we scale the body portion of `joints_out.weight` by `1/sqrt(21)`.

In the current baseline, `joints_out` is a single `Linear(hidden_dim, 3)` applied to every token — there is no separate body slice inside the layer itself; the slicing happens at the token level. Since the 22 body queries (indices 0..21) and 48 hand queries (indices 22..69) go through the **same** linear layer, we CANNOT selectively scale only the weight rows for body tokens (there are no such rows — the 3 output dims are per-token regardless of which query).

Therefore the scale-init MUST be applied to the entire `joints_out.weight` matrix. This slightly shrinks the init scale of the hand direct-regression predictions too, by `1/sqrt(21)`. This is **acceptable**: the hand regression is from exactly the same shared hidden features and is not part of the supervised body loss; its early-training trajectory is unaffected by a 4.6× smaller init (empirically robust in the baseline because hand outputs have no supervised signal anyway — see rationale below).

Explicit rule for `_init_head_weights`: after the existing trunc-normal init (`std=0.02`) on `self.joints_out.weight`, multiply it in-place by `1/sqrt(num_body_bones)` where `num_body_bones = 21` (or equivalently `len(bone_parents) - 1`). Do this only when `self.kinematic_parametrization` is True:

```python
def _init_head_weights(self) -> None:
    # Query embeddings
    nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
    # Output projections
    for m in [self.joints_out, self.depth_out, self.uv_out]:
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # Scale-init body bone-vec head to keep recovered-joint variance
    # comparable to baseline's direct-regression variance after the
    # cumulative-sum forward-kinematics transform.
    if self.kinematic_parametrization:
        num_body_bones = 21  # 22 body joints - 1 root
        scale = 1.0 / math.sqrt(num_body_bones)
        with torch.no_grad():
            self.joints_out.weight.mul_(scale)
```

Constraints:
- The `with torch.no_grad()` context is required because we're modifying a `Parameter` in-place without tracking gradients.
- `self.joints_out.bias` is already zero from the existing init — do NOT scale it (it is already at the correct value after scaling).
- The `Linear(hidden_dim, 3)` has a tiny weight (`256 × 3 = 768`) so an in-place `.mul_` is sub-microsecond.
- Design 003 will override this behaviour (per-limb heads); see design003.md.

#### 3. `forward()` — apply forward kinematics on the body slice

Modify `forward()` AFTER the `joints = self.joints_out(decoded)` line and BEFORE the pelvis depth/UV computations. Insert the kinematic-recovery block:

```python
joints = self.joints_out(decoded)  # (B, num_joints, 3)

if self.kinematic_parametrization:
    # Interpret the first 22 entries as bone-translation vectors
    # (entry 0 is the root — forced to zero — and entries 1..21 are
    # bone vectors from parent[i] -> child[i] in root-relative space).
    body_bone_vecs = joints[:, 0:22, :]                     # (B, 22, 3)
    hand_coords   = joints[:, 22:self.num_joints, :]        # (B, 48, 3)
    body_rr = self._forward_kinematics(body_bone_vecs)      # (B, 22, 3)
    joints = torch.cat([body_rr, hand_coords], dim=1)       # (B, num_joints, 3)

pelvis_token = decoded[:, 0, :]  # (B, hidden_dim)
# ... unchanged ...
```

Implement `_forward_kinematics` as a method on `Pose3dTransformerHead`:

```python
def _forward_kinematics(self, bone_vecs: torch.Tensor) -> torch.Tensor:
    """Recover root-relative body joint positions from bone-translation
    vectors via cumulative sum along the parent chain.

    Args:
        bone_vecs: Tensor of shape (B, 22, 3). Entry [:, 0, :] is the
            root and is ignored (overwritten with zero). Entries
            [:, 1..21, :] are interpreted as bone vectors from
            parent[i] to joint i.

    Returns:
        Tensor of shape (B, 22, 3) with root-relative joint positions.
        The output at index 0 is exactly the zero vector.
    """
    # Clone to avoid in-place mutation of the upstream tensor (would
    # break autograd on a view of `joints`). Zero the root.
    body_rr = bone_vecs.clone()
    body_rr[:, 0, :] = 0.0
    parents = self.bone_parents  # (22,) long on the correct device
    for child in range(1, 22):
        parent = int(parents[child].item())
        body_rr[:, child, :] = body_rr[:, parent, :] + bone_vecs[:, child, :]
    return body_rr
```

Constraints:
- The loop is `for child in range(1, 22)` — exactly 21 iterations. This is a constant (not batch-dependent), tiny overhead.
- `int(parents[child].item())` MUST be used to pull the parent index as a Python int for use in the slicing. (PyTorch does NOT allow tensor-valued indices at arbitrary positions inside a slice expression like `body_rr[:, parent, :]`; Python int indexing is required.)
- This `.item()` call creates a host↔device sync (one per iteration = 21 syncs per forward). On 1080 Ti this is ~5 μs per sync = ~0.1 ms per forward — negligible relative to the ~250 ms backbone forward. Alternative approach: keep a Python list of ints (e.g., `self._bone_parents_list = list(bone_parents)`) attached as a non-buffer attribute in `__init__` and iterate over that instead to avoid the syncs. **Use this alternative** (set `self._bone_parents_list = list(bone_parents)` in `__init__` when `kinematic_parametrization=True`, and in `_forward_kinematics` do `parent = self._bone_parents_list[child]`). The buffer `self.bone_parents` is kept for serialisation/debugging.
- The `bone_vecs.clone()` is critical: without it, `body_rr[:, child, :] = body_rr[:, parent, :] + bone_vecs[:, child, :]` would silently write back into `joints` (a view) and the next iteration could use stale values. The clone ensures we accumulate into a fresh tensor.
- The `body_rr[:, 0, :] = 0.0` write is **in-place on the cloned tensor**, not on the original. This preserves autograd because the clone is a new tensor.
- The output is `(B, 22, 3)` and differentiable with respect to `bone_vecs`.

#### 4. `loss()` — unchanged shape, add optional bone-length term

The main joint loss reads `pred['joints'][:, _BODY]` — which, after forward(), is now the **recovered** (root-relative) body-joint tensor. So the baseline loss line is unchanged:

```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

For Design 001 (no bone-length loss), do NOT add any new term. For Designs 002/003 the following optional block appears (included here only for shared-file compatibility). In `loss()`, AFTER the three existing loss assignments and BEFORE the `with torch.no_grad():` MPJPE block, insert:

```python
# Auxiliary bone-length loss on body joints.
if self.kinematic_parametrization and self.bone_length_loss_weight > 0.0:
    # GT bone vectors: child - parent in root-relative space.
    # Indices skip the root.
    child_idx = torch.arange(1, 22, device=pred['joints'].device)
    parent_idx = self.bone_parents[1:22]  # (21,) long
    gt_body = gt_joints[:, _BODY]         # (B, 22, 3)
    gt_bones = gt_body[:, child_idx, :] - gt_body[:, parent_idx, :]   # (B, 21, 3)
    # Predicted bone vectors: these are exactly the raw joints_out for
    # indices 1..21 under kinematic parametrization.
    # Recover them from the (already-recovered) joint positions:
    pred_body = pred['joints'][:, _BODY]
    pred_bones = pred_body[:, child_idx, :] - pred_body[:, parent_idx, :]  # (B, 21, 3)
    gt_bone_len = gt_bones.norm(dim=-1)       # (B, 21)
    pred_bone_len = pred_bones.norm(dim=-1)   # (B, 21)
    L_bone_len = (pred_bone_len - gt_bone_len).abs().mean()
    losses['loss/bone_length/train'] = self.bone_length_loss_weight * L_bone_len
```

In Design 001, `self.bone_length_loss_weight == 0.0` (the default) — the entire block is skipped by the `> 0.0` guard and no new loss key is added.

Keep the `with torch.no_grad():` block UNCHANGED. `self._train_mpjpe` reads `pred['joints'][:, _BODY]` which is already the recovered joint tensor after `forward()`.

#### 5. `predict()` — unchanged

No change. `forward()` is called inside `predict()` and already applies `forward_kinematics`, so the returned `pred['joints']` is the recovered coordinate tensor — identical shape `(B, 70, 3)` to baseline.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`, add the new kwargs at the end (after `loss_weight_uv=1.0,`):

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
    bone_length_loss_weight=0.0,
    per_limb_heads=False,
    limb_index=None,
),
```

`bone_parents` is a plain Python list of 22 `int` literals (including `-1` for the root sentinel) — fully MMEngine-config compliant (no imports needed). `limb_index=None` is a literal `None` (not used by Design 001 but included for signature uniformity). All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights, `custom_imports` list, dataloaders, evaluators) are identical to the baseline.

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
| **bone_length_loss_weight** | **0.0 (new; disabled in Design 001)** |
| **per_limb_heads** | **False (new; disabled in Design 001)** |
| **limb_index** | **None (new; disabled in Design 001)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change (NPZ mmap FD issue).
2. Loss restricted to body joints 0-21 only for `loss/joints/train` (`_BODY = list(range(0, 22))`). Unchanged from baseline.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `kinematic_parametrization`, `bone_length_loss_weight`, `per_limb_heads` are bool/float literals. `bone_parents` is a list of 22 int literals. `limb_index` is `None` literal.
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package). Do NOT add any new relative imports. The existing `import math` already provides `math.sqrt`; no new top-level imports are needed.
6. `kinematic_parametrization` default MUST be `False` (so omitting it reproduces baseline behaviour exactly, bit-for-bit).
7. `bone_parents` default MUST be `None`. When `kinematic_parametrization=True`, a non-None 22-entry list MUST be provided; otherwise `AssertionError` with a clear message MUST be raised.
8. `bone_parents` MUST satisfy `parent[0] == -1` and `0 <= parent[child] < child` for all `child ∈ {1..21}`. These invariants MUST be asserted in `__init__`.
9. The internal `self._bone_parents_list` (or equivalent) MUST be a Python `list[int]` so the `_forward_kinematics` loop uses host-side indexing (no per-iteration device syncs).
10. The `forward_kinematics` loop MUST process children in **topological order** — the simple `for child in range(1, 22)` works correctly iff `parent[child] < child` for all `child` (guaranteed by constraint 8).
11. `_forward_kinematics` MUST clone its input tensor BEFORE writing into it — writing into a view of `joints` would corrupt autograd. The clone is a new leaf in the autograd graph; gradient flow is preserved from `body_rr` back to `bone_vecs`.
12. `_forward_kinematics` MUST overwrite the root slot `body_rr[:, 0, :] = 0.0` so the recovered root is *exactly* zero regardless of what the model predicted for index 0. This also zeroes any bias/drift from the raw bone-vec head at the root.
13. The scale-init multiplication `joints_out.weight.mul_(1/sqrt(21))` MUST be applied ONLY when `kinematic_parametrization=True` and MUST be wrapped in `torch.no_grad()`.
14. `joints_out.bias` MUST remain at zero (unchanged from baseline init).
15. `forward()` MUST apply `forward_kinematics` on the body slice `[0:22]` and pass the hand slice `[22:num_joints]` through unchanged. Concatenation MUST happen along `dim=1` (the joint dimension) to produce a `(B, num_joints, 3)` tensor.
16. The concatenation order is `[body_rr, hand_coords]` — same as the original joint-index order. Do NOT reorder.
17. `predict()` MUST NOT be modified. It reads `pred['joints']` which is already the kinematically-recovered tensor.
18. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, and `BedlamMPJPEMetric` are untouched — they see `pred['joints']` with shape `(B, 70, 3)` just as in the baseline.
19. No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
20. No changes to `pelvis_utils.py`.
21. The head `__init__` signature MUST remain backward compatible. The new kwargs MUST be keyword-only (they have defaults, so adding them after existing kwargs is safe) and MUST NOT reorder any existing kwargs.
22. Parameter count is **bit-identical** to the baseline. No new learnable parameters are added by Design 001. (The `bone_parents` buffer is not a `Parameter`.)
23. Pelvis depth/UV pathway: unchanged. `depth_out` and `uv_out` still take `decoded[:, 0, :]` — the pelvis-token's raw decoder embedding — NOT the recovered coordinate. This is important: the pelvis depth/UV is predicted from the same transformer token-0 embedding as in the baseline; the kinematic reparameterization is only about the joint-coordinate output semantics.
24. The loss `smooth_l1(pred['joints'][:, _BODY], gt_joints[:, _BODY])` remains unchanged in *value semantics* — it compares recovered joint positions to GT joint positions. Because forward_kinematics is a bijection with zero additive offset at the root, `pred['joints'][:, 0, :]` is exactly zero, matching `gt_joints[:, 0, :]` which is also zero after `SubtractRootJoint` in the data pipeline. The joint-0 loss term is therefore identically zero.

---

## Expected Behaviour After Change

- `forward()` produces the same tensor shape `(B, 70, 3)` as the baseline. Extra compute per forward: 21 sequential tensor adds + 1 clone on a `(B, 22, 3)` tensor = <0.2 ms on 1080 Ti (negligible vs. the ~250 ms backbone forward).
- Training emits the SAME THREE loss scalars per step: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`. No new loss keys are added in Design 001.
- At init (epoch 0 step 0):
  - `pred['joints'][:, 0, :]` is exactly zero (forced by `_forward_kinematics`).
  - `pred['joints'][:, 1..21, :]` has mean ≈ 0 and standard deviation on the same scale as the baseline's direct-regression init, thanks to the `1/sqrt(21)` weight scaling.
  - `loss/joints/train` is a finite positive scalar, on the same order of magnitude as the baseline's step-0 joint loss (validated by the Builder via a short unit-test / first-iteration log inspection).
- The gradient on every body bone vector flows back from **every descendant joint's** loss term. For example, `bone_vec[3]` (pelvis→spine1) receives gradient from the loss at joints {3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21} (every spine and upper-body descendant). This is the structural inductive bias.
- Validation metrics (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`) are computed by the unchanged `BedlamMPJPEMetric` on `pred['joints']` — the recovered-coordinate tensor. No change to evaluation code paths.
- `MetricsCSVHook` writes the same CSV columns as baseline.
- Extra parameter count: **0**. Extra non-learnable buffer: `bone_parents` (22 int64 = 176 bytes).
- Expected result vs. baseline: `mpjpe_body_val` improves by 5–15 mm (target < 140 mm, breaking the 140.96 mm prior best). `mpjpe_pelvis_val` neutral (pelvis pathway unchanged). `mpjpe_hand_val` neutral (hand regression unchanged). `mpjpe_abs_val` mild positive (indirect benefit from better body). `composite_val` target < 153.
- At inference the tensor shapes and dtypes are **identical** to the baseline; only the semantics of the joints_out output are reinterpreted before concatenation.

---

## Rationale Summary

- **Why scale-init by 1/sqrt(21)?** If each of the 21 body bone-vec outputs has variance `σ²`, a cumulative sum of `k` of them has variance `k·σ²`. Joint 21 (right_wrist) sits at chain depth 5 (pelvis→spine1→spine2→spine3→neck does not lead to the wrist; the wrist chain is pelvis→right_hip does not either — let's count the actual wrist chain: right_wrist(21) ← right_elbow(19) ← right_shoulder(17) ← right_collar(14) ← spine3(9) ← spine2(6) ← spine1(3) ← pelvis(0) → 7 bones deep). The deepest chain in the SMPL-X 22-joint tree has 7 segments. The average depth across 21 children is ~4. To keep the recovered-joint variance comparable to baseline's direct-regression variance at init, a uniform scale of `1/sqrt(mean_depth)` ≈ `1/sqrt(4)` = 0.5 would be ideal for the average joint. Using `1/sqrt(21)` ≈ 0.218 (as suggested in idea.md) is stricter — it ensures the *worst-case* (root+ 21 cumulative adds) variance is bounded. This conservative choice is justified because initial-iteration stability matters more than a slight underflow of early-epoch signal; the network rapidly recovers from small scaling via the LR warmup.
- **Why not bias-scale?** The bias is `zeros_` at init; scaling `0 × 0.218 = 0` changes nothing. Leave it.
- **Why only 22 body joints and not all 70?** The kinematic-chain prior is meaningful only for the body skeleton. Hands have a different tree (48 hand joints, with their own parent structure in the SMPL-X hand model) and are NOT supervised by the `loss/joints/train` term (which restricts to `_BODY`). Reparameterizing hands would add complexity with no supervised signal to benefit from. Keep hands as direct regression.
- **Why a single shared Linear head?** This is the minimal-change baseline variant. Design 003 explores per-limb decoupled heads as an architectural refinement.
- **Why force root to zero?** After `SubtractRootJoint` in the data pipeline, `gt_joints[:, 0, :] = 0` exactly. Under kinematic parametrization, the recovered `body_rr[:, 0, :]` is *always* zero by construction. This means the joint-0 loss term contributes exactly zero to the training signal (consistent with baseline, where it is also zero-centred). No gradient is wasted on "predicting zero for the root".

---

## Risk and Mitigation Specific to Design 001

- **Initialization drift from cumulative sum**: Mitigated by `1/sqrt(21)` weight scaling. The Builder SHOULD include a sanity check in a one-shot forward pass (via a minimal `torch.randn`-fed test in Builder's local debugging, not in SLURM training): `pred_joints_std = pred['joints'][:, 1:22, :].std().item()` should be in `[0.005, 0.03]` at init — comparable to baseline's `joints_out` output std after trunc-normal `std=0.02` init.
- **Gradient scale imbalance between body and hand heads**: Since body outputs are cumulatively summed, the gradient on `joints_out.weight` rows for body queries is larger than for hand queries (each body bone gets `sum of descendant gradients`). The LR is unchanged; if instability arises, the Builder MAY set a smaller `lr_mult` on the head via `paramwise_cfg` but THIS IS NOT PART OF DESIGN 001. Design 001 is the minimal-change variant; any such tuning is deferred to a follow-up design.
- **In-place ops and autograd**: the forward-kinematics loop does a series of in-place tensor slice assignments on a cloned tensor. This is autograd-safe as long as each assignment overwrites (not reads then writes) — and here each step writes a *different* slot `[:, child, :]` that has not yet been assigned in the current loop iteration. The source slots `body_rr[:, parent, :]` and `bone_vecs[:, child, :]` are both valid and distinct from the destination slot. No aliasing issue.
- **Parent list correctness**: `bone_parents` is verified by assertion in `__init__`. The list is identical to idea012 (validated there).
- **Fixed-point stability of forward_kinematics**: It is NOT a fixed-point — it is a finite topologically-ordered loop of 21 steps. No convergence concerns.
- **Per-iteration `.item()` syncs**: Avoided by pre-caching `self._bone_parents_list = list(bone_parents)` as a Python list of ints, bypassing the tensor index.
- **Tensor version counter / autograd graph breakage**: the `bone_vecs.clone()` creates a new tensor that is linked into the autograd graph via `clone`'s forward/backward. The subsequent in-place writes on this cloned tensor modify its version counter but do NOT invalidate the forward graph (we never re-read the cloned tensor before writing each slot). PyTorch's in-place-op autograd check will not complain: writes are to distinct slots.
- **Eval/inference compatibility**: `predict()` calls `forward()`, which now returns recovered coordinates. `bedlam_metric.py` sees the same `(B, 70, 3)` tensor shape and computes MPJPE exactly as before.
- **MMEngine config constraint**: all new kwargs are bool/float/int-list literals. No `import` statements introduced in the config.
- **Interaction with prior ideas**: orthogonal. Can compose with idea002, idea008, idea011, idea012 (none of which change the output-semantics parameterization).
- **Memory / speed**: one extra `(B, 22, 3)` tensor allocation (clone) per forward = 264 B × batch-size = ~1 KB; 21 tensor adds = negligible on 1080 Ti.
- **Param count change**: none. Shape-bit-identical to baseline model state dict after rewriting the `joints_out.weight` values in place.
