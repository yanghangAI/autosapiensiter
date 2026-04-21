# Design 003 — Log-Scaled Pairwise Distance-Matrix Loss (scale-invariant / proportion-aware)

**Design Description:** Same upper-triangular pairwise auxiliary loss as Design 001 (231 body-joint pairs, `dist_loss_weight=0.5`), but applied to the **log** of distances: `L_dist = mean |log(d_pred + eps) - log(d_gt + eps)|` with `eps=1e-3`. This makes the loss scale-invariant — a 10% error on a 0.1 m spine segment counts the same as a 10% error on a 1.5 m head-to-ankle diagonal — which emphasises anatomical proportions rather than absolute lengths.

**Starting Point:** `baseline/`

---

## Overview

Absolute pairwise distances in a human body range over roughly two orders of magnitude:

- Small bones (spine segments, neck, head stalk): ~0.05–0.15 m.
- Medium bones (upper arm, forearm, thigh, shin): ~0.25–0.45 m.
- Long diagonals (head-to-ankle, wrist-to-ankle across body): ~1.2–1.7 m.

Under the Design 001 absolute-L1 loss, a 0.05 m error on a 0.1 m bone (a 50% relative error) contributes the same 0.05 m to the loss as a 0.05 m error on a 1.5 m diagonal (a 3% relative error). That is, the absolute loss **under-weights proportional errors on small bones**. For anatomical correctness, a 50% error on a spine segment is far more egregious than a 3% error on a head-to-ankle diagonal.

Design 003 addresses this via the classic scale-invariance trick: supervise the log of the distance.

```
L_dist = mean_{i<j} | log(d_pred[i,j] + eps) - log(d_gt[i,j] + eps) |
       = mean_{i<j} | log( (d_pred[i,j] + eps) / (d_gt[i,j] + eps) ) |
losses['loss/dist_matrix/train'] = dist_loss_weight * L_dist   # dist_loss_weight = 0.5
```

The `| log(a/b) |` formulation means the error is directly a relative/proportional term: a 10% stretch anywhere on the body contributes the same gradient magnitude regardless of which bone it's on. This is the natural signal for anatomical proportion.

`eps = 1e-3` m (= 1 mm) is added inside both logs to:
- avoid `log(0)` if two predicted joints ever coincide (practically impossible in a trained model, but defensively safe),
- keep the loss finite and differentiable throughout training,
- keep the effective scale of the log near `log(d)` for typical `d ≥ 0.05` m (since `0.05 + 0.001 ≈ 0.05`, the eps is <2% of the smallest real bone distance and does not distort the loss).

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are identical to the baseline.

---

## Files to Change

1. `pose3d_transformer_head.py` — accept the same three new kwargs (`dist_loss_weight`, `dist_loss_mode`, `dist_loss_eps`) as Design 001; in `loss()`, take the `'log'` branch.
2. `config.py` — add `dist_loss_weight=0.5`, `dist_loss_mode='log'`, `dist_loss_eps=1e-3` in the `head=dict(...)` block.
3. `pelvis_utils.py` — **no change**.

No new imports are introduced beyond those already present (`torch`, `torch.nn`). `torch.log` is in the base `torch` namespace.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

The head-file changes are **structurally identical to Design 001** — same three new kwargs, same three-branch `if/elif/else` in `loss()`. The only difference from Design 001 is the branch actually taken at runtime (`'log'` instead of `'abs'`).

#### 1. `Pose3dTransformerHead.__init__` — new parameters

Add three kwargs to the `__init__` signature, placed immediately after `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`:

```python
dist_loss_weight: float = 0.0,
dist_loss_mode: str = 'abs',
dist_loss_eps: float = 1e-3,
```

Store them as attributes:

```python
self.dist_loss_weight = dist_loss_weight
self.dist_loss_mode = dist_loss_mode
self.dist_loss_eps = dist_loss_eps

assert dist_loss_mode in ('abs', 'bone_weighted', 'log'), (
    f"dist_loss_mode must be one of 'abs' | 'bone_weighted' | 'log', "
    f"got {dist_loss_mode!r}")
```

Constraints:
- Default `dist_loss_weight=0.0` preserves baseline behaviour exactly.
- Default `dist_loss_mode='abs'` (unchanged across the three designs). Design 003 sets `'log'` via config.
- Default `dist_loss_eps=1e-3`. Design 003 keeps this default.
- NO new `nn.Module`, `nn.Parameter`, or `register_buffer` in Design 003. Both `dist_loss_eps` and `dist_loss_weight` are plain Python floats. `bone_parents` / `self.bone_weights` do NOT exist in Design 003 (mode is `'log'`, not `'bone_weighted'`).

#### 2. `_init_head_weights` — unchanged

No change. No new learnable parameters.

#### 3. `forward()` — unchanged

No change. The distance-matrix loss operates on `pred['joints']` in `loss()` only.

#### 4. `loss()` — append auxiliary distance-matrix term (log branch)

Inside `Pose3dTransformerHead.loss`, AFTER the existing three loss assignments (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`) and BEFORE the `with torch.no_grad():` MPJPE block, insert the three-branch block:

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
        # Design 002 path — unreachable in Design 003 (self.bone_weights is None).
        w = self.bone_weights.to(d_pred.device)
        L_dist = (w * (d_pred - d_gt).abs()).mean()
    else:  # 'log' — Design 003 path.
        eps = self.dist_loss_eps
        L_dist = (torch.log(d_pred + eps) - torch.log(d_gt + eps)).abs().mean()

    losses['loss/dist_matrix/train'] = self.dist_loss_weight * L_dist
```

In Design 003 the taken branch is `'log'`.

Constraints:
- `torch.log(d_pred + eps)` is differentiable everywhere `d_pred + eps > 0`. Since `d_pred ≥ 0` (it is an `L2` norm) and `eps = 1e-3 > 0`, the argument is always strictly positive. No NaN risk from `log`.
- The derivative `∂ log(d + eps) / ∂ d = 1 / (d + eps)`. At `d = 0` this is `1/eps = 1000`, which is bounded; backprop through the outer `cdist` with predicted coords far from each other (typical, > 0.05 m) yields `1 / 0.05 = 20`. This is within a well-behaved range and does not trigger gradient explosion. The `clip_grad max_norm=1.0` in the optimizer wrapper provides an additional safety net.
- The key name remains `'loss/dist_matrix/train'` (same as Designs 001 and 002).
- `self.dist_loss_weight * L_dist` is applied AFTER the raw mean.
- The full `if/elif/else` structure MUST be present so the head file is identical across Designs 001/002/003 apart from instance attribute values. The `elif 'bone_weighted'` branch accesses `self.bone_weights` — in Design 003 this attribute is `None` (set in `__init__` — see §1), so accessing `.to()` on it would raise `AttributeError`. Since `self.dist_loss_mode == 'log'` in Design 003, the `elif` branch is unreachable at runtime and no error occurs.
- DO NOT replace `.abs()` with `**2` or any other norm; the loss is an L1 in log-distance space, which is robust and proportional.

Keep the `with torch.no_grad():` block UNCHANGED.

#### 5. `predict()` — unchanged

No change.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`, add the three new kwargs at the end (after `loss_weight_uv=1.0,`):

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
    dist_loss_mode='log',
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
| **dist_loss_mode** | **'log' (new)** |
| **dist_loss_eps** | **1e-3 (new, used by mode 'log')** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only (`_BODY = list(range(0, 22))`). The log-distance loss is defined only over the 22 body joints.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `dist_loss_eps=1e-3` is a float literal.
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package). No new imports needed.
6. `dist_loss_weight` default MUST be `0.0`; `dist_loss_mode` default MUST be `'abs'`; `dist_loss_eps` default MUST be `1e-3`.
7. `dist_loss_mode` MUST be validated in `__init__` against `('abs', 'bone_weighted', 'log')`.
8. `bone_parents` kwarg is NOT required in Design 003 (mode is `'log'`, not `'bone_weighted'`). If the head-file signature includes `bone_parents=None` (matching Design 002's superset signature), the kwarg simply is not passed by Design 003's config and defaults to `None`; `self.bone_weights` is set to `None` in the `else` branch of the mode check in `__init__`.
9. The new loss term MUST appear with key `'loss/dist_matrix/train'`.
10. `torch.cdist` MUST use `p=2`. `torch.triu_indices` MUST use `offset=1`.
11. `eps` MUST be added INSIDE each `torch.log(...)` call. Applying `torch.log(d_pred).sub(torch.log(d_gt))` without `eps` would NaN if `d_pred` happens to be exactly 0 (unlikely but possible at very first step if weights random-init the output close to constant).
12. `eps = 1e-3` (1 mm). The Builder MUST NOT substitute a smaller value like `1e-6`: with `eps=1e-6`, the gradient `1/(d+eps)` at `d=0` is `10^6`, which would dominate the loss and destabilise training. With `eps=1e-3`, the worst-case gradient magnitude is `10^3 = 1000`, already bounded by `clip_grad max_norm=1.0`.
13. `forward()` MUST NOT be modified.
14. `predict()` MUST NOT be modified.
15. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, and `BedlamMPJPEMetric` are untouched.
16. No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
17. No changes to `pelvis_utils.py`.
18. The head `__init__` signature MUST remain backward compatible. New kwargs are keyword-only with defaults.
19. No extra learnable parameters. No buffers for Design 003 (mode `'log'` uses only scalars).
20. If the Builder chooses to reuse a single head-file implementation across Designs 001/002/003 (recommended), the three-branch `if/elif/else` on `self.dist_loss_mode` is mandatory; the `elif 'bone_weighted'` branch accesses `self.bone_weights`, which is `None` in Design 003 — the branch MUST be guarded by the `elif self.dist_loss_mode == 'bone_weighted'` condition (it is), so `None.to(...)` is never invoked at runtime.

---

## Expected Behaviour After Change

- `forward()` is identical in compute and output to the baseline.
- Training emits FOUR loss scalars: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/dist_matrix/train` (new).
- The raw `L_dist` magnitude under `'log'` mode is roughly `log(1 + relative_error)` on average. For a typical 10% relative distance error, `log(1.1) ≈ 0.095`; compared with Design 001's absolute L1 (~0.1–0.3 m raw), the `'log'` raw loss is of similar-to-smaller magnitude. With the same `dist_loss_weight=0.5`, the aggregate contribution to the total loss is of similar order.
- The auxiliary loss starts positive and decreases monotonically.
- At init, `loss/dist_matrix/train` is a finite positive scalar (not NaN, not Inf). If NaN, the Builder should verify `pred_body + eps > 0` everywhere (trivially true when `eps > 0`) and that `gt_body` has no NaN.
- Validation metrics are computed by the unchanged `BedlamMPJPEMetric` on `pred['joints']`.
- Extra parameter count: **0**.
- Expected result vs. baseline: on top of the general Design 001 gain, small-bone / proportion errors are expected to tighten (e.g., spine segments, neck, head). Hand-proximal joints (wrists) may benefit most since their bones are the shortest in the body-22 set.
- At inference the behaviour is bit-identical to the baseline.

---

## Rationale Summary

- **Why log(d)?** The log transform maps relative errors to absolute log-space errors: `|log(d_pred) - log(d_gt)| = |log(d_pred / d_gt)|`. A 10% stretch (`d_pred = 1.1 * d_gt`) contributes `|log(1.1)| ≈ 0.095` regardless of whether the bone is 5 cm or 50 cm. This is the scale-invariance property needed for anatomical proportion.
- **Why eps inside the log?** Numerical safety at `d = 0` (coincident joints, theoretically possible with random init). `eps = 1e-3` m is much smaller than the smallest real body bone (~5 cm), so it doesn't distort the loss on real inputs.
- **Why `dist_loss_weight=0.5` unchanged?** The `'log'` raw loss has roughly the same magnitude as the `'abs'` raw loss (both are around 0.1–0.3 in natural-body units), so the same scalar weight keeps the aggregate contribution balanced. A sweep (0.25 / 0.5 / 1.0) is a natural follow-up but is outside the scope of this single-variation design.
- **Complementary to Designs 001 and 002**: Design 001 tests the raw pairwise signal; Design 002 emphasises bone pairs by static weighting; Design 003 emphasises small/short distances by logarithmic scaling. The three designs test three orthogonal ways to shape the distance-matrix loss.

---

## Risk and Mitigation Specific to Design 003

- **Gradient magnitude near `d = 0`**: `d / dd log(d + eps) = 1/(d + eps)`. For `eps = 1e-3`, the max is `1000`; for typical `d ≈ 0.1`, the gradient is `~10`. `clip_grad max_norm=1.0` provides belt-and-braces clipping at the optimizer level.
- **`log(x).abs()` differentiability at `x = 1`**: at `d_pred == d_gt`, the log difference is 0 and `|·|` has a subgradient of 0. `torch.abs` at 0 picks the subgradient 0 automatically (PyTorch standard behaviour for `abs` at 0). No special handling needed.
- **Interaction with existing ideas**: orthogonal to every prior idea; composes cleanly.
- **MMEngine config constraint**: `dist_loss_eps=1e-3` is a float literal. `dist_loss_mode='log'` is a string literal. Fully compliant.
- **Memory / speed**: same as Design 001 plus two `torch.log` ops on `(B, 231)` tensors — sub-millisecond on 1080 Ti.
- **No NaN / Inf risk**: `d_pred + eps > 0` always. `log` is always finite on positive input. `torch.abs` is always finite. The product `self.dist_loss_weight * L_dist` is a bounded positive scalar.
- **Eval/inference compatibility**: training-only. `predict()` unchanged.
- **Per-bone imbalance**: because log is scale-invariant, it effectively normalises the gradient contribution per pair. This is the desired behaviour but means pairs with very small distances (spine segments) contribute a larger relative gradient per coordinate error than in Design 001. If the model over-corrects on short bones at the expense of long ones, the Builder could revisit `dist_loss_weight` (tune to 0.25 as an alternative). For Design 003 we keep `0.5` per the idea spec.
