# Design 001 — Fixed Log-Uniform Depth Bins + SORD Soft-Argmax (Pure Classification)

**Design Description:** Replace the single-scalar pelvis depth regression head with a K=64-way softmax over fixed log-uniform depth bins in [1.0, 15.0] m. Recover the scalar pelvis depth as the soft-argmax expectation of the distribution. Supervise with SORD-style soft-target cross-entropy centred at GT depth; NO auxiliary regression term.

**Starting Point:** `baseline/`

---

## Overview

The baseline depth pathway is a single `Linear(hidden_dim, 1)` applied to `decoded[:, 0, :]` producing a scalar metre value, trained with `SoftWeightSmoothL1Loss(beta=0.05)`. Under this design, that scalar head is replaced by a K=64-way classification head over fixed log-uniform depth bins in [1.0, 15.0] m. The continuous depth scalar is recovered as a fully differentiable soft-argmax expectation over the bin centres. Loss is SORD-style soft-target cross-entropy: the target is a Gaussian in log-depth-space centred at the GT depth with σ=1.5 × bin width (in log-space), truncated-softmax-normalised across the K bins.

All other code paths are identical to the baseline:
- UV head, joint head, depth recovery (`recover_pelvis_3d`), `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, hooks, optimizer, LR schedule, seed, batch size, accumulation, evaluation.
- The returned dict still contains `'pelvis_depth': (B, 1)` metric-scalar tensor, preserving all downstream shape contracts.

---

## Files to Change

1. `pose3d_transformer_head.py` — add new kwargs; replace `self.depth_out = Linear(hidden_dim, 1)` with `Linear(hidden_dim, K)` when classification mode is active; register the `log_bin_centres` buffer; update `forward()` to compute soft-argmax expectation and additionally expose `depth_logits` in the returned dict; update `loss()` to compute SORD soft-target cross-entropy when classification mode is active.
2. `config.py` — add the new head kwargs.
3. `pelvis_utils.py` — **no change**.

No new top-level imports are introduced beyond those already in `pose3d_transformer_head.py` (`math`, `torch`, `torch.nn`). The `math.log` call for log-space bin placement uses the already-imported `math` module.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `Pose3dTransformerHead.__init__` — new parameters

Add FIVE kwargs to the `__init__` signature, placed immediately after `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`. The same five-parameter signature is shared across all three designs of idea014; only the passed values differ. Hence Design 001 uses:

```python
depth_head_type: str = 'regression',
num_depth_bins: int = 64,
depth_range_min: float = 1.0,
depth_range_max: float = 15.0,
depth_soft_label_sigma: float = 1.5,
depth_aux_reg_weight: float = 0.0,
```

All five defaults reproduce the baseline behaviour bit-for-bit when the config does not set them. Note: this is actually SIX kwargs (5 classification params + `depth_aux_reg_weight`). The Builder MUST accept all six in the `__init__` signature even for Design 001 (shared signature across designs).

Acceptable `depth_head_type` values (validated by assertion in `__init__`):
- `'regression'` — baseline behaviour (single scalar linear head).
- `'classification'` — Design 001: fixed log-uniform bins + SORD soft CE, no aux regression.
- `'classification_hybrid'` — Design 002: same as Design 001 + auxiliary SmoothL1 on expectation.
- `'classification_adaptive'` — Design 003: Design 002 + per-sample adaptive bin widths.

For Design 001:
- `depth_head_type = 'classification'`
- `num_depth_bins = 64`
- `depth_range_min = 1.0`
- `depth_range_max = 15.0`
- `depth_soft_label_sigma = 1.5`
- `depth_aux_reg_weight = 0.0`  (inactive in Design 001; enforced by `depth_head_type == 'classification'`)

Store them as attributes:

```python
self.depth_head_type = depth_head_type
self.num_depth_bins = num_depth_bins
self.depth_range_min = depth_range_min
self.depth_range_max = depth_range_max
self.depth_soft_label_sigma = depth_soft_label_sigma
self.depth_aux_reg_weight = depth_aux_reg_weight
```

Place this block immediately after the existing `self.loss_weight_uv = loss_weight_uv` line.

Validate the values in `__init__`:

```python
assert depth_head_type in ('regression', 'classification',
                           'classification_hybrid', 'classification_adaptive'), (
    f"Invalid depth_head_type='{depth_head_type}'. Must be one of: "
    f"'regression', 'classification', 'classification_hybrid', "
    f"'classification_adaptive'.")
if depth_head_type != 'regression':
    assert num_depth_bins >= 4, (
        f"num_depth_bins must be >= 4, got {num_depth_bins}")
    assert 0.0 < depth_range_min < depth_range_max, (
        f"Require 0 < depth_range_min < depth_range_max, got "
        f"({depth_range_min}, {depth_range_max})")
    assert depth_soft_label_sigma > 0.0, (
        f"depth_soft_label_sigma must be > 0, got {depth_soft_label_sigma}")
    assert depth_aux_reg_weight >= 0.0, (
        f"depth_aux_reg_weight must be >= 0, got {depth_aux_reg_weight}")
```

#### 2. `__init__` — output-head allocation and bin-centre buffer

Currently the baseline allocates:

```python
self.depth_out = nn.Linear(hidden_dim, 1)
```

Replace this block (keep placement in the same sequence — AFTER `self.joints_out = nn.Linear(hidden_dim, 3)` and BEFORE `self.uv_out = nn.Linear(hidden_dim, 2)`) with a conditional allocation:

```python
self.joints_out = nn.Linear(hidden_dim, 3)
if self.depth_head_type == 'regression':
    self.depth_out = nn.Linear(hidden_dim, 1)
else:
    # Classification modes: emit logits over K bins.
    self.depth_out = nn.Linear(hidden_dim, self.num_depth_bins)
    # Log-uniform bin CENTRES in log-space; exp() at forward/loss time.
    log_min = math.log(self.depth_range_min)
    log_max = math.log(self.depth_range_max)
    log_centres = torch.linspace(log_min, log_max, self.num_depth_bins)
    self.register_buffer('log_bin_centres', log_centres, persistent=False)
self.uv_out = nn.Linear(hidden_dim, 2)
```

Constraints:
- `self.log_bin_centres` MUST be a non-persistent buffer (recomputable from kwargs on reload, so no checkpoint coupling).
- `log_bin_centres` has shape `(K,)` and dtype `torch.float32`.
- `bin_centres = log_bin_centres.exp()` at forward time (shape `(K,)`, metres).
- For `depth_head_type='regression'`, Baseline shape `(hidden_dim, 1)` preserved exactly. Zero new buffers, zero new kwargs in use.
- For Design 001: `self.depth_out` is `Linear(256, 64)` instead of `Linear(256, 1)`. Net parameter delta: `256 × 63 + 63 = 16191` additional float32 weights (~63 kB).
- `num_depth_bins=64` is a fixed Python int literal. The module shape is set at construction time; changing bins after construction is not supported.

Design 003 (adaptive) adds a SECOND linear head `self.depth_bins_head = Linear(hidden_dim, num_depth_bins)` that predicts per-sample bin widths. Design 001 (and Design 002) do NOT allocate this second head. Design 001's head is ONLY `self.depth_out = Linear(hidden_dim, K)`.

#### 3. `_init_head_weights` — unchanged except for the classification head width

The baseline loop iterates `for m in [self.joints_out, self.depth_out, self.uv_out]:` and calls `nn.init.trunc_normal_(m.weight, std=0.02)`. Leave this loop UNCHANGED — it correctly applies to the new K-wide classification head because `nn.init.trunc_normal_` accepts any linear-layer weight shape. No special init tweaks; softmax of near-zero logits will start near-uniform over bins. See "Risk" section for discussion of the init-expectation value.

Optional (NOT required for Design 001): if the Builder wishes to initialise the classification head so that the initial expectation is near the dataset mean pelvis depth (≈ 4 m), they MAY set the bias of `self.depth_out` to `-(log_bin_centres - log(4.0))**2 / 2`. Design 001 does NOT require this — default zero-bias init produces a near-uniform softmax whose expectation is the geometric mean of the bin centres = `sqrt(1.0 × 15.0) ≈ 3.87 m`, which is already a reasonable starting value.

#### 4. `forward()` — compute soft-argmax expectation in classification modes

Currently `forward()` has:

```python
pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
```

Replace with a conditional block (still producing a `(B, 1)` `pelvis_depth` tensor as the primary return — so downstream consumers are unchanged). ALSO add `depth_logits` and `depth_bin_centres` to the returned dict for the loss function's use:

```python
if self.depth_head_type == 'regression':
    pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
    depth_logits = None
    depth_bin_centres = None
else:
    depth_logits = self.depth_out(pelvis_token)  # (B, K)
    # Bin centres: exp(log_bin_centres) for fixed mode. (K,)
    bin_centres = self.log_bin_centres.exp()  # (K,) on buffer's device
    depth_bin_centres = bin_centres.unsqueeze(0).expand(depth_logits.size(0), -1)  # (B, K)
    probs = torch.softmax(depth_logits, dim=-1)  # (B, K)
    expected_depth = (probs * depth_bin_centres).sum(dim=-1, keepdim=True)  # (B, 1)
    pelvis_depth = expected_depth
```

The returned dict becomes:

```python
return {
    'joints': joints,
    'pelvis_depth': pelvis_depth,
    'pelvis_uv': pelvis_uv,
    'depth_logits': depth_logits,
    'depth_bin_centres': depth_bin_centres,
}
```

Constraints:
- `pelvis_depth` MUST remain shape `(B, 1)` dtype `torch.float32` — matches baseline contract.
- `depth_logits` is `(B, K)` in classification modes, `None` in regression mode.
- `depth_bin_centres` is `(B, K)` in classification modes, `None` in regression mode.
- In regression mode, `None` values pass through; `loss()` MUST check for the active mode and skip logits-based loss if `depth_logits is None`.
- Downstream `predict()` MUST be unchanged — it reads `pred['pelvis_depth']` only.
- The soft-argmax is fully differentiable: gradients flow through `probs` (via softmax) back into `depth_logits` and ultimately into `depth_out.weight`.

#### 5. `loss()` — classification CE with SORD soft targets (Design 001)

The baseline loss computes:

```python
losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
    pred['pelvis_depth'], gt_depth)
```

In Design 001 (`depth_head_type='classification'`), replace the depth-loss line with:

```python
if self.depth_head_type == 'regression':
    losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
        pred['pelvis_depth'], gt_depth)
else:
    # SORD soft cross-entropy on depth logits.
    #   target_i = softmax_i( -(log_bin_centres - log(z_gt))^2 / (2 * sigma_log^2) )
    # with sigma_log = soft_label_sigma * bin_width_log.
    log_bin_centres = self.log_bin_centres            # (K,)
    K = log_bin_centres.numel()
    log_min = math.log(self.depth_range_min)
    log_max = math.log(self.depth_range_max)
    bin_width_log = (log_max - log_min) / max(K - 1, 1)
    sigma_log = self.depth_soft_label_sigma * bin_width_log

    # Clamp GT depth into the bin range for numerical safety; otherwise
    # exp()/log() with slightly-out-of-range GT still works but produces
    # one-sided target distributions.
    z_gt = gt_depth.clamp(min=self.depth_range_min,
                          max=self.depth_range_max)  # (B, 1)
    log_z_gt = z_gt.log()                             # (B, 1)

    # log_bin_centres broadcast to (1, K); log_z_gt is (B, 1).
    log_diff = log_bin_centres.unsqueeze(0) - log_z_gt   # (B, K)
    target_logits = -(log_diff ** 2) / (2.0 * sigma_log ** 2)  # (B, K)
    target = torch.softmax(target_logits, dim=-1)       # (B, K)

    # Soft-target cross-entropy = -sum(target * log_softmax(logits)).
    log_probs = torch.log_softmax(pred['depth_logits'], dim=-1)  # (B, K)
    ce_per_sample = -(target * log_probs).sum(dim=-1)    # (B,)
    L_depth_ce = ce_per_sample.mean()
    losses['loss/depth/train'] = self.loss_weight_depth * L_depth_ce

    if self.depth_aux_reg_weight > 0.0:
        # Design 002/003: hybrid with SmoothL1 on expectation.
        # Design 001: depth_aux_reg_weight == 0.0 — this branch is skipped.
        import torch.nn.functional as F  # local import (head file already
        # has torch; `F.smooth_l1_loss` is the stable, autograd-friendly form).
        L_depth_reg = F.smooth_l1_loss(
            pred['pelvis_depth'], gt_depth.to(pred['pelvis_depth'].device),
            reduction='mean', beta=0.05)
        losses['loss/depth_reg/train'] = self.depth_aux_reg_weight * L_depth_reg
```

For Design 001, `self.depth_aux_reg_weight == 0.0`; the `if self.depth_aux_reg_weight > 0.0:` block is skipped — no `loss/depth_reg/train` key is emitted.

Constraints:
- The computation `bin_width_log = (log_max - log_min) / max(K - 1, 1)` uses K-1 in the denominator because `torch.linspace(log_min, log_max, K)` produces endpoints-inclusive bin centres; the gap between adjacent centres is `(log_max - log_min) / (K - 1)`.
- `z_gt.log()` requires `z_gt > 0` — this is guaranteed by the `.clamp(min=depth_range_min)` with `depth_range_min = 1.0`. BEDLAM2 pelvis depths are positive by physical construction.
- The `-(target * log_probs).sum(dim=-1)` formulation is the soft-cross-entropy (equivalent to `F.kl_div(log_probs, target, reduction='batchmean') + H(target)` up to the constant entropy term; we use the direct form for simplicity and numerical stability).
- Gradient flows through `pred['depth_logits']` (via `log_softmax`) back into the depth head weights. `pred['pelvis_depth']` (the expectation) is NOT used in the classification loss term — only in the hybrid/regression branches.
- `target` MUST be re-computed every step (not cached across steps) because `z_gt` varies per batch.
- `target` detached from the autograd graph implicitly because it is computed from `gt_depth` which is a non-differentiable ground-truth tensor.
- If the Builder uses `F.kl_div` instead, they MUST pass `log_target=False` and `reduction='batchmean'` — otherwise the scale is wrong. The explicit `-(target * log_probs).sum(dim=-1).mean()` form is preferred for clarity.

Leave the MPJPE `with torch.no_grad():` block UNCHANGED. `self._train_mpjpe_abs = _compute_mpjpe_abs(..., pred['pelvis_depth'], gt_depth, ...)` reads `pred['pelvis_depth']` which is the soft-argmax expectation — same shape `(B, 1)` as the baseline scalar. The `_compute_mpjpe_abs` function does not care whether the scalar came from regression or expectation.

#### 6. `predict()` — unchanged

`predict()` reads `pred['pelvis_depth']` (the expectation scalar) and writes it to `inst.pelvis_depth`. No modification. `BedlamMPJPEMetric` sees the same tensor shape and units.

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
    depth_head_type='classification',
    num_depth_bins=64,
    depth_range_min=1.0,
    depth_range_max=15.0,
    depth_soft_label_sigma=1.5,
    depth_aux_reg_weight=0.0,
),
```

All new values are `str` / `int` / `float` literals — fully MMEngine-config compliant (no Python imports required). `loss_depth` is retained in the config (the head file instantiates `self.loss_depth_module` for all modes, but the module is simply unused in classification modes since `loss/depth/train` is computed inline as CE). Keeping `loss_depth=dict(...)` in the config preserves backward compatibility of the head signature and keeps the configuration uniform across designs. The `loss_weight_depth=1.0` multiplier is still applied to `loss/depth/train` whether that value came from regression or CE.

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights, `custom_imports` list, dataloaders, evaluators) are identical to the baseline.

---

## Exact Config Values (unchanged from baseline except six head kwargs)

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
| loss_depth loss_weight | 1.0 (unused in classification mode but retained for signature uniformity) |
| loss_uv loss_weight | 1.0 (× loss_weight_uv=1.0) |
| loss_weight_depth | 1.0 (multiplies the CE scalar `loss/depth/train`) |
| **depth_head_type** | **'classification' (new)** |
| **num_depth_bins** | **64 (new)** |
| **depth_range_min** | **1.0 (new)** |
| **depth_range_max** | **15.0 (new)** |
| **depth_soft_label_sigma** | **1.5 (new)** |
| **depth_aux_reg_weight** | **0.0 (new; disabled in Design 001)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change (NPZ mmap FD issue).
2. Loss restricted to body joints 0-21 only for `loss/joints/train` (`_BODY = list(range(0, 22))`). Unchanged from baseline.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. All six new kwargs are `str`/`int`/`float` literals.
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package). Do NOT add any new top-level imports beyond what already exists (`math`, `torch`, `torch.nn`, `mmengine.structures`, `mmpose.*`, `pelvis_utils`). `torch.nn.functional` if needed is already implicitly available via `torch.nn.functional` (but Design 001 does not require it).
6. `depth_head_type` default MUST be `'regression'` (so omitting it in the config reproduces the baseline head bit-for-bit).
7. `num_depth_bins` default MUST be `64`, `depth_range_min` default `1.0`, `depth_range_max` default `15.0`, `depth_soft_label_sigma` default `1.5`, `depth_aux_reg_weight` default `0.0`. When `depth_head_type='regression'`, NONE of these params has any effect.
8. The `self.depth_out` module MUST have shape `(hidden_dim, 1)` for regression mode and `(hidden_dim, num_depth_bins)` for classification modes. The module is constructed ONCE in `__init__` and never swapped.
9. The `self.log_bin_centres` buffer MUST be registered as non-persistent (`persistent=False`). The buffer is not part of the checkpoint; on load, it is recomputed from `depth_range_min`, `depth_range_max`, `num_depth_bins`.
10. In classification modes, the returned `pred['pelvis_depth']` MUST be the soft-argmax expectation — a `(B, 1)` float tensor in metres, NOT logits.
11. In classification modes, the returned dict MUST contain `'depth_logits'` (shape `(B, K)`) and `'depth_bin_centres'` (shape `(B, K)`). In regression mode, these keys MAY be `None` or omitted; the `loss()` function MUST be robust to either.
12. The MPJPE computation in `loss()` MUST use `pred['pelvis_depth']` (the expectation scalar) — NOT the logits. `_compute_mpjpe_abs` receives a `(B, 1)` scalar tensor in metres regardless of mode.
13. `self.loss_depth_module` is still instantiated (for regression mode fallback); in classification modes it is never called. Do NOT remove it.
14. Parameter count delta vs. baseline: `+16191` float32 weights and biases on `self.depth_out` (`Linear(256, 64)` vs `Linear(256, 1)`: `256*64+64 - 256*1-1 = 16383 + 1 = 16191` additional params; the +1 is the bias: `64 − 1 = 63` extra bias entries plus `256 × 63 = 16128` extra weight entries = `16191`). Equivalently: `+16191` new trainable parameters. The head param count is tiny (< 1 M) so this is < 2% of the head and < 0.01% of the total model.
15. `self.log_bin_centres` MUST be registered as a buffer BEFORE `_init_head_weights()` is called. Equivalent: register in `__init__` (standard flow) — the ordering is automatic because buffer registration happens during `__init__` while weight init happens in `_init_head_weights()` (called last).
16. The `target` tensor in `loss()` MUST be computed in full float32 — no casting to half-precision. If AMP/mixed precision is ever enabled later, target should live outside the autocast context (wrap in `with torch.autocast(enabled=False):` — NOT required for Design 001 since AMP is disabled per `FixedAMPOptimWrapper` defaults).
17. GT depth MUST be clamped to `[depth_range_min, depth_range_max]` before `log()` to avoid NaN from `log(0)` or weird negative-infinity behaviour on rare out-of-range samples.
18. The soft-target normalisation MUST use `torch.softmax(target_logits, dim=-1)` — NOT any hand-rolled Gaussian followed by `/sum`, because `softmax` is numerically stable in the tails.
19. `predict()` MUST NOT be modified. It reads `pred['pelvis_depth']` which is already the expectation scalar.
20. `BedlamMPJPEMetric`, `TrainMPJPEAveragingHook`, `MetricsCSVHook` are untouched — all continue to read the `(B, 1)` depth scalar.
21. No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
22. No changes to `pelvis_utils.py`.
23. The head `__init__` signature MUST remain backward compatible. The six new kwargs MUST be appended at the end of the existing signature (before `init_cfg`) and MUST have defaults matching the baseline behaviour (`depth_head_type='regression'`).
24. The Shared Signature across all three idea014 designs: all three designs use the same six new kwargs; only the *values* differ (Design 001: `'classification'`, `0.0` aux; Design 002: `'classification_hybrid'`, `0.3` aux; Design 003: `'classification_adaptive'`, `0.3` aux).

---

## Expected Behaviour After Change

- `forward()` produces a `(B, 1)` `pelvis_depth` tensor identical in shape/dtype to the baseline; additionally exposes `(B, K)` `depth_logits` and `(B, K)` `depth_bin_centres` for `loss()` consumption.
- Training emits the SAME THREE loss keys as the baseline: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`. The *value* of `loss/depth/train` is now a cross-entropy instead of SmoothL1. Its magnitude at init is ~`log(64) ≈ 4.16`, higher than baseline's SmoothL1 (~`0.5` early). This is accounted for: the baseline's `loss_weight_depth=1.0` is kept; no rebalancing is required (see Risk section).
- At init (epoch 0 step 0):
  - Softmax over near-zero logits is near-uniform: `probs ≈ 1/K` per bin.
  - Expectation: `E[z] = sum(probs * exp(log_bin_centres))` ≈ `sum_i (1/K) * exp(log_centre_i)`. For K=64 log-uniform bins over `[1, 15]`, this is approximately the *arithmetic mean* of `exp(linspace(log 1, log 15, 64))`. Numerically this evaluates to ≈ 5.5 m — the arithmetic mean is biased toward the upper end of the log-uniform distribution. (The geometric mean is `sqrt(1 × 15) ≈ 3.87 m`, but we are NOT computing the geometric mean; we are computing the arithmetic mean of the already-exponentiated centres.) This is a *reasonable* starting prediction for BEDLAM2.
  - After a handful of SGD steps the expectation converges rapidly toward the per-batch GT mean.
- `BedlamMPJPEMetric` sees the same `(B, 1)` depth scalar and computes `mpjpe_pelvis_val` and `mpjpe_abs_val` identically.
- Extra parameter count: `+16191` trainable params.
- Extra runtime cost per forward/backward: negligible (< 0.1% overhead — one extra `linspace`-sized `exp()`, one `softmax`, one `sum`, one `log_softmax` per step). Backbone dominates compute by > 99%.
- Extra memory: `(B, K) = (4, 64) = 256` float32 values = 1 kB per batch for the logits. Negligible.
- Expected result vs. baseline: `mpjpe_pelvis_val` improves by 3–8 mm (target < 170 mm, breaking the 174.43 mm prior best). `mpjpe_abs_val` improves proportionally (target < 440 mm). `mpjpe_body_val`, `mpjpe_rel_val`, `mpjpe_hand_val` neutral.
- At inference tensor shapes and dtypes are identical to the baseline; only the internal computation of the pelvis depth scalar is different (expectation over softmax bins instead of a direct linear output).

---

## Rationale Summary

- **Why log-uniform bins over `[1.0, 15.0]`?** BEDLAM2's pelvis depths in the synthetic renderings concentrate in the 2–8 m band but extend from ≈ 1 m (close subjects) to ≈ 15 m (far subjects). Log-uniform spacing gives tighter bin widths where depth is small (where metric error is most sensitive) and broader bins where depth is large. This mirrors DORN / AdaBins / BinsFormer's use of log-uniform or log-spaced bins for monocular depth estimation.
- **Why K=64 bins?** A sweet spot between resolution and gradient signal. With K=64 log-uniform bins over `[1, 15]` m, the minimum bin width is ≈ 2.4 cm at z=1 m (good resolution for close subjects where millimetric error matters most) and ≈ 36 cm at z=15 m (fine since far-subject MPJPE scales with depth anyway). K=64 is also a GPU-friendly power of two.
- **Why SORD-style soft targets with σ=1.5 × bin_width?** In vanilla hard-label CE, all non-GT bins are equally wrong — but for depth, a bin near the GT is *less* wrong than a bin far from GT. SORD uses a Gaussian (in log-space) centred at `log(z_gt)` with σ = 1.5 × bin width (log-space), which spreads target probability mass across 3–4 adjacent bins. This gives sub-bin-width resolution and gradient smoothness. σ=1.5 is standard in the literature; wider σ loses discriminative signal, narrower σ degenerates toward hard-label CE.
- **Why no aux regression term in Design 001?** Design 001 is the minimal change variant — pure classification replacement. Design 002 adds the hybrid SmoothL1 regression term. This isolates the *classification-only* effect so we can attribute any gain (or loss) to the bin-classification structure alone. If Design 001 succeeds but Design 002 is neutral, we've shown the aux term is unnecessary. If Design 001 underperforms but Design 002 succeeds, we've shown the aux term is essential — informative either way.

---

## Risk and Mitigation Specific to Design 001

- **CE loss magnitude vs. baseline SmoothL1**: Cross-entropy over 64 bins is ~`log(64) ≈ 4.16` at init, while SmoothL1 on depth with ~0.1 m error and β=0.05 is ~`0.5 × 0.1 = 0.05`. The CE is ~80× larger in magnitude. However, the `loss/joints/train` magnitude at init (SmoothL1 on 22 joints with ~0.5 m each) is ~`0.5 × 22 × 0.05 ≈ 0.55`, so the ratio `L_depth_CE / L_joints ≈ 4.16 / 0.55 ≈ 7.6` at init. This is larger than the baseline ratio (~0.09) but the CE gradient magnitude is bounded in [0, 1] per sample (since `log_softmax` derivatives saturate), so the *effective* gradient scale on depth weights is not 80× — it's about 1× the baseline magnitude. The optimizer is AdamW with adaptive per-parameter LR; it normalises per-parameter gradient magnitudes automatically. Mitigation: keep `loss_weight_depth = 1.0` as-is; monitor `loss/depth/train` in the first 500 iters; if it dominates and destabilises training, the Builder MAY set `loss_weight_depth=0.25` (not required for Design 001 — the expectation is stable training at 1.0).
- **Ambiguous init expectation (≈ 5.5 m)**: Could mislead early training if the dataset mean is ≈ 4 m. The SORD soft-target CE provides a strong signal to pull the bin mass toward the correct range within a few hundred iterations (the LR warmup covers iters 0 to `3 × iter_per_epoch`, giving ample time). No mitigation needed.
- **Gradient through the expectation ↔ through the softmax**: The soft-argmax expectation is used in `_compute_mpjpe_abs` (train-time no-grad) and not in any gradient-producing path in Design 001 — the only gradient-producing path is CE(logits, soft_target). Gradient flows through `log_softmax(depth_logits)`, which is well-behaved. In Designs 002/003, the expectation additionally receives a SmoothL1 gradient; see those designs.
- **Bin-range correctness (BEDLAM2 depth distribution)**: If pelvis depths occasionally fall outside `[1.0, 15.0]`, the `clamp` mitigates NaN but produces one-sided target distributions for those samples. The data distribution for BEDLAM2 render subset has been observed in prior ideas to concentrate in 1–15 m; the fraction outside is < 0.1%. No impact on training signal. If the Builder observes outliers (via log inspection), they MAY widen to `[0.5, 30.0]` — not required for Design 001.
- **MMEngine config constraint**: all six new kwargs are str/int/float literals. No `import` statements introduced.
- **Parameter state-dict backward compatibility**: The `self.depth_out` shape changes from `(1, hidden_dim)` to `(K, hidden_dim)`. This means the baseline checkpoint *cannot* be directly loaded into a classification-mode head — but this design does not `load_from` any baseline checkpoint (pretrained weights come from the backbone only; head weights are fresh-initialised). No compatibility issue.
- **Composition with prior ideas**: fully orthogonal. idea005 (uncertainty weighting) rescales the scalar `loss/depth/train`, whether it is SmoothL1 or CE — works identically. idea010 (2D reprojection) uses `pelvis_depth` as the expectation scalar — fully differentiable, gradients flow through the expectation into the softmax and into the logits.
- **Memory/speed**: Extra memory `(B, 64) × 4 bytes` ≈ 1 kB per batch; extra compute one softmax + one expectation per forward = < 0.1 ms. Negligible.
