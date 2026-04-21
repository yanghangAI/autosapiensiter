# Design 003 — Adaptive Per-Sample Bin Widths (AdaBins) + Soft-Argmax + SmoothL1 Hybrid

**Design Description:** Design 002 plus per-sample adaptive depth-bin widths à la AdaBins. A second head `depth_bins_head: Linear(hidden_dim, K)` emits per-sample softmax-normalised bin *widths* (summing to `depth_range_max − depth_range_min`), which are cumulatively summed to produce K+1 edges in `[z_min, z_max]` and K midpoint centres. Classification logits from `self.depth_out` are soft-argmaxed over these per-sample centres; loss is SORD soft CE + `λ_reg = 0.3` SmoothL1 on the expectation (same as Design 002).

**Starting Point:** `baseline/`

---

## Overview

Design 003 inherits the entire Design 002 hybrid loss formulation (SORD soft CE + `λ_reg = 0.3` SmoothL1 on expectation) and additionally adopts AdaBins-style per-sample bin widths (Bhat et al. CVPR 2021). Instead of using fixed log-uniform bin centres for every sample, the model predicts PER-SAMPLE bin widths from the same pelvis token `decoded[:, 0, :]` via a second linear head `depth_bins_head: Linear(hidden_dim, K)`. The width predictions are softmax-normalised (so they sum to 1), multiplied by `depth_range_max − depth_range_min` (so they sum to the full depth range), cumulatively summed to produce K+1 edges in `[z_min, z_max]`, and then the midpoints of consecutive edges give K per-sample bin centres.

The classification logits from `self.depth_out` are unchanged in architecture; the only change is that the soft-argmax uses per-sample bin centres. The SORD soft CE is computed against the **per-sample** bin centres (log-space), so the loss is fully data-adaptive: close subjects will learn tighter bins around their GT depth, far subjects learn broader bins.

All other code paths are identical to the baseline:
- UV head, joint head, depth recovery (`recover_pelvis_3d`), `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, hooks, optimizer, LR schedule, seed, batch size, accumulation, evaluation.
- The returned dict still contains `'pelvis_depth': (B, 1)` metric-scalar tensor.

---

## Files to Change

1. `pose3d_transformer_head.py` — the shared six-kwarg signature is identical to Design 001/002. In `__init__`, when `depth_head_type == 'classification_adaptive'`, allocate a SECOND linear head `self.depth_bins_head = nn.Linear(hidden_dim, num_depth_bins)` for per-sample bin-width prediction. In `forward()`, when classification_adaptive mode is active, compute per-sample bin centres from `depth_bins_head(pelvis_token)` and use them in the soft-argmax AND pass them to `loss()` via `depth_bin_centres` in log-space (so SORD can compute targets in log-space consistently). In `loss()`, use the per-sample `log_bin_centres` in the SORD target computation.
2. `config.py` — same six new kwargs as Design 001/002; different `depth_head_type='classification_adaptive'` value.
3. `pelvis_utils.py` — **no change**.

No new top-level imports. The `import torch.nn.functional as F` local-scope import inside `loss()` (for the SmoothL1 aux term) is the same as Design 002.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `__init__` — Design-003-specific values and the second head

Same six kwargs as Design 001/002, with:
- `depth_head_type = 'classification_adaptive'`
- `num_depth_bins = 64`
- `depth_range_min = 1.0`
- `depth_range_max = 15.0`
- `depth_soft_label_sigma = 1.5`
- `depth_aux_reg_weight = 0.3`

Inside the conditional head-allocation block in `__init__`, in addition to allocating `self.depth_out = nn.Linear(hidden_dim, K)` (same as Design 001/002), allocate a SECOND head when adaptive mode is active:

```python
self.joints_out = nn.Linear(hidden_dim, 3)
if self.depth_head_type == 'regression':
    self.depth_out = nn.Linear(hidden_dim, 1)
else:
    self.depth_out = nn.Linear(hidden_dim, self.num_depth_bins)
    log_min = math.log(self.depth_range_min)
    log_max = math.log(self.depth_range_max)
    log_centres = torch.linspace(log_min, log_max, self.num_depth_bins)
    self.register_buffer('log_bin_centres', log_centres, persistent=False)
    if self.depth_head_type == 'classification_adaptive':
        # AdaBins-style second head: predicts per-sample bin WIDTHS.
        self.depth_bins_head = nn.Linear(hidden_dim, self.num_depth_bins)
self.uv_out = nn.Linear(hidden_dim, 2)
```

Constraints:
- `self.depth_bins_head` is a standard `Linear(hidden_dim, K)` and participates in trunc-normal init via the same loop as the other heads (see §2).
- The `self.log_bin_centres` buffer is still registered as the *default* bin centres; it is used as a **fallback** / for the FIXED modes (Design 001/002) and as reference for the adaptive-mode SORD target computation (see §5 below). In Design 003 `forward()`, the log-centres used for soft-argmax are PER-SAMPLE (computed from `depth_bins_head`); the buffer is unused at forward time in adaptive mode but retained for serialisation consistency.
- Parameter count delta vs. Design 002: `+16448` additional float32 weights (`Linear(256, 64)` = `256 × 64 + 64 = 16448`). Total delta vs. baseline: `+16191 + 16448 = +32639` params. Still < 2% of the head.

#### 2. `_init_head_weights` — extend to cover `self.depth_bins_head`

Currently the loop is:

```python
for m in [self.joints_out, self.depth_out, self.uv_out]:
    nn.init.trunc_normal_(m.weight, std=0.02)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
```

Extend to also initialise `self.depth_bins_head` when it exists:

```python
modules_to_init = [self.joints_out, self.depth_out, self.uv_out]
if self.depth_head_type == 'classification_adaptive':
    modules_to_init.append(self.depth_bins_head)
for m in modules_to_init:
    nn.init.trunc_normal_(m.weight, std=0.02)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
```

The zero-bias + trunc-normal weights produce near-uniform softmax over bin widths at init → bin centres at init are approximately equal to the fixed log-uniform centres `self.log_bin_centres.exp()` (since uniform widths in linear space, cumsum, then midpoints = arithmetic uniform, NOT log-uniform). See "Risk" section below for why this is acceptable.

#### 3. `forward()` — adaptive bin centre computation + soft-argmax

Extend the classification `else` branch from Design 001/002:

```python
if self.depth_head_type == 'regression':
    pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
    depth_logits = None
    depth_bin_centres = None
else:
    depth_logits = self.depth_out(pelvis_token)  # (B, K)
    if self.depth_head_type == 'classification_adaptive':
        # ── AdaBins-style per-sample bin widths. ──
        width_logits = self.depth_bins_head(pelvis_token)  # (B, K)
        widths = torch.softmax(width_logits, dim=-1)        # (B, K), sum to 1 per sample
        widths = widths * (self.depth_range_max - self.depth_range_min)  # (B, K), sum to R
        # Cumulative sum gives K+1 edges in [z_min, z_max]. Shift by z_min.
        edges = torch.cumsum(widths, dim=-1)                # (B, K), values in (0, R]
        # Prepend a zero column so edges has shape (B, K+1) with edges[:,0]=0
        zero_col = torch.zeros(widths.size(0), 1, device=widths.device, dtype=widths.dtype)
        edges = torch.cat([zero_col, edges], dim=-1)        # (B, K+1), values in [0, R]
        edges = edges + self.depth_range_min                # (B, K+1), values in [z_min, z_max]
        # Bin centres = midpoints of consecutive edges.
        bin_centres = 0.5 * (edges[:, :-1] + edges[:, 1:])  # (B, K)
        depth_bin_centres = bin_centres                     # (B, K) per-sample
    else:
        # FIXED log-uniform bin centres (Design 001/002).
        bin_centres = self.log_bin_centres.exp()            # (K,)
        depth_bin_centres = bin_centres.unsqueeze(0).expand(depth_logits.size(0), -1)  # (B, K)
    probs = torch.softmax(depth_logits, dim=-1)             # (B, K)
    expected_depth = (probs * depth_bin_centres).sum(dim=-1, keepdim=True)  # (B, 1)
    pelvis_depth = expected_depth
```

Constraints:
- The per-sample bin centres in adaptive mode are in **linear** depth space (metres), NOT log space. The AdaBins paper uses linear-space widths because the range `[z_min, z_max] = [1, 15]` is small enough that linear adaptive widths are well-posed.
- `widths` is the result of a softmax → strictly positive → `cumsum` is monotonically increasing → edges are strictly increasing within each sample → `bin_centres` are well-defined midpoints.
- The `torch.zeros(widths.size(0), 1, ...)` prepend is needed because `cumsum` returns K values but we need K+1 edges (including both `z_min` and `z_max`).
- `edges[:, 0] = z_min` and `edges[:, K] = z_min + sum(widths) = z_min + (z_max - z_min) = z_max`. Both endpoints are exactly respected, **per sample**.
- The returned `depth_bin_centres` shape is `(B, K)` in all classification modes — already broadcast across the batch. The `loss()` function uses this per-sample tensor directly.
- Gradient flows through both heads:
  - `depth_logits` → `softmax` → `probs` → `expected_depth`
  - `width_logits` → `softmax` → `widths` → `cumsum` → `edges` → `bin_centres` → `expected_depth`
  - Both paths are fully differentiable. The adaptive bin-width head receives gradient from BOTH the SmoothL1 aux loss (via the expectation) AND the SORD soft CE (via the `log(bin_centres)` used in the SORD target — see §5 below).

The returned dict is the same as Design 002:

```python
return {
    'joints': joints,
    'pelvis_depth': pelvis_depth,
    'pelvis_uv': pelvis_uv,
    'depth_logits': depth_logits,
    'depth_bin_centres': depth_bin_centres,
}
```

#### 4. `_init_head_weights` / head init consistency (cross-reference)

See §2 above — `self.depth_bins_head` is added to the init loop when adaptive mode is active.

#### 5. `loss()` — SORD soft CE against PER-SAMPLE centres + SmoothL1 hybrid

The loss computation is modified to use `pred['depth_bin_centres']` (per-sample centres from `forward()`) instead of the fixed `self.log_bin_centres` buffer. Concretely:

```python
if self.depth_head_type == 'regression':
    losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
        pred['pelvis_depth'], gt_depth)
else:
    # Per-sample bin centres (in metres). Shape (B, K).
    bin_centres = pred['depth_bin_centres']  # (B, K)
    K = bin_centres.size(-1)

    # Use per-sample centres in LOG space for SORD target computation.
    # This handles both fixed-log-uniform (Design 001/002) and adaptive
    # (Design 003) modes uniformly: log(bin_centres) per sample.
    log_bin_centres_per_sample = bin_centres.clamp(
        min=self.depth_range_min * 1e-3).log()  # (B, K)

    # Estimate per-sample sigma_log:
    #   In fixed mode, bin widths are uniform in log space: sigma_log = const.
    #   In adaptive mode, widths vary; use the median absolute difference
    #   between consecutive log-centres as the sample-level bin width:
    if self.depth_head_type == 'classification_adaptive':
        log_diffs = (log_bin_centres_per_sample[:, 1:]
                     - log_bin_centres_per_sample[:, :-1]).abs()  # (B, K-1)
        bin_width_log_per_sample = log_diffs.median(dim=-1, keepdim=True).values  # (B, 1)
        sigma_log = self.depth_soft_label_sigma * bin_width_log_per_sample  # (B, 1)
    else:
        # Fixed bins: constant bin width in log space (same as Design 001/002).
        log_min = math.log(self.depth_range_min)
        log_max = math.log(self.depth_range_max)
        bin_width_log = (log_max - log_min) / max(K - 1, 1)
        sigma_log = torch.full((bin_centres.size(0), 1),
                               self.depth_soft_label_sigma * bin_width_log,
                               device=bin_centres.device,
                               dtype=bin_centres.dtype)  # (B, 1)

    # GT depth, clamped, log'd.
    z_gt = gt_depth.clamp(min=self.depth_range_min,
                          max=self.depth_range_max)   # (B, 1)
    log_z_gt = z_gt.log()                             # (B, 1)

    # SORD target logits: -(log_centre - log_z_gt)^2 / (2 * sigma_log^2)
    log_diff = log_bin_centres_per_sample - log_z_gt  # (B, K)  (broadcasts log_z_gt)
    target_logits = -(log_diff ** 2) / (2.0 * sigma_log ** 2)  # (B, K)
    target = torch.softmax(target_logits, dim=-1)     # (B, K)

    # Detach target from autograd: it depends on bin_centres (and in adaptive
    # mode, on the width head). We want the SORD target to be a fixed
    # (per-step) reference; gradient should flow ONLY through `log_probs`,
    # NOT back through the target into the width head.
    target = target.detach()

    log_probs = torch.log_softmax(pred['depth_logits'], dim=-1)  # (B, K)
    ce_per_sample = -(target * log_probs).sum(dim=-1)             # (B,)
    L_depth_ce = ce_per_sample.mean()
    losses['loss/depth/train'] = self.loss_weight_depth * L_depth_ce

    # Auxiliary SmoothL1 regression on expectation (Design 002/003 active).
    if self.depth_aux_reg_weight > 0.0:
        import torch.nn.functional as F
        L_depth_reg = F.smooth_l1_loss(
            pred['pelvis_depth'],
            gt_depth.to(pred['pelvis_depth'].device),
            reduction='mean', beta=0.05)
        losses['loss/depth_reg/train'] = self.depth_aux_reg_weight * L_depth_reg
```

Constraints:
- The SORD target MUST be `.detach()`ed in adaptive mode so that the gradient of `L_depth_ce` flows ONLY through `log_softmax(depth_logits)` — NOT back into the `depth_bins_head` via the target. Otherwise the target would "chase" the prediction: a moving target is unstable and converges to a degenerate solution (e.g., all width mass piled up at one bin). The `.detach()` is ALSO applied in fixed mode — it has no effect there (since `log_bin_centres` is a buffer anyway), but unifies the code path. This is a load-bearing correctness invariant.
- The per-sample `sigma_log` in adaptive mode uses the **median** of absolute consecutive log-centre differences: this is robust to outliers (a single huge width wouldn't explode sigma). The `keepdim=True` keeps `sigma_log` shape `(B, 1)` for broadcasting.
- Gradient flow paths to the two heads:
  - `depth_out.weight` receives gradient from:
    - CE (via `log_softmax(depth_logits)`).
    - SmoothL1 aux (via `probs = softmax(depth_logits)` in the expectation).
  - `depth_bins_head.weight` receives gradient from:
    - SmoothL1 aux (via `bin_centres` in the expectation).
    - NOT from CE (target is detached — see constraint above).
  - This is the correct AdaBins gradient structure: the width head is trained via the SmoothL1 signal, the probability head is trained via both CE and SmoothL1.
- `bin_centres.clamp(min=self.depth_range_min * 1e-3)` is defensive: in pathological early training, cumsum of softmax widths could be nearly zero for a specific bin; clamping at `1e-3 × z_min` prevents `log(0)` NaN. In practice bin centres are always at least `0.5 × (min_gap)` from zero because of softmax positivity and `+ depth_range_min` shift. The clamp is a safety valve.

The MPJPE `with torch.no_grad():` block is UNCHANGED. It reads `pred['pelvis_depth']` (the expectation scalar) — shape `(B, 1)`.

#### 6. `predict()` — unchanged

Same as baseline and Designs 001/002.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`, add the six new kwargs at the end (after `loss_weight_uv=1.0,`):

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
    depth_head_type='classification_adaptive',
    num_depth_bins=64,
    depth_range_min=1.0,
    depth_range_max=15.0,
    depth_soft_label_sigma=1.5,
    depth_aux_reg_weight=0.3,
),
```

All new values are `str` / `int` / `float` literals — fully MMEngine-config compliant.

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
| **depth_head_type** | **'classification_adaptive' (new)** |
| **num_depth_bins** | **64 (new)** |
| **depth_range_min** | **1.0 (new)** |
| **depth_range_max** | **15.0 (new)** |
| **depth_soft_label_sigma** | **1.5 (new)** |
| **depth_aux_reg_weight** | **0.3 (new; activates SmoothL1 on expectation)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

All 24 constraints from Design 001 and all extra constraints from Design 002 (25–30) apply unchanged to Design 003, with these additional constraints:

31. The second head `self.depth_bins_head = nn.Linear(hidden_dim, num_depth_bins)` MUST be allocated ONLY when `depth_head_type == 'classification_adaptive'`. In Designs 001/002 this head MUST NOT exist (and attempting to access it would raise `AttributeError`).
32. `self.depth_bins_head` MUST be included in the `_init_head_weights()` init loop ONLY when adaptive mode is active.
33. In `forward()` adaptive mode, the per-sample bin widths MUST be produced by `torch.softmax(width_logits, dim=-1) * (z_max - z_min)`. Using raw softmax without the `* (z_max - z_min)` scaling would give widths summing to 1 m total, NOT to `z_max - z_min = 14 m`, and the resulting bin centres would be bunched near `z_min`. This is a load-bearing correctness invariant.
34. The cumulative-sum edge computation MUST prepend a zero column to `cumsum(widths)` to produce K+1 edges. Omitting the zero prepend gives only K edges (starting from the first width, not from zero), which shifts all bin centres to the right by half a bin and misses z_min entirely.
35. The `+ self.depth_range_min` shift on edges MUST be applied AFTER the cumsum-and-prepend, so the final edges are in `[z_min, z_max]` (not `[0, z_max − z_min]`).
36. Bin centres MUST be the midpoints of consecutive edges: `0.5 * (edges[:, :-1] + edges[:, 1:])`. Using bin left-edges (e.g., `edges[:, :-1]`) is a common bug that biases the expectation toward smaller z; using right-edges biases the other way. The midpoint is the correct choice.
37. The SORD target tensor `target` in `loss()` MUST be `.detach()`ed. In adaptive mode, forgetting to detach causes the target to depend on `bin_centres` (which depend on `depth_bins_head.weight`), which would create a degenerate "moving target" loss. This is a load-bearing correctness invariant.
38. Per-sample `sigma_log` in adaptive mode is computed as `depth_soft_label_sigma × median(|consecutive log-centre diffs|)`. Using the mean instead of median works but is less robust to a single huge width; using a global `sigma_log` (fixed formula) destroys the adaptive benefit. Median is the chosen compromise.
39. Parameter count delta vs. baseline: `+32639` float32 weights (~130 kB). Still under 0.02% of total model.
40. Total emitted loss keys: `loss/joints/train`, `loss/depth/train` (SORD soft CE against per-sample centres), `loss/uv/train`, `loss/depth_reg/train` (SmoothL1 × 0.3 on expectation). Four keys total, same as Design 002.
41. `self.log_bin_centres` buffer is still registered (for serialisation consistency across all classification designs) but is NOT used in `forward()` adaptive mode.
42. `self.depth_bins_head` output `width_logits` (shape `(B, K)`) is NOT part of `pred['depth_logits']` — that key refers EXCLUSIVELY to the classification logits from `self.depth_out`. Do NOT confuse the two.
43. Per-sample bin centres are in **linear** depth space (metres). The SORD target computation in `loss()` converts them to log-space via `log()` — this conversion MUST be done inside `loss()`, not passed pre-log from `forward()`. The `pred['depth_bin_centres']` key exposes the LINEAR-space centres.

---

## Expected Behaviour After Change

- `forward()` produces the same `(B, 1)` `pelvis_depth` tensor as Designs 001/002 and baseline; same shape contract for downstream code.
- Training emits FOUR loss scalars: `loss/joints/train`, `loss/depth/train` (CE against per-sample SORD target), `loss/uv/train`, `loss/depth_reg/train` (SmoothL1 on expectation × 0.3).
- At init (epoch 0 step 0):
  - `depth_bins_head` weights are trunc-normal std=0.02; zero-biased. `width_logits` ≈ 0 for all K bins → softmax near-uniform → widths ≈ `(z_max - z_min) / K` ≈ 0.219 m each → edges linearly spaced in `[1, 15]` m → bin centres at `[1.109, 1.328, 1.547, …, 14.89]` m (arithmetic, NOT geometric). This is DIFFERENT from fixed log-uniform init (where centres are `[1.000, 1.043, 1.089, …, 15.0]` m) — adaptive-init bin centres are UNIFORM in linear space, not log space. For K=64 bins over [1, 15] m, this is a reasonable starting grid.
  - CE ~`log(64) ≈ 4.16`.
  - Expected depth at init with uniform bin widths: arithmetic mean of linearly-spaced centres = `(1 + 15) / 2 = 8 m`. Higher than Design 002's init expectation (~5.5 m) because the bin centres are arithmetically uniform (vs. log-uniform, which concentrates centres near z_min).
  - SmoothL1 aux at init: `0.3 × |8 − 4| = 1.2` — larger than Design 002's ~0.45. This adds a stronger pull toward the dataset mean at init, which is beneficial for fast early convergence.
- Convergence expectations: Design 003 should converge to a bin-width distribution that concentrates width where the data is densest (≈ 3–6 m for BEDLAM2), achieving sub-bin-width metric resolution in that range.
- `BedlamMPJPEMetric` sees the same `(B, 1)` depth scalar and computes `mpjpe_pelvis_val` and `mpjpe_abs_val` identically.
- Extra parameter count vs. baseline: `+32639` trainable params (`Linear(256, 64) × 2`).
- Extra runtime cost per forward/backward: two additional `Linear(256, 64)` calls (one for `depth_out`, one for `depth_bins_head`), one extra softmax, one cumsum, one midpoint computation — all on `(B, 64)` tensors. Total overhead < 0.2 ms per step, negligible vs. backbone.
- Extra memory: `(B, K+1)` edges tensor = 65 × B × 4 B ≈ 1 kB; negligible.
- Expected result vs. baseline: `mpjpe_pelvis_val` improves by 7–12 mm (target < 165 mm; more aggressive than Designs 001/002 because adaptive bins provide per-sample resolution). `mpjpe_abs_val` improves proportionally (target < 430 mm). `mpjpe_body_val`, `mpjpe_rel_val`, `mpjpe_hand_val` neutral.
- Expected outcome vs. Design 002: 2–4 mm further improvement in pelvis MPJPE, conditional on the model being able to learn useful adaptive bin-width structure within 20 epochs. This is the *most ambitious* of the three designs.

---

## Rationale Summary

- **Why adaptive bins?** In fixed log-uniform bins, every sample uses the same bin centres. If a sample's pelvis is at z=4 m but the nearest log-uniform bin centre is at 3.87 m (geometric mean), the expectation can only be pulled to z=3.87 m via probability mass; the resolution at z=4 m is limited by the bin width at that location. Adaptive bins allow the model to REPOSITION the bin centres per-sample to concentrate resolution where the GT is most likely. For BEDLAM2's depth distribution (concentrated 2–8 m), adaptive bins can place the majority of their K=64 "slots" in that range, achieving effective resolution of ≈ 10 cm instead of the 36 cm achievable with fixed log-uniform bins at z=15 m.
- **Why SORD against per-sample centres?** If the target distribution were computed against fixed log-uniform centres while the prediction uses per-sample centres, the CE gradient would be inconsistent (target and prediction are in different "bin spaces"). Using per-sample centres in both keeps CE sensible.
- **Why detach the target?** With adaptive centres, if the target were NOT detached, the `depth_bins_head` could collapse all width mass into a single bin near z_gt, making the SORD target a delta function at that bin and trivially minimising CE to zero — but with no useful resolution elsewhere. Detaching the target keeps the width head trained ONLY by the SmoothL1-on-expectation signal (which prefers a calibrated expectation) and the probability head trained by the SORD CE (which prefers a well-calibrated distribution shape). These signals are complementary and non-degenerate.
- **Why not pre-softmax the width head with `F.softplus`?** AdaBins uses softmax (which is what Design 003 uses) because softmax has the convenient property of summing to 1, making the cumulative-sum edge computation trivially bounded. Softplus widths would require an additional per-sample normalisation to sum to `z_max − z_min`, which is strictly more computation and not obviously better.

---

## Risk and Mitigation Specific to Design 003

- **Degenerate bin-width collapse**: If the SORD target were NOT detached, the width head could collapse all mass into one bin. Mitigation: target is detached (constraint 37).
- **Init-mismatch from arithmetic vs. log-uniform centres**: At init, adaptive-mode centres are arithmetically uniform (≈ [1.11, 1.33, …, 14.89]) rather than log-uniform. This puts fewer centres near z=1 m than the fixed log-uniform mode does. For BEDLAM2 (where subjects are rarely at z=1 m), this is actually BETTER — the init bins are biased toward the mass of the data. No mitigation needed.
- **20-epoch training budget may be insufficient**: AdaBins results in the depth estimation literature are typically trained for 100+ epochs. With only 20 epochs, the width head may not fully converge to the optimal per-sample resolution. This is the main caveat for Design 003 — it is the most ambitious design and may show smaller gains than Design 002 if the width head is under-trained. If observed at validation, this is a useful empirical datapoint, not a bug.
- **Second head parameter overhead**: `+16448` additional params on a head that is already tiny. No practical concern.
- **Config kwarg literal**: all six new kwargs are str/int/float literals; no Python `import` needed.
- **`.detach()` correctness**: the `target = target.detach()` line MUST appear AFTER `target = torch.softmax(target_logits, dim=-1)` and BEFORE `target` is used in the CE computation. Putting it before the softmax would still work (softmax + detach is equivalent to detach + softmax for autograd purposes since softmax is purely a forward op), but the ordering used here is clearer.
- **Numerical stability of `log()` on per-sample centres**: the `bin_centres.clamp(min=self.depth_range_min * 1e-3).log()` prevents `log(0)` NaN. In practice bin centres are always ≥ `z_min / (2K) ≈ 0.008 m` even in pathological early training, but the clamp is a cheap safety valve.
- **Backward compatibility**: The additional `loss/depth_reg/train` key auto-appears in `MetricsCSVHook`; same as Design 002.
- **Composition with prior ideas**: fully orthogonal. idea002 (decoupled pelvis query) composes cleanly because the pelvis query is already shared between the depth and UV heads; fan-out to two depth heads is a straightforward extension. idea005 (uncertainty weighting) rescales the scalar `loss/depth/train`. idea010 (2D reprojection) uses the expectation scalar, which is fully differentiable in Design 003.
- **Runtime/memory**: extra `Linear(256, 64)` forward + softmax + cumsum + midpoint = < 0.2 ms/step. Negligible.
- **Early training instability check**: the Builder SHOULD verify at iter 50 that `loss/depth/train` and `loss/depth_reg/train` both decrease. ALSO verify that the learned bin widths (if printed for debugging) do not collapse (i.e., `widths.std()` per sample should remain > 0 after a few hundred iters, not degenerate to a single non-zero bin).
- **Detached-target failure mode check**: if the Builder accidentally omits the `.detach()` call, the Builder would observe `widths.std()` collapsing to near zero after a few hundred iters and `loss/depth/train` converging toward 0 while `mpjpe_pelvis_val` stays poor — a distinctive and diagnosable failure signature.
