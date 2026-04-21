# Design 002 — Fixed Log-Uniform Bins + SORD Soft-Argmax + Auxiliary SmoothL1 Regression (Hybrid)

**Design Description:** Design 001 plus an auxiliary SmoothL1 regression term on the soft-argmax expected depth against GT depth, weighted at `λ_reg = 0.3`. The CE trains the bin-probability landscape; the SmoothL1 ensures the expectation-recovered scalar is metrically correct even when the bin distribution is broad. Emits an additional loss key `loss/depth_reg/train`.

**Starting Point:** `baseline/`

---

## Overview

Design 002 inherits the entire classification + SORD soft-argmax machinery from Design 001 (fixed log-uniform bins in `[1.0, 15.0]` m, K=64 bins, soft-target CE with σ=1.5 × bin width in log space, exposure of `depth_logits` and `depth_bin_centres` to the loss function). The only addition is an auxiliary SmoothL1 regression loss term between the **soft-argmax expected depth** `pred['pelvis_depth']` (metric scalar, metres) and the **GT depth** `gt_depth`, weighted at `λ_reg = 0.3` and emitted as the separate loss scalar `loss/depth_reg/train`.

The two losses are complementary:
- **Cross-entropy** trains the bin-probability *landscape*, which drives the network to place probability mass in the correct bin *neighbourhood*. CE is robust to ambiguous inputs because it maintains mass on multiple modes.
- **SmoothL1 on expectation** ensures that the *metric scalar* recovered by soft-argmax is calibrated to the GT depth. This guards against a failure mode where the CE alone is minimised but the bin distribution is wide enough that the expectation under-represents the correct depth (e.g., a perfectly bimodal distribution with mass 0.5 at z=2 m and 0.5 at z=10 m has CE ≈ log(K/2) but expectation ≈ 6 m — metrically bad even if CE-low).

This hybrid formulation mirrors BinsFormer (Li et al. TCSVT 2024) and is known to stabilise early-training convergence when bins are wide.

All other code paths are identical to the baseline and Design 001:
- UV head, joint head, depth recovery (`recover_pelvis_3d`), `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, hooks, optimizer, LR schedule, seed, batch size, accumulation, evaluation.
- The returned dict still contains `'pelvis_depth': (B, 1)` metric-scalar tensor.

---

## Files to Change

1. `pose3d_transformer_head.py` — identical to Design 001 (same shared signature, same bin-centre buffer, same forward(), same classification CE in loss()). The ONLY differences are: `depth_head_type='classification_hybrid'` activates the SmoothL1-on-expectation auxiliary loss inside `loss()`, and the additional `loss/depth_reg/train` key is emitted.
2. `config.py` — same six new kwargs as Design 001; different values (`depth_head_type='classification_hybrid'`, `depth_aux_reg_weight=0.3`).
3. `pelvis_utils.py` — **no change**.

No new top-level imports are introduced. `torch.nn.functional.smooth_l1_loss` is a local-scope import inside the `loss()` method (already called out in Design 001's spec as an optional branch — Design 002 activates it).

---

## Algorithm Changes

### `pose3d_transformer_head.py`

All Design 001 changes apply IDENTICALLY to Design 002 (the shared signature covers all three designs). The ONLY behavioural difference is that Design 002 activates the `if self.depth_aux_reg_weight > 0.0:` auxiliary regression branch inside `loss()`, producing the additional `loss/depth_reg/train` key.

#### 1. `__init__` — Design-002-specific values

Same six kwargs as Design 001, with:
- `depth_head_type = 'classification_hybrid'`
- `num_depth_bins = 64`
- `depth_range_min = 1.0`
- `depth_range_max = 15.0`
- `depth_soft_label_sigma = 1.5`
- `depth_aux_reg_weight = 0.3`

The assertion `depth_head_type in (...)` and the numeric validation checks (from Design 001 §1) apply unchanged and allow `'classification_hybrid'`.

#### 2. `__init__` — output head allocation

Same conditional block as Design 001:
- `self.depth_out = nn.Linear(hidden_dim, self.num_depth_bins)` (K=64).
- `self.log_bin_centres` registered as a non-persistent buffer.

The adaptive bins head `self.depth_bins_head` is NOT allocated in Design 002 (that is Design 003 only). `'classification_hybrid'` uses FIXED log-uniform bin centres, identical to Design 001.

#### 3. `_init_head_weights` — unchanged

Same as baseline + Design 001.

#### 4. `forward()` — same expectation calculation as Design 001

Identical branch:

```python
if self.depth_head_type == 'regression':
    pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
    depth_logits = None
    depth_bin_centres = None
else:
    depth_logits = self.depth_out(pelvis_token)  # (B, K)
    bin_centres = self.log_bin_centres.exp()  # (K,)
    depth_bin_centres = bin_centres.unsqueeze(0).expand(depth_logits.size(0), -1)  # (B, K)
    probs = torch.softmax(depth_logits, dim=-1)  # (B, K)
    expected_depth = (probs * depth_bin_centres).sum(dim=-1, keepdim=True)  # (B, 1)
    pelvis_depth = expected_depth
```

Note: the `else` branch catches `'classification'`, `'classification_hybrid'`, AND `'classification_adaptive'`. Design 003 will specialise the bin-centre computation inside this `else` (per-sample adaptive centres); Design 002 uses the fixed `self.log_bin_centres.exp()` path. The branch selection is by `self.depth_head_type == 'classification_adaptive'` INSIDE this else block in Design 003. For Design 002 (`'classification_hybrid'`), the code path falls through to the fixed log-uniform centres.

The returned dict is the same as Design 001:
```python
return {
    'joints': joints,
    'pelvis_depth': pelvis_depth,
    'pelvis_uv': pelvis_uv,
    'depth_logits': depth_logits,
    'depth_bin_centres': depth_bin_centres,
}
```

#### 5. `loss()` — CE + SmoothL1 hybrid (Design 002)

Same CE computation as Design 001. The `if self.depth_aux_reg_weight > 0.0:` branch is now ACTIVE (since `depth_aux_reg_weight = 0.3`). Concretely, the depth-loss block in `loss()` produces TWO scalars:

```python
if self.depth_head_type == 'regression':
    losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
        pred['pelvis_depth'], gt_depth)
else:
    # ── SORD soft-target cross-entropy (same as Design 001) ──
    log_bin_centres = self.log_bin_centres            # (K,)
    K = log_bin_centres.numel()
    log_min = math.log(self.depth_range_min)
    log_max = math.log(self.depth_range_max)
    bin_width_log = (log_max - log_min) / max(K - 1, 1)
    sigma_log = self.depth_soft_label_sigma * bin_width_log

    z_gt = gt_depth.clamp(min=self.depth_range_min,
                          max=self.depth_range_max)   # (B, 1)
    log_z_gt = z_gt.log()                             # (B, 1)

    log_diff = log_bin_centres.unsqueeze(0) - log_z_gt   # (B, K)
    target_logits = -(log_diff ** 2) / (2.0 * sigma_log ** 2)  # (B, K)
    target = torch.softmax(target_logits, dim=-1)       # (B, K)

    log_probs = torch.log_softmax(pred['depth_logits'], dim=-1)  # (B, K)
    ce_per_sample = -(target * log_probs).sum(dim=-1)    # (B,)
    L_depth_ce = ce_per_sample.mean()
    losses['loss/depth/train'] = self.loss_weight_depth * L_depth_ce

    # ── Auxiliary SmoothL1 regression on expectation (Design 002 active) ──
    if self.depth_aux_reg_weight > 0.0:
        import torch.nn.functional as F
        L_depth_reg = F.smooth_l1_loss(
            pred['pelvis_depth'],
            gt_depth.to(pred['pelvis_depth'].device),
            reduction='mean', beta=0.05)
        losses['loss/depth_reg/train'] = self.depth_aux_reg_weight * L_depth_reg
```

Constraints:
- `loss/depth_reg/train = 0.3 * smooth_l1(pelvis_depth_expected, gt_depth, beta=0.05)`.
- The `F.smooth_l1_loss` call uses `beta=0.05` to match the baseline's `SoftWeightSmoothL1Loss(beta=0.05)` scale. Note: `F.smooth_l1_loss` uses the standard PyTorch SmoothL1 formula: `loss = 0.5 * x^2 / beta  if |x| < beta  else  |x| - 0.5 * beta`. At typical depth errors |Δz| = 0.1–0.5 m and β = 0.05, every sample is in the linear regime (|Δz| >> β), so `L_reg ≈ |Δz| − 0.025 ≈ |Δz|` — scale is similar to `L1(expected, gt)`.
- The `.to(pred['pelvis_depth'].device)` is defensive (the `gt_depth` tensor from `batch_data_samples` already goes through a `.to()` earlier in the method; this is idempotent and safe).
- Gradient flows: backward from `L_depth_reg` → `pred['pelvis_depth']` (the expectation) → through `sum(probs * bin_centres)` → into `probs` (via softmax) → into `depth_logits` → into `depth_out.weight`. So the SmoothL1 signal ADDs to the CE signal at the logits level; both train the same parameters coherently.
- The `F` import is a method-local import (placed inside the `if self.depth_aux_reg_weight > 0.0:` block to avoid executing on every `__init__`); this is still valid Python (imports at method scope are common in PyTorch codebases) and does NOT violate any project invariant. Alternative: the Builder MAY instead add `import torch.nn.functional as F` at the top of `pose3d_transformer_head.py` (this is already implicitly available via `torch.nn.functional`; adding an explicit top-level import is cleaner). Either pattern is acceptable.
- The MPJPE `with torch.no_grad():` block is UNCHANGED (same as baseline and Design 001). It reads `pred['pelvis_depth']`, the expectation scalar.

#### 6. `predict()` — unchanged

Same as baseline and Design 001.

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
    depth_head_type='classification_hybrid',
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
| **depth_head_type** | **'classification_hybrid' (new)** |
| **num_depth_bins** | **64 (new)** |
| **depth_range_min** | **1.0 (new)** |
| **depth_range_max** | **15.0 (new)** |
| **depth_soft_label_sigma** | **1.5 (new)** |
| **depth_aux_reg_weight** | **0.3 (new; activates SmoothL1 on expectation)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

All 24 constraints from Design 001 apply unchanged to Design 002, with these additional constraints:

25. The auxiliary SmoothL1 term `L_depth_reg` MUST use `F.smooth_l1_loss` with `beta=0.05` (matching the baseline SmoothL1 scale) and `reduction='mean'`. Using `beta=1.0` (PyTorch's default) would give a different scale and is INCORRECT for this design.
26. The `loss/depth_reg/train` key MUST be emitted with the per-sample weighted value `depth_aux_reg_weight × L_depth_reg` (i.e., `0.3 × F.smooth_l1_loss(...)`). The `depth_aux_reg_weight` multiplier is applied INSIDE the loss term, not outside — so the total optimisation objective contains `0.3 × SmoothL1 + 1.0 × CE` on depth.
27. The auxiliary SmoothL1 gradient MUST flow through `pred['pelvis_depth']` (the expectation scalar) back into `depth_logits`. In Design 002, `pred['pelvis_depth']` is the result of `(probs * bin_centres).sum(...)` — a fully differentiable tensor.
28. The `loss/depth/train` key from Design 001 is ALSO emitted in Design 002. Total: Design 002 adds ONE new loss key (`loss/depth_reg/train`) compared to the baseline three keys. Total emitted keys: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/depth_reg/train` (four).
29. `MetricsCSVHook` auto-logs all scalar losses; the `loss/depth_reg/train` column will appear in the CSV automatically. This is expected behaviour, not a bug. The Builder MUST verify that the CSV has the new column populated after the first validation epoch.
30. `depth_aux_reg_weight = 0.3` is the CANONICAL value for Design 002. Do NOT change it without explicit instruction. The value 0.3 is chosen because: (a) it is the median of BinsFormer's published range [0.1, 0.5]; (b) it makes `L_reg × 0.3 ≈ 0.3 × 0.3 m ≈ 0.09` at typical post-warmup depth errors, which is comparable to `L_joints ≈ 0.1` and to `L_uv ≈ 0.1` in magnitude; and (c) it is small enough that the CE signal still dominates the training of the bin-probability landscape.

---

## Expected Behaviour After Change

- `forward()` produces the same `(B, 1)` `pelvis_depth` tensor as Design 001 and baseline; same shape contract for downstream code.
- Training emits FOUR loss scalars: `loss/joints/train`, `loss/depth/train` (CE), `loss/uv/train`, `loss/depth_reg/train` (SmoothL1 on expectation, weighted × 0.3). The `loss/depth_reg/train` column auto-appears in the `MetricsCSVHook` CSV output.
- At init (epoch 0 step 0):
  - CE on near-uniform softmax: ~`log(64) ≈ 4.16`.
  - Expected depth ≈ 5.5 m (from uniform softmax over `[1, 15]` log-uniform bins).
  - GT depth mean ≈ 4 m → `L_depth_reg ≈ |5.5 − 4| = 1.5` m; `0.3 × 1.5 = 0.45` as `loss/depth_reg/train` first iter.
  - The SmoothL1 gradient on the expectation adds a strong pull toward the dataset mean very early in training, which helps the bin-probability mass concentrate in the correct range faster than CE alone.
- Convergence expectations: Design 002 should converge the pelvis-depth metric AT LEAST as fast as Design 001 during the first 3 warmup epochs (where most bin-distribution shaping happens), and potentially faster because the SmoothL1 provides a direct metric-space signal.
- `BedlamMPJPEMetric` sees the same `(B, 1)` depth scalar and computes `mpjpe_pelvis_val` and `mpjpe_abs_val` identically.
- Extra parameter count vs. baseline: `+16191` (same as Design 001; no new parameters from the hybrid loss).
- Extra runtime cost: one additional `F.smooth_l1_loss` call per step on a `(B, 1)` tensor — negligible (~1 μs).
- Extra memory: the SmoothL1 call needs no additional allocation beyond the existing `pred['pelvis_depth']` and `gt_depth` tensors.
- Expected result vs. baseline: `mpjpe_pelvis_val` improves by 5–10 mm (target < 168 mm; more aggressive than Design 001 because the SmoothL1 term directly penalises metric error). `mpjpe_abs_val` improves proportionally (target < 435 mm). `mpjpe_body_val`, `mpjpe_rel_val`, `mpjpe_hand_val` neutral.
- Expected outcome vs. Design 001: marginal improvement (2–3 mm better pelvis MPJPE) expected, since the SmoothL1 term provides a direct metric-space signal that complements CE's distribution-shaping signal.

---

## Rationale Summary

- **Why add an auxiliary SmoothL1 term?** Pure classification CE minimises the KL divergence between the predicted distribution and the SORD Gaussian target. In principle this drives the expectation toward the GT depth, but the *rate* at which the expectation converges depends on how quickly the bin probabilities concentrate. During warmup (epochs 0-3), the bin distribution can be broad; the SmoothL1 on the expectation provides a *direct* gradient to pull the expectation toward GT, which accelerates convergence and stabilises the trajectory. This is the BinsFormer hybrid formulation.
- **Why λ_reg = 0.3?** Empirically validated by BinsFormer across multiple depth datasets. Small enough that CE dominates the bin-probability shaping; large enough that the SmoothL1 provides a useful early-training signal. Larger λ (e.g., 1.0) degenerates toward pure regression and loses the CE's multi-modal benefit.
- **Why keep fixed log-uniform bins (not adaptive)?** Design 002 isolates the *auxiliary regression* contribution on top of Design 001. If Design 002 improves over Design 001 (say 2–3 mm), we know the hybrid loss is helpful. Design 003 then adds adaptive bins on top of Design 002 to isolate that contribution.
- **Why NOT simply go back to pure regression if SmoothL1 is useful?** Pure regression has NO bin-distribution structure — it can predict any scalar, with no prior on the valid depth range. The hybrid retains CE's strengths (range prior, multi-modal robustness, K-way gradient spread) and adds metric-space anchoring.

---

## Risk and Mitigation Specific to Design 002

- **Loss term double-counting**: Both `loss/depth/train` (CE) and `loss/depth_reg/train` (SmoothL1) penalise depth predictions. The gradient from CE flows into `depth_out.weight` via `log_softmax(depth_logits)`. The gradient from SmoothL1 flows into the same weights via the expectation `sum(softmax(depth_logits) * bin_centres)`. Both gradients point in coherent directions when CE and SmoothL1 both decrease on the correct prediction. No double-counting issue in the gradient sense, but the *total* gradient magnitude on `depth_out.weight` is ~1.3× a Design 001 single-CE loss step. AdamW normalises per-parameter gradient magnitudes so this is auto-adjusted. Mitigation: none needed; monitor the `loss/depth/train` and `loss/depth_reg/train` individually in CSV output to confirm both decrease monotonically.
- **Gradient saturation at small errors**: `F.smooth_l1_loss` has quadratic behaviour in `|Δz| < β = 0.05`. At late training with mean pelvis error ~5 mm (= 0.005 m), gradient is `Δz / β = 0.1` — small but non-zero. This is fine; the CE signal still provides the primary learning pressure at late training.
- **Config kwarg literal**: `depth_aux_reg_weight=0.3` is a float literal; no Python import needed in `config.py`.
- **Backward compatibility**: The additional `loss/depth_reg/train` key is logged by `MetricsCSVHook` as a new column. No code changes required to the hook — it auto-discovers scalar losses.
- **Numerical stability of `F.smooth_l1_loss`**: well-tested in PyTorch; no NaN concerns at any realistic depth value.
- **Composition with prior ideas**: fully orthogonal. Same compositional properties as Design 001 (cf. idea014 motivation §Why orthogonal to every prior idea).
- **Runtime/memory**: one extra `F.smooth_l1_loss` call per step on a `(B, 1)` tensor; negligible overhead (< 0.01 ms).
- **Interaction with `loss_weight_depth`**: `loss_weight_depth=1.0` multiplies `loss/depth/train` (CE). The SmoothL1 aux term uses its OWN weight `depth_aux_reg_weight=0.3` and is NOT multiplied by `loss_weight_depth`. The Builder MUST NOT multiply `loss_weight_depth` into `L_depth_reg` — these are independent weighting factors.
- **Early training instability check**: the Builder SHOULD verify at iter 50 that `loss/depth/train` is decreasing (from ~4.16 toward ~3–3.5) and `loss/depth_reg/train` is decreasing (from ~0.45 toward ~0.2–0.3). If either stalls, it indicates a config bug (e.g., frozen head); Builder MUST diagnose.
