**Idea Name:** Per-Joint Online Difficulty Weighting for Body Joint Loss

**Approach:** Maintain an exponential moving average of per-joint absolute prediction error during training, and use these per-joint difficulty estimates to compute adaptive loss weights that up-weight consistently-hard joints and down-weight easy joints, focusing gradient signal where it is most needed without requiring any architectural changes.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Uniform Joint Loss Bottleneck

The baseline head applies a single `SoftWeightSmoothL1Loss` uniformly across all 22 body joints:

```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

This treats all 22 body joints as equally important and equally difficult. In practice, joints vary dramatically in their prediction difficulty:

- **Easy joints**: spine/torso joints (indices 0–4) are large, anatomically prominent, and well-constrained by global pose. They converge quickly.
- **Hard joints**: distal limb joints such as wrists (indices 9, 16) and ankles (indices 20, 21) are smaller, more occluded, and require precise spatial localization. They tend to have 2–3× higher absolute error throughout training.

When all joints are equally weighted, the loss gradient is dominated by the easy joints (which contribute a large proportion of the total per-sample loss early in training because there are more of them and they start with high initial error). As easy joints converge, the model's gradient from joint loss weakens — but hard joints still have substantial remaining error. The training signal is misallocated.

### Evidence from the Results Table

Looking at the training history across all 23 prior ideas, `mpjpe_body_val` plateaus in the 183–196 mm range at stage-1 for most ideas, with the best designs reaching ~183 mm. Stage-2 best is 156.6 mm (idea002/design003). The plateau pattern suggests that once easy joints are well-fit, the uniform loss does not provide sufficient gradient pressure on the remaining hard joints to push `mpjpe_body_val` lower.

No prior idea has addressed **within-joint task allocation** of the gradient budget:

| Idea | Mechanism | Difference |
|---|---|---|
| idea005 | Uncertainty weighting across 3 tasks (joints, depth, UV) | Per-task level, not per-joint; all 22 joints treated equally |
| idea006 | Skeleton self-attention bias | Attention structure, not loss allocation |
| idea012 | Pairwise distance-matrix structural prior | Adds a new loss term for bone lengths; does not reweight existing joint loss |
| All others | Architecture / loss coupling | No per-joint difficulty modelling |

**idea005** is the closest prior: it applies learnable scalar weights to the 3 loss terms (joints, depth, UV). This idea applies per-joint scalar weights **within** the joint loss term — 22 weights vs. 1 weight. The mechanisms are orthogonal; this idea can compose with idea005 cleanly.

### Why Online Difficulty Estimation

Rather than hand-tuning 22 scalar weights (which would be a hyperparameter sweep), we estimate joint difficulty online from training statistics. Concretely, we maintain an exponential moving average (EMA) of the per-joint MPJPE during training:

```
ema_err[j] ← β * ema_err[j] + (1 - β) * mean_batch_err[j]     for j ∈ {0, ..., 21}
```

with momentum `β` (e.g., 0.99) so that `ema_err[j]` represents the recent per-joint error level. We then derive adaptive weights:

```
raw_w[j] = (ema_err[j] / mean_j(ema_err[j])) ^ alpha
w[j] = 22 * raw_w[j] / sum(raw_w)   # normalised to sum = 22 (preserve total gradient scale)
```

where `alpha > 0` controls the focusing strength. With `alpha=0` all weights are 1.0 (baseline). With `alpha=1.0` weights are proportional to normalised difficulty. With `alpha=2.0` harder joints get quadratically more weight (focal-loss style).

The weight vector `w[j]` is applied to the per-joint residuals before computing the loss:

```python
# In loss():
per_joint_residual = pred_joints[:, _BODY] - gt_joints[:, _BODY]   # (B, 22, 3)
w = self._get_adaptive_weights()                                      # (22,)
weighted_residual = per_joint_residual * w.view(1, 22, 1)
loss = smooth_l1(weighted_residual, torch.zeros_like(weighted_residual))
```

Note: smooth_l1 is applied to the weighted residuals directly (target = 0) rather than weighting outputs of loss_joints_module. This is equivalent to per-joint loss weighting and avoids needing to modify or subclass `SoftWeightSmoothL1Loss`.

### Properties of the Approach

1. **Self-adaptive**: weights automatically shift as training progresses. When a joint improves, its weight decreases; when another joint stagnates, its weight increases. No fixed schedule or hyperparameter tuning required beyond `alpha` and `beta`.

2. **Gradient budget preservation**: the normalisation `22 * w / sum(w)` ensures the total gradient magnitude is the same as the baseline at every step (sum of weights = 22 = number of joints). The model receives the same *total* gradient from the joint loss, but it is redistributed to harder joints.

3. **Zero initialization matches baseline**: with `ema_err` initialized to equal values (all 1.0), `w[j] = 1.0` for all j — exactly the baseline uniform weighting. The model starts identically to the baseline.

4. **No extra computation**: the EMA update is a simple element-wise operation on a 22-vector. The weight computation is 22 scalar operations. Negligible overhead on 2080 Ti.

5. **Implementable entirely within `pose3d_transformer_head.py`**: the EMA buffer is an `nn.Buffer` (non-learnable, updated in `loss()` with `torch.no_grad()`). No data pipeline changes, no config imports, no external dependencies.

### Grounding in Observed Results

- **idea001/design001**: best stage-2 composite (224.52); body MPJPE = 176.1 mm. A per-joint difficulty weighting could push the hard joints below this while not disturbing the already-well-fit easy joints.
- **idea002/design003**: best stage-2 body MPJPE (156.6 mm); dedicated pelvis query helped decouple pathways. Per-joint difficulty weighting is orthogonal: it helps within the body joint loss regardless of query architecture.
- **Stage-1 body MPJPE plateau at 183–196 mm**: across 23 prior ideas, the range is narrow. The plateau is consistent with easy joints being well-fit and hard joints stagnating under uniform weighting. Shifting gradient pressure to hard joints should break this plateau.
- **mpjpe_rel_val** (root-relative full body MPJPE including hands): baseline 438mm, best 333mm (idea008). The relative metric is mainly driven by distal limb joints. Per-joint difficulty weighting targeting distal joints should help `mpjpe_rel_val` indirectly.

---

## Proposed Variations

### Design A — EMA difficulty weighting, alpha=0.5 (mild focusing)

Introduce a `per_joint_difficulty_weighting: bool = True` flag and an `ema_momentum: float = 0.99` parameter. Register `self.joint_err_ema = nn.Buffer(torch.ones(22))` (non-learnable, updated during `loss()`). Compute per-joint weights with `alpha=0.5`.

Alpha=0.5 takes the square root of the normalised difficulty — a mild, conservative schedule that redistributes gradient without drastically penalising easy joints. This is the safe diagnostic: does any per-joint difficulty signal help?

Config kwargs: `per_joint_difficulty_weighting=True`, `ema_alpha=0.5`, `ema_momentum=0.99`.

### Design B — EMA difficulty weighting, alpha=1.0 with softmax temperature normalisation

Use `alpha=1.0` for full-proportional weighting. Additionally, normalise the raw difficulty estimates via a temperature-scaled softmax rather than a simple linear normalisation:

```python
# Softmax normalisation
T = 1.0  # temperature
soft_w = 22 * F.softmax(ema_err / T, dim=0)
```

The softmax normalisation is slightly different from simple proportional scaling: it is always positive and produces a smoother weight distribution. The temperature `T` controls concentration: `T → ∞` gives uniform weights (baseline), `T → 0` concentrates all weight on the hardest joint. A fixed `T=1.0` at the EMA scale of per-joint MPJPE (typical range 50–400 mm) produces well-calibrated weights.

Config kwargs: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='softmax'`, `weight_temperature=1.0`.

### Design C — EMA difficulty weighting, alpha=1.0 + separate upper/lower body EMA + staged warmup

Use two separate EMA buffers: `self.upper_err_ema` (joints 0–12: spine, shoulders, elbows, wrists) and `self.lower_err_ema` (joints 13–21: hips, knees, ankles). Compute group-normalised weights: within each group, weights sum to the group's joint count (proportional within group, equal across groups). This prevents the lower body (more joints, typically harder) from completely dominating the upper body gradient budget.

Additionally, apply a **warmup ramp**: for the first 5 epochs (based on iteration counter), linearly interpolate between uniform weights (baseline) and the EMA-computed weights. This prevents noisy early-training EMA estimates from destabilising the first few epochs before the EMA has converged to a meaningful estimate.

Config kwargs: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `group_normalise=True`, `ema_warmup_epochs=5`.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

**`__init__` additions:**

```python
# New constructor kwargs (all with defaults matching baseline behaviour):
per_joint_difficulty_weighting: bool = False   # False = baseline (uniform weights)
ema_alpha: float = 0.5                          # focusing exponent
ema_momentum: float = 0.99                      # EMA decay for joint error tracking
weight_norm: str = 'linear'                     # 'linear' or 'softmax'
weight_temperature: float = 1.0                 # softmax temperature (Design B)
group_normalise: bool = False                   # separate upper/lower body groups (Design C)
ema_warmup_epochs: int = 0                      # ramp warmup epochs (Design C)

# When per_joint_difficulty_weighting=True:
self.register_buffer('joint_err_ema', torch.ones(22))
# Design C: additional per-group buffers
# self.register_buffer('upper_err_ema', torch.ones(13))
# self.register_buffer('lower_err_ema', torch.ones(9))

# Track iteration count for warmup ramp:
self.register_buffer('_train_iter', torch.zeros(1, dtype=torch.long))
```

**`_get_adaptive_weights` method:**

```python
def _get_adaptive_weights(self, cur_iter: int = None) -> torch.Tensor:
    """Compute per-joint adaptive loss weights from EMA error.

    Returns:
        (22,) float tensor of per-joint weights, normalised to sum=22.
    """
    if not self.per_joint_difficulty_weighting:
        return torch.ones(22, device=self.joint_err_ema.device)

    ema = self.joint_err_ema.detach()  # (22,) — no gradient

    if self.weight_norm == 'softmax':
        # Softmax-normalised weights (Design B)
        raw = ema / self.weight_temperature
        w = 22.0 * torch.softmax(raw, dim=0)
    else:
        # Linear proportional (Design A/C)
        normalised = ema / (ema.mean() + 1e-6)
        w = normalised ** self.ema_alpha
        w = 22.0 * w / (w.sum() + 1e-6)

    # Warmup ramp: blend from uniform to difficulty-weighted (Design C)
    if self.ema_warmup_epochs > 0 and cur_iter is not None:
        # Approximate iterations per epoch based on typical train100 size
        # ~350 seqs × ~30 frames / batch_size=4 / accum=8 ≈ 328 effective steps/epoch
        iters_per_epoch = 328
        ramp_iters = self.ema_warmup_epochs * iters_per_epoch
        ramp = min(1.0, float(cur_iter) / max(ramp_iters, 1))
        uniform = torch.ones(22, device=w.device)
        w = (1.0 - ramp) * uniform + ramp * w

    return w
```

**`loss()` modifications:**

After computing `pred = self.forward(feats)` and assembling GT tensors, replace the joint loss with:

```python
_BODY = list(range(0, 22))

# --- Adaptive per-joint difficulty weighting ---
if self.per_joint_difficulty_weighting:
    # Compute per-joint error for this batch (detached — used for EMA update only)
    with torch.no_grad():
        per_joint_err = (
            pred['joints'][:, _BODY] - gt_joints[:, _BODY]
        ).norm(dim=-1).mean(dim=0) * 1000.0   # (22,) in mm

        # Update EMA
        self.joint_err_ema = (
            self.ema_momentum * self.joint_err_ema +
            (1.0 - self.ema_momentum) * per_joint_err
        )
        self._train_iter += 1

    # Compute adaptive weights
    w = self._get_adaptive_weights(int(self._train_iter.item()))  # (22,)

    # Apply per-joint weights to residuals (weight broadcast over B and 3 dims)
    pred_j = pred['joints'][:, _BODY]   # (B, 22, 3)
    gt_j   = gt_joints[:, _BODY]        # (B, 22, 3)
    # Compute weighted smooth-L1 manually (equivalent to reweighted per-joint loss)
    residuals = pred_j - gt_j                    # (B, 22, 3)
    weighted  = residuals * w.view(1, 22, 1)     # (B, 22, 3) — scaled by per-joint weight
    losses['loss/joints/train'] = self.loss_joints_module(
        weighted + gt_j, gt_j)                   # pass weighted pred against GT
else:
    losses['loss/joints/train'] = self.loss_joints_module(
        pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

Note: the `loss_joints_module` (SoftWeightSmoothL1Loss) is called with `(weighted_pred, gt)` where `weighted_pred = gt + w * (pred - gt)`. This is equivalent to multiplying the residual by `w` before the smooth-L1 penalty. At `w=1` for all joints this is exactly the baseline.

**Alternative (simpler) weighting implementation** — apply the weight directly via `weight` parameter of SoftWeightSmoothL1Loss if it supports per-sample per-joint weights. If not, compute the smooth-L1 loss manually:

```python
diff = (pred_j - gt_j).abs()  # (B, 22, 3)
beta = 0.05
smooth_l1 = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
weighted_loss = (smooth_l1 * w.view(1, 22, 1)).mean()
losses['loss/joints/train'] = weighted_loss
```

The Designer should verify which implementation is cleanest given the `SoftWeightSmoothL1Loss` API.

### `config.py`

**Design A:**
```python
per_joint_difficulty_weighting=True,
ema_alpha=0.5,
ema_momentum=0.99,
weight_norm='linear',
```

**Design B:**
```python
per_joint_difficulty_weighting=True,
ema_alpha=1.0,
ema_momentum=0.99,
weight_norm='softmax',
weight_temperature=1.0,
```

**Design C:**
```python
per_joint_difficulty_weighting=True,
ema_alpha=1.0,
ema_momentum=0.99,
weight_norm='linear',
group_normalise=True,
ema_warmup_epochs=5,
```

All values are bool/int/float/str literals. No Python import statements required. Fully compliant with MMEngine no-Python-imports restriction.

---

## Expected Outcome

- **Primary gain — mpjpe_body_val**: by shifting gradient toward consistently-hard distal joints, the model should reduce error on wrists and ankles (the main contributors to body MPJPE plateau). Target: `mpjpe_body_val < 183` at stage-1 (matching best prior), `< 155` at stage-2 (below best prior of 156.6 mm from idea002/design003).

- **Secondary gain — composite_val**: since composite = 0.67 * body + 0.33 * pelvis and body MPJPE improves, composite improves proportionally. Pelvis MPJPE should be unaffected (pelvis loss is unchanged). Target: `composite_val < 328` at stage-1 (best prior: 328.14).

- **mpjpe_rel_val**: distal limb improvements (wrist, ankle) directly reduce root-relative MPJPE for the full body. Target: `mpjpe_rel_val < 420` at stage-1.

- **Design A** (alpha=0.5, linear): mild difficulty focusing — safe baseline test of the mechanism. Expected composite_val < 340 at stage-1.

- **Design B** (alpha=1.0, softmax): full proportional focusing with stable normalisation. Primary bet. Expected composite_val < 330 at stage-1.

- **Design C** (group normalised + warmup): most controlled variant — prevents any single group (upper or lower) from being starved. The warmup avoids instability from noisy early-training EMA. Expected composite_val < 328 at stage-1, with a cleaner loss curve.

- **Composite target (stage-2)**: aim for `composite_val < 222` (best prior: 224.52 — idea001/design001), with primary improvement in body MPJPE.

---

## Risk and Mitigation

- **EMA cold-start instability**: at step 0, `joint_err_ema` is initialised to 1.0 for all joints (uniform weights = baseline). For the first few hundred steps, the EMA is dominated by the initial constant and weights remain near uniform. This is safe: the model effectively starts as the baseline and transitions gradually to difficulty-weighted training. For Design C, the `ema_warmup_epochs=5` ramp provides an additional soft transition.

- **EMA divergence from preemption/resume**: when a SLURM job is preempted and resumed from a checkpoint, the `joint_err_ema` buffer is saved and restored with the checkpoint (because it is registered as an `nn.Buffer`). The EMA continues from where it left off — no cold-start on resume. This is correct behaviour.

- **Noisy per-batch EMA updates**: with batch size 4 and EMA momentum 0.99, each update moves `joint_err_ema` by at most 1% per step toward the batch estimate. Over a typical epoch (~350 steps for train100), the EMA has an effective window of ~100 recent steps. This is stable and noise-resistant.

- **Loss scale preservation**: the normalisation `22 * w / sum(w)` ensures the mean weight is always 1.0, so the total joint loss magnitude is preserved. The AdamW learning rate schedule is calibrated for the baseline loss scale — preserving scale prevents unexpected LR sensitivity from this change.

- **Hard joint starvation (too much weight on easy joints)**: with `alpha < 0` one could down-weight hard joints (wrong direction). The alpha values proposed (0.5 and 1.0) are strictly positive — harder joints always get higher weight. The normalisation additionally ensures no joint's weight exceeds a reasonable range (typically 0.5–3.0× relative to uniform).

- **`SoftWeightSmoothL1Loss` API compatibility**: the baseline uses `SoftWeightSmoothL1Loss` with its built-in `soft_weight` (a sample-level mask, not a per-joint weight). The proposed implementation bypasses this by computing the weighted smooth-L1 directly. The Designer should verify the loss function signature and choose the cleanest implementation path (either use the `weight` parameter if it supports per-joint tensors, or compute smooth-L1 manually as shown above).

- **Interaction with idea005 (uncertainty weighting)**: idea005 applies scalar uncertainty weights per task (joints as a whole, depth, UV). This idea applies per-joint weights within the joint task. They are additive and orthogonal. If combined: the per-joint difficulty weights would be multiplied by the uncertainty weight of the joint task as a whole — the product is still a valid per-joint weight. The Designer should compose them cleanly.

- **Interaction with idea001 (multi-layer decoder)**: adding per-joint difficulty weighting to the multi-layer decoder + intermediate supervision (idea001) is straightforward. The difficulty weights apply to the final layer's joint loss; intermediate layer aux losses use the same weights or a fixed uniform weight. Left for a future combined design.

- **MMEngine config constraint**: `per_joint_difficulty_weighting` is bool, `ema_alpha` and `ema_momentum` are float, `weight_norm` is str, `weight_temperature` is float, `group_normalise` is bool, `ema_warmup_epochs` is int. All are literals. No Python import statements in config.py. Fully compliant.

- **Evaluation compatibility**: `BedlamMPJPEMetric`, `TrainMPJPEAveragingHook`, and `MetricsCSVHook` are invariant. The adaptive weighting affects only the training loss gradient. Validation output shapes are unchanged.
