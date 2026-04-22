**Idea Name:** Per-Joint Laplace NLL Uncertainty Regression for Body Joint Loss

**Approach:** Replace the fixed-scale SoftWeightSmoothL1 body-joint loss with a Laplace negative log-likelihood loss where the model simultaneously predicts each body joint's 3D coordinate AND a per-joint coordinate log-scale parameter (log-variance proxy), so that the network can adaptively down-weight genuinely hard or ambiguous joints while concentrating learning signal on confident predictions — without requiring any external visibility annotations.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Fixed-Loss-Scale Bottleneck

The baseline joint loss is `SoftWeightSmoothL1Loss(pred_joints[:, 0:22], gt_joints[:, 0:22])` — a single fixed-scale loss applied uniformly to all 22 body joints in all samples. This treats every joint equally regardless of its geometric difficulty: the pelvis (low depth ambiguity, large spatial footprint) receives the same gradient weight as wrists and ankles (high depth ambiguity, small spatial footprint, frequent occlusion in BEDLAM2).

Two concrete problems arise:

**1. No adaptive gradient routing per joint.** In BEDLAM2, synthetic data covers a wide range of body poses and viewing angles. Joints that are systematically harder to predict (end-effectors, distal limbs) saturate the loss quickly during early training, producing noisy large-gradient updates that interfere with learning on easier joints. The model has no mechanism to express confidence and route gradients accordingly.

**2. Output head is a single scalar mean — no uncertainty.** The output head `joints_out: Linear(hidden_dim, 3)` produces a point estimate. There is no way for the model to express that a particular joint is highly uncertain, and there is no auxiliary training signal to encourage the model to be appropriately calibrated.

### The Laplace NLL Solution

The **Laplace distribution** provides the most direct probabilistic generalization of L1 regression. The NLL of a prediction `μ` with scale parameter `s` under a Laplace distribution centred at ground-truth `y` is:

```
NLL(μ, s; y) = log(2s) + |μ - y| / s
```

Minimising this over `μ` (fixed `s`) recovers standard L1 regression. Minimising jointly over `(μ, s)` gives the **optimal** `s = E[|μ - y|]` — the model learns to set `s` proportional to its own expected error at that joint. Joints where the model is confident (small expected error) automatically get a large gradient signal from `|μ - y|/s`; joints where the model is uncertain (large expected error) have their gradient naturally normalised.

This is equivalent to learned per-joint loss weighting, but with a principled probabilistic interpretation and minimal hyperparameter sensitivity:
- At init (`log_s = 0 → s = 1`): exactly recovers the L1 baseline loss on `μ` (modulo the log(2s) entropy term which is a constant at init).
- After a few epochs: the model sets `s_j` to its empirical prediction error for joint `j`, and subsequent training focuses gradient on *correctable* errors.

### How This Differs from All Prior Ideas

| Idea | What it changes | Distinction |
|---|---|---|
| idea005 (Uncertainty-Weighted Multi-Task Loss) | 3 learnable scalars: one per **task** (joints/depth/uv) | Global per-task weighting; the 22 body joints share one variance scalar. The model cannot express that *wrist* is harder than *hip*. |
| idea012 (Pairwise Distance Loss) | Adds a loss term over inter-joint distances | Structural regularisation; does not address per-joint gradient routing |
| idea025 (Bilateral Symmetry Loss) | Pairs left-right joints | Symmetry consistency; does not address per-joint uncertainty |
| All other ideas | Decoder architecture, query init, attention mechanism | Do not touch the body-joint loss function |

**This idea is the first to introduce per-joint, per-coordinate output uncertainty — each body joint gets its own predicted scale vector `(s_x, s_y, s_z)` (or a shared scalar `s_j`)**. The three variants below explore the parameter sharing strategy. This is orthogonal to every prior idea and can compose with any of them.

### Connection to Laplace NLL in Practice

Laplace NLL as a regression loss is standard in probabilistic deep learning:
- **Uncertainty estimation** (Kendall & Gal, NeurIPS 2017): demonstrated that predicting aleatoric uncertainty via NLL loss improves both calibration and accuracy for depth estimation and semantic segmentation.
- **Monocular depth** (AdaBins, BinsFormer, etc.): log-likelihood regression with learned uncertainty is consistently superior to fixed-scale L1/L2 regression on complex datasets.
- **Human pose estimation** (Kong et al., 2019): per-joint uncertainty improved MPJPE on Human3.6M by ~8 mm compared to a deterministic baseline.

The Laplace variant is preferred over Gaussian NLL because:
1. `|μ - y|` (L1 surrogate) is more robust to outlier GT annotations than `(μ - y)^2` (L2).
2. BEDLAM2 is synthetic but the baseline already uses SoftWeightSmoothL1 (L1-like), confirming L1 is the appropriate regime.

### Grounding in Observed Results

- **Body MPJPE plateau**: stage-1 body MPJPE has a floor around 183–195 mm across 25 ideas. The fixed-scale joint loss is a common thread across all ideas that have not broken this floor. Adaptive per-joint loss weighting is the most direct intervention.
- **Wrist/ankle difficulty**: in BEDLAM2, end-effectors exhibit the highest variability in 3D position (rotation of wrist adds many mm of error at the joint). The fixed loss treats these identically to stable joints like the spine. A learned scale for wrists/ankles will naturally grow, dampening noisy gradients and allowing the model to focus on correctable errors.
- **idea005 result**: idea005/design003 (task-level uncertainty, best variant) achieved composite_val 330.78 at stage-1 — slightly better than baseline 346.58. The task-level signal is too coarse to improve body MPJPE significantly. Per-joint uncertainty is the natural next granularity.
- **idea013 (kinematic bone vectors) result**: design003 achieved 328.14 stage-1 — best overall at stage-1. Bone vectors enforce structural priors at the output level. Per-joint uncertainty at the loss level is complementary and orthogonal: the model can simultaneously enforce kinematic structure AND learn which joints to trust.

---

## Proposed Variations

### Design A — Shared per-joint scalar scale (22 scalars, log-init)

Add a second output head `log_scale_out: Linear(hidden_dim, 22)` initialized to **all-zeros** (so `s = exp(0) = 1` at training start — exactly recovering the L1 baseline). The Laplace NLL loss for the body joints is:

```python
# pred_joints: (B, 22, 3)
# pred_log_scale: (B, 22) — one shared scale per joint across x, y, z
# gt_joints: (B, 22, 3)

s = torch.exp(pred_log_scale).unsqueeze(-1)          # (B, 22, 1)
abs_err = (pred_joints - gt_joints).abs()            # (B, 22, 3)
laplace_nll = torch.log(2 * s) + abs_err / s         # (B, 22, 3)
loss_joints = laplace_nll.mean()
```

The shared scalar per joint (`log_scale_out` outputs `(B, 22)` → expanded to `(B, 22, 1)`) is the simplest variant. The log-entropy term `log(2s)` prevents the model from trivially setting `s → ∞` to zero the L1 term. The joint prediction `pred_joints` comes from the same unchanged `joints_out: Linear(hidden_dim, 3)`.

Config kwargs: `use_per_joint_uncertainty=True`, `per_joint_uncertainty_mode='shared_scalar'`, `laplace_entropy_weight=1.0`.

### Design B — Per-joint, per-axis scale (22×3 = 66 scale parameters)

Same as Design A, but `log_scale_out: Linear(hidden_dim, 66)` outputs a separate scale for each of the 3 axes (x, y, z) at each joint — `pred_log_scale: (B, 22, 3)`. The Laplace NLL is:

```python
s = torch.exp(pred_log_scale)                         # (B, 22, 3)
abs_err = (pred_joints - gt_joints).abs()             # (B, 22, 3)
laplace_nll = torch.log(2 * s) + abs_err / s          # (B, 22, 3)
loss_joints = laplace_nll.mean()
```

This allows the model to express that, e.g., wrist X (depth component, harder due to depth ambiguity) has higher uncertainty than wrist Y/Z. In BEDLAM2's coordinate system (X=forward/depth), depth-axis uncertainty should naturally be learned as higher for end-effectors.

Init: `log_scale_out.weight` and `log_scale_out.bias` set to zero → s=1 → exact L1 baseline start.

Config kwargs: `use_per_joint_uncertainty=True`, `per_joint_uncertainty_mode='per_axis'`, `laplace_entropy_weight=1.0`.

### Design C — Per-joint scalar scale + entropy weight annealing

Same as Design A (shared scalar per joint), but with an **entropy weight annealing schedule**: the `log(2s)` entropy term is multiplied by a weight `w_ent` that starts at 0.1 (allowing the model to freely grow `s`) and linearly increases to 1.0 over 10 epochs (forcing the model to commit to tight uncertainty estimates).

This two-phase approach is motivated by the observation that at init the model's predictions are far from GT — letting `s` grow freely early avoids an unstable start where large `log(2s)` dominates. As the model improves, the entropy penalty increases, forcing the model to be appropriately confident.

Implementation: the entropy weight is passed as a float kwarg to the head from a custom `MMEngine` hook that updates it each epoch. Since hooks that modify model attributes are complex to implement cleanly in the allowed files, the simpler approximation is to use a linear anneal computed from the current iteration/epoch count accessible via the `train_cfg` dict (which is passed to `loss()` and may contain `epoch_count` or similar state).

The Designer should implement this as: `w_ent = min(1.0, 0.1 + (current_epoch / 10) * 0.9)`. The epoch count can be read from `train_cfg.get('current_epoch', 0)` if MMEngine passes it, or the head can maintain a `self._epoch` counter incremented each time the loss is called (counting every ~100 steps = 1 epoch approximately via iteration counting). The simplest approach for the Designer is to pass `entropy_anneal_steps=500` as a config kwarg: the head counts `self._loss_calls` and linearly ramps up the entropy weight over the first 500 gradient steps.

Config kwargs: `use_per_joint_uncertainty=True`, `per_joint_uncertainty_mode='shared_scalar'`, `laplace_entropy_weight_start=0.1`, `laplace_entropy_weight_end=1.0`, `laplace_entropy_anneal_steps=500`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or training infrastructure.

### `pose3d_transformer_head.py`

**1. `__init__` additions**

New kwargs (all have defaults matching baseline behaviour):

```python
use_per_joint_uncertainty: bool = False
per_joint_uncertainty_mode: str = 'shared_scalar'  # 'shared_scalar' or 'per_axis'
laplace_entropy_weight: float = 1.0
laplace_entropy_weight_start: float = 1.0           # Design C
laplace_entropy_weight_end: float = 1.0             # Design C
laplace_entropy_anneal_steps: int = 0               # 0 = no annealing (Designs A/B)
```

When `use_per_joint_uncertainty=True`:
```python
out_dim = 22 if per_joint_uncertainty_mode == 'shared_scalar' else 66
self.log_scale_out = nn.Linear(self.hidden_dim, out_dim)
nn.init.zeros_(self.log_scale_out.weight)
nn.init.zeros_(self.log_scale_out.bias)
# For Design C annealing
self._loss_call_count = 0
```

**2. `forward()` additions**

After decoder, extract `pelvis_token = decoded[:, 0, :]` as before. Add:

```python
if self.use_per_joint_uncertainty:
    # Only on body queries (first 22)
    body_decoded = decoded[:, :22, :]   # (B, 22, hidden_dim)
    # Flatten to (B, 22*hidden_dim) is NOT needed; apply per-token:
    # Actually: Linear(hidden_dim, out_dim) applied per token is wrong.
    # log_scale_out maps hidden_dim → 22 or 66, applied to pooled body feature.
    # Clean approach: average body query features → global body embedding → scale
    body_feat = body_decoded.mean(dim=1)   # (B, hidden_dim) — pooled body feature
    log_scale_raw = self.log_scale_out(body_feat)  # (B, 22) or (B, 66)
    if self.per_joint_uncertainty_mode == 'per_axis':
        log_scale = log_scale_raw.view(B, 22, 3)    # (B, 22, 3)
    else:
        log_scale = log_scale_raw.view(B, 22, 1)    # (B, 22, 1)
    pred['log_scale'] = log_scale
```

**Designer note on pooling strategy**: using `body_decoded.mean(dim=1)` collapses per-joint information. A better alternative, preserving per-joint information, is to apply `log_scale_out: Linear(hidden_dim, 1)` to each body query token independently (treating the 22 tokens separately), producing `(B, 22, 1)` directly. For Design B (per-axis), use `Linear(hidden_dim, 3)` applied per-token → `(B, 22, 3)`. This is cleaner and per-joint:

```python
# Apply log_scale_out to each body query token independently
body_decoded = decoded[:, :22, :]   # (B, 22, hidden_dim)
log_scale = self.log_scale_out(body_decoded)  # (B, 22, 1) or (B, 22, 3)
# (log_scale_out has out_features = 1 or 3, applied broadcast over 22 tokens)
pred['log_scale'] = log_scale
```

This is the preferred design: `log_scale_out: Linear(hidden_dim, 1)` for Design A (shared scalar per joint), `Linear(hidden_dim, 3)` for Design B (per-axis). The Designer should use `out_features = 1` for Design A and `out_features = 3` for Design B, passing `out_features` as a derived config value. For MMEngine config compliance, this is a simple integer literal.

Updated config kwarg: `log_scale_out_features: 1` (Design A/C) or `3` (Design B).

**3. `loss()` additions**

Replace the existing joint loss line:

```python
# Baseline:
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

With the Laplace NLL version when `use_per_joint_uncertainty=True`:

```python
if self.use_per_joint_uncertainty:
    log_s = pred['log_scale']               # (B, 22, 1) or (B, 22, 3)
    s = torch.exp(log_s)                    # (B, 22, 1) or (B, 22, 3)
    abs_err = (pred['joints'][:, _BODY] - gt_joints[:, _BODY]).abs()  # (B, 22, 3)
    # Entropy weight annealing (Design C)
    if self.laplace_entropy_anneal_steps > 0:
        self._loss_call_count += 1
        progress = min(1.0, self._loss_call_count / float(self.laplace_entropy_anneal_steps))
        w_ent = self.laplace_entropy_weight_start + progress * (
            self.laplace_entropy_weight_end - self.laplace_entropy_weight_start)
    else:
        w_ent = self.laplace_entropy_weight
    # Laplace NLL
    nll = w_ent * torch.log(2.0 * s) + abs_err / s   # (B, 22, 3) or (B, 22, 3)
    losses['loss/joints/train'] = nll.mean()
else:
    losses['loss/joints/train'] = self.loss_joints_module(
        pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

The entropy term `torch.log(2 * s) = log(2) + log(s)` prevents scale collapse. The scale `s` is bounded below by exp(-inf) → gradient clips at zero; in practice, adding a small epsilon `s = torch.exp(log_s).clamp(min=1e-4)` prevents exact zero from log(2*0).

**4. `predict()` — no change required.** The scale prediction is training-only; `predict()` only reads `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']` which are unchanged. The `pred['log_scale']` key in the forward output is silently ignored by `predict()` since it only reads specific keys.

**Note on `forward()` output dict**: adding `log_scale` to the forward output dict is safe for `predict()` since Python dict access is key-specific. The metric code in `bedlam_metric.py` receives `InstanceData` with `keypoints`, `keypoint_scores`, `pelvis_depth`, `pelvis_uv` — none of which involve `log_scale`. Invariant preserved.

### `config.py`

**Design A:**
```python
use_per_joint_uncertainty=True,
per_joint_uncertainty_mode='shared_scalar',
log_scale_out_features=1,
laplace_entropy_weight=1.0,
laplace_entropy_anneal_steps=0,
```

**Design B:**
```python
use_per_joint_uncertainty=True,
per_joint_uncertainty_mode='per_axis',
log_scale_out_features=3,
laplace_entropy_weight=1.0,
laplace_entropy_anneal_steps=0,
```

**Design C:**
```python
use_per_joint_uncertainty=True,
per_joint_uncertainty_mode='shared_scalar',
log_scale_out_features=1,
laplace_entropy_weight_start=0.1,
laplace_entropy_weight_end=1.0,
laplace_entropy_anneal_steps=500,
```

All values are bool/int/float/str literals. No Python import statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — mpjpe_body_val**: adaptive per-joint gradient weighting is expected to improve body MPJPE by allowing the model to focus learning on correctable errors while dampening noisy gradients from hard joints. Target: `mpjpe_body_val < 185` at stage-1 (competitive with best: 183mm from idea023/design001), `< 170` at stage-2.

- **Secondary gain — mpjpe_rel_val**: per-joint uncertainty naturally helps relative pose accuracy because the model learns to be more confident about joint-to-joint geometric relationships as scale uncertainty is absorbed by the scale parameter. Expected improvement: `mpjpe_rel_val < 420` at stage-1.

- **Design A** (shared scalar per joint, λ_ent=1.0): most conservative — equivalent to learned per-joint L1 weighting. Diagnostic: does per-joint gradient routing help over uniform loss? Expected composite_val < 340 at stage-1.

- **Design B** (per-axis scales): allows the model to express that X (depth axis) is systematically harder than Y/Z at each joint. In BEDLAM2's coordinate system (X=forward), this is a well-motivated inductive bias. Expected composite_val < 335, competitive with best prior stage-1 of 323.75.

- **Design C** (entropy annealing): by starting with low entropy penalty, the model can initially grow `s` freely to avoid destabilisation, then commit to tight estimates. Expected most stable training trajectory. Expected composite_val < 335.

- **Composite target (stage-1)**: `composite_val < 330` for the best design, matching or improving on idea023/design001's 323.75.
- **Composite target (stage-2)**: `composite_val < 225`, competitive with best prior stage-2 of 224.52 (idea001/design001).

---

## Risk and Mitigation

- **Scale collapse (s → 0)**: if the entropy weight `log(2s)` term is zero or negligible, the model can trivially minimize NLL by setting `s → 0`, making the `|μ - y|/s` term blow up. Mitigation: the entropy term is always included (`laplace_entropy_weight >= 0.1`); the Designer should also add `s = exp(log_s).clamp(min=1e-4)` to prevent exact zero.

- **Scale explosion (s → ∞)**: if the model learns to set `s` very large for all joints, the NLL reduces to a constant and no learning on `μ` occurs. The entropy term `log(2s)` penalises large `s`, preventing explosion.

- **Initialisation safety**: `log_scale_out` is zero-initialised, so `s=1` at training start. The Laplace NLL at `s=1` is `log(2) + |μ - y|` — a constant offset from pure L1. Gradient w.r.t. `μ` at init is `sign(μ-y)/s = sign(μ-y)` — identical to L1 gradient. Gradient w.r.t. `log_s` at init is `1 - |μ-y|` — positive when prediction error is less than 1 (s grows) and negative when error is larger (s shrinks). For early training when errors are large (>1m), `log_s` decreases → `s < 1` for most joints — but the clamping `clamp(min=1e-4)` prevents instability. The Designer should verify training loss curves for the first 5 epochs.

- **Interaction with AMP (float16)**: `torch.exp(log_s)` can overflow in float16 if `log_s > 88`. Mitigation: clamp `log_s = log_s.clamp(-10, 5)` before exp, giving `s ∈ [4.5e-5, 148]`. In practice, log_s should stay near 0 (baseline) to a few units. AMP uses float16 for compute but float32 for master weights; the exp should be computed in float32 via `autocast`. Designer should verify with a `.float()` cast on log_s if needed.

- **Metric invariance**: the metric `BedlamMPJPEMetric` receives only `pred['joints']` (the mean, unchanged shape `(B, 70, 3)`). The `log_scale` output does not propagate to the metric. `predict()` does not write `log_scale` to `InstanceData`. Fully invariant.

- **Memory**: `log_scale_out: Linear(256, 1)` adds 256 params; `Linear(256, 3)` adds 768 params. Negligible. The `log_scale` tensor in forward is `(B, 22, 1) or (B, 22, 3)` ≈ < 1 KB. No memory concern.

- **MMEngine config compliance**: all new kwargs are bool/int/float/str literals. No import statements in config. Fully compliant.

- **Interaction with idea005**: idea005 adds one learnable log-variance per task (3 scalars total). If both are active, there are two uncertainty mechanisms: task-level (idea005) and joint-level (this idea). They are additive and composable. The Designer should not combine them without explicit instruction, but can note composability for future ideas.

- **Composability with idea013 (bone vectors)**: the Laplace NLL loss acts on the recovered joint coordinates (after forward kinematics) — it is agnostic to the output parameterisation. Combining idea013 + idea026 means the joint loss is applied to kinematically-recovered coordinates with adaptive scale. This is a natural and powerful combination left for a future idea.
