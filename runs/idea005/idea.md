**Idea Name:** Uncertainty-Weighted Multi-Task Loss Balancing

**Approach:** Replace the fixed loss weights for the joint, pelvis depth, and pelvis UV tasks with learnable log-variance parameters (homoscedastic uncertainty weighting), so that the model automatically adapts the relative contribution of each task's gradient during training rather than relying on hand-tuned fixed weights.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline head applies three losses with equal fixed weights:

- `loss/joints/train` — SoftWeightSmoothL1Loss on root-relative joint XYZ (metres, ~0–1 m scale)
- `loss/depth/train` — SoftWeightSmoothL1Loss on pelvis depth (metres, ~2–8 m scale)
- `loss/uv/train` — SoftWeightSmoothL1Loss on pelvis UV (normalised [-1, 1])

These tasks operate at very different output scales and difficulty levels. Fixed equal weighting means the optimiser applies the same magnitude of gradient to all three heads regardless of their individual loss landscapes. In practice:

1. **Pelvis depth** has a much larger absolute scale (metres of absolute camera distance) than root-relative joints, so its raw loss magnitude is larger and may dominate early training.
2. **Joint regression** involves 70 joints (only 22 body joints contribute to the body loss) whereas depth and UV are single scalars — the gradient from joint loss is distributed over many output dimensions.
3. The composite metric weights body MPJPE at 0.67 and pelvis MPJPE at 0.33. If the model over-invests gradient budget in pelvis depth at the expense of joint regression, the composite suffers more than if the reverse is true. This is consistent with the observed baseline pattern where pelvis_val ≈ body_val despite the former being a geometrically simpler task (1 scalar).

### Evidence from prior ideas

- **idea001** showed that architectural changes (multi-layer decoder) that benefited body MPJPE hurt pelvis MPJPE — suggesting the two tasks compete for the model's capacity and gradient budget.
- **idea002** addressed this by decoupling the representational pathway (separate pelvis query). idea005 addresses it from the loss/optimisation side: even with a shared query, if the depth loss consistently overwhelms the joint loss gradient, decoupling alone may not be sufficient.
- **idea003** and **idea004** target query initialisation and positional encoding respectively. None of these address the relative contribution of each loss to the gradient.

### Why uncertainty weighting

Kendall & Gal (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" showed that for regression tasks, the optimal loss weight for task $i$ under Gaussian likelihood is:

```
L_i_weighted = (1 / (2 * σ_i^2)) * L_i  +  log(σ_i)
```

where $σ_i$ is the task's output noise (homoscedastic uncertainty). In practice, we learn `log(σ_i^2)` as a free parameter. This is:
- **Self-balancing**: tasks with high uncertainty (large loss, hard to fit) automatically get down-weighted; tasks with low uncertainty get up-weighted.
- **Parameter-efficient**: only 3 scalar parameters added (one per task).
- **Well-behaved at initialisation**: initialising all `log_var` to 0 recovers exactly the baseline equal-weight loss, so training starts from the baseline configuration.
- **Confined to `pose3d_transformer_head.py`**: the log-variance parameters are `nn.Parameter` on the head module; loss computation changes are purely arithmetic — no architectural changes.

This is orthogonal to all prior ideas. It can be layered on top of any architecture change (multi-layer decoder, decoupled pelvis query, adaptive queries, depth positional encoding) without conflict.

---

## Proposed Variations

**Design A — Uncertainty weighting on all three tasks (full)**

Add three learnable parameters `log_var_joints`, `log_var_depth`, `log_var_uv` (each scalar `nn.Parameter`, initialised to 0):

```
loss/joints = (exp(-log_var_joints) * raw_loss_joints) + log_var_joints
loss/depth  = (exp(-log_var_depth)  * raw_loss_depth)  + log_var_depth
loss/uv     = (exp(-log_var_uv)     * raw_loss_uv)     + log_var_uv
```

This is the reference implementation of Kendall & Gal's formulation applied to all three tasks. The `log_var` parameters are included in the head's parameters and updated by AdamW alongside all other parameters. The fixed `loss_weight_depth` and `loss_weight_uv` scalar multipliers from the baseline are replaced by the learned uncertainty terms (or removed from the config and set to 1.0 since the uncertainty term subsumes them).

This is the most principled design: tests whether learned task balancing alone, without any architectural change, reduces the composite metric.

**Design B — Uncertainty weighting on depth and UV only (joint loss anchored)**

Only `log_var_depth` and `log_var_uv` are learnable; the joint loss is kept with a fixed weight of 1.0. Rationale: the joint loss is the primary task and its weight is the natural anchor. Allowing all three weights to be learned simultaneously may cause the joint loss to be spuriously down-weighted if it is harder than depth/UV early in training. Anchoring the joint loss prevents this failure mode.

```
loss/joints = raw_loss_joints                                      (fixed weight 1.0)
loss/depth  = (exp(-log_var_depth) * raw_loss_depth)  + log_var_depth
loss/uv     = (exp(-log_var_uv)    * raw_loss_uv)     + log_var_uv
```

This is a more conservative variant: the model is free to balance depth vs. UV against each other, and both against the fixed joint anchor. This targets the pelvis subtask directly (the known weak point in the composite metric) while protecting body MPJPE from degradation.

**Design C — Uncertainty weighting + composite-aware joint-task weight**

Build on Design B but also apply a fixed joint loss multiplier scaled to reflect the composite metric's weighting (0.67 joint / 0.33 pelvis). Concretely, scale the fixed joint loss weight to 2.0 (= 0.67/0.33) so that the raw gradient balance before learning matches the composite's intended relative importance:

```
loss/joints = 2.0 * raw_loss_joints                                (fixed, composite-proportional)
loss/depth  = (exp(-log_var_depth) * raw_loss_depth)  + log_var_depth
loss/uv     = (exp(-log_var_uv)    * raw_loss_uv)     + log_var_uv
```

This tests whether biasing the starting point of the optimisation toward the metric's own weighting (before uncertainty adaptation takes over) accelerates convergence within the 20-epoch budget. If the composite metric was designed with domain knowledge about task difficulty, incorporating its weights as a prior is a reasonable inductive bias.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py`:

1. In `__init__`:
   - **Design A**: Add `self.log_var_joints = nn.Parameter(torch.zeros(1))`, `self.log_var_depth = nn.Parameter(torch.zeros(1))`, `self.log_var_uv = nn.Parameter(torch.zeros(1))`. The existing `loss_weight_depth` and `loss_weight_uv` scalars are effectively superseded (can be kept at 1.0 in config, or a new constructor kwarg `use_uncertainty_weighting: bool = True` controls the mode).
   - **Design B**: Only `self.log_var_depth` and `self.log_var_uv`; joint loss unchanged.
   - **Design C**: Same as Design B but with a `joint_loss_scale: float = 2.0` constructor kwarg used as a fixed multiplier for the joint loss.

2. In `loss()`:
   - Compute raw losses as before, then apply the uncertainty formula:
     ```python
     # Design A example:
     l_j = self.loss_joints_module(...)  # raw
     l_d = self.loss_depth_module(...)   # raw
     l_u = self.loss_uv_module(...)      # raw

     losses['loss/joints/train'] = torch.exp(-self.log_var_joints) * l_j + self.log_var_joints
     losses['loss/depth/train']  = torch.exp(-self.log_var_depth)  * l_d + self.log_var_depth
     losses['loss/uv/train']     = torch.exp(-self.log_var_uv)     * l_u + self.log_var_uv
     ```
   - The `loss_weight_depth` and `loss_weight_uv` scalars from the baseline are replaced by or subsumed into the uncertainty terms.

3. In `config.py`:
   - Add `use_uncertainty_weighting: True` (and optionally `joint_loss_scale` for Design C) as head kwargs.
   - The existing `loss_weight_depth` and `loss_weight_uv` entries can remain as 1.0 (they become no-ops when uncertainty weighting is active).

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, or training infrastructure.

---

## Expected Outcome

- **Primary gain**: better composite score through improved trade-off between body MPJPE and pelvis MPJPE. If depth loss was previously over-weighted (causing pelvis overfitting at the cost of joints), the model will learn `log_var_depth > 0` (down-weighting depth) and `log_var_joints < 0` (up-weighting joints), shifting optimisation toward the body task.
- **Design A**: tests whether all three tasks self-balance. Riskier because joint loss could be down-weighted. Provides a diagnostic of the current loss imbalance.
- **Design B**: the conservative but targeted fix — balance only the two pelvis tasks while anchoring joint regression. Expected to improve composite by improving pelvis accuracy without body regression.
- **Design C**: explicit composite-proportional bias for fastest convergence toward the metric.
- **Composite target**: aim for composite_val < 163 (vs. baseline 169.99, idea001 best 165.98 at epoch 12).

---

## Risk and Mitigation

- **Unconstrained `log_var` collapse**: if `log_var` grows too large (unbounded up-weighting of regularisation term), the loss can behave unexpectedly. Mitigation: clip `log_var` to a reasonable range (e.g., [−4, 4]) in the loss computation, or rely on weight decay from AdamW acting on the `log_var` parameters. The log-variance parameters should be excluded from the backbone `lr_mult` `paramwise_cfg` — they are head parameters and should use the full learning rate.
- **Joint loss down-weighting in Design A**: if joint regression is hard early in training, `log_var_joints` may grow and reduce joint gradient. Mitigation: Design B and C avoid this by anchoring joint loss. Monitoring `log_var` values in the log is a useful diagnostic.
- **Interaction with idea002 (Dedicated Pelvis Query)**: if combined, the dedicated pelvis query and uncertainty weighting are orthogonal. The uncertainty formulation applies to whatever losses are present, regardless of which query produces the pelvis prediction.
- **Memory/speed**: 3 scalar parameters. Negligible overhead.
- **MMEngine config constraint**: `log_var` parameters are `nn.Parameter` inside the head module, not hyperparameters in config.py. No import statements needed in config.py. The `use_uncertainty_weighting` flag and `joint_loss_scale` are simple bool/float literals. Fully compliant with the no-Python-imports MMEngine config restriction.
