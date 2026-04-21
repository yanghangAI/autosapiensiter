**APPROVED**

**Design:** idea022/design003 — 2-layer cascaded decoder with dynamic Gaussian reprojection bias, auxiliary loss (weight=0.4), and learnable per-joint σ and γ initialized to (4.0, 2.0).

**Verdict:** APPROVED

---

## Review Summary

The design is complete, explicit, and implementation-ready. It precisely specifies the single differentiating change from design002: replacing fixed scalar σ/γ with learnable per-joint `nn.Parameter` tensors. All other details are inherited correctly from design001/002.

### Feasibility and Completeness

All three allowed files are addressed:
- **`pelvis_utils.py`**: Same `project_joints_to_feat_grid` helper — Builder instructed to skip if already present.
- **`pose3d_transformer_head.py`**: All structural changes from design001/002 apply with three precise additions/changes for Design C.
- **`config.py`**: Only `reproj_bias_learnable=True` differs from design002. All literal values.

### Key Change: Learnable Per-Joint Parameters

The design specifies exactly:

1. **Parameter creation in `__init__`** (guarded by `if reproj_bias_learnable:`):
   ```python
   self.bias_sigma = nn.Parameter(torch.ones(num_joints) * reproj_bias_sigma)
   self.bias_gamma = nn.Parameter(torch.ones(num_joints) * reproj_bias_gamma)
   ```
   Shape `(J,)`, initialized to `(4.0, 2.0)` — matching Design B's fixed values. Correct.

2. **Conditional sigma/gamma computation in `loss()`**:
   - When `reproj_bias_learnable=True`: `sigma = F.softplus(self.bias_sigma)` (ensures positivity), `gamma = self.bias_gamma` (unconstrained).
   - When `reproj_bias_learnable=False`: falls back to `torch.full(...)` with scalar values.
   - Both branches cast to `feat_coords.device` and `feat_coords.dtype`.
   
   The `_build_gaussian_bias` function is unchanged — it accepts `(J,)` tensors regardless of origin. Correct.

3. **Gradient flow through learnable parameters**: `F.softplus(self.bias_sigma)` is differentiable; gradients flow from the bias into `self.bias_sigma`. The bias enters the main loss path via `self._reproj_bias` in `forward()`, and also via the auxiliary loss path (auxiliary loss does not directly use sigma/gamma, but the bias quality affects the main loss gradient). Correct.

### Architecture Correctness

All points from design001 and design002 apply.

Additional correctness checks for Design C:

1. **`F.softplus` positivity**: guarantees sigma > 0. `_build_gaussian_bias` also clamps to >= 0.5. Both safeguards preserved. Correct.

2. **`bias_gamma` unconstrained**: The design explicitly does not apply positivity constraint on gamma. Negative gamma would invert the bias (attention suppressor). The design accepts this as a deliberate choice and prohibits the Builder from adding a constraint proactively. Correct.

3. **Parameters excluded from backbone LR multiplier**: `self.bias_sigma` and `self.bias_gamma` are head parameters; the backbone has `lr_mult=0.1`. These parameters fall under the default LR, which is correct and consistent with other head parameters.

4. **AMP compatibility**: Float32 `nn.Parameter` cast to `feat_coords.dtype` before passing to `_build_gaussian_bias`. Correct.

5. **Shared code path**: The design explicitly states the same `pose3d_transformer_head.py` handles all three designs via the `reproj_bias_learnable` flag. The Builder should implement this as a single unified file.

### Invariant Preservation

Same as design001/002 — no invariant files modified. Loss restricted to body joints. Config uses only literals.
