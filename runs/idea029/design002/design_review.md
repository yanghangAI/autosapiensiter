# Design Review — idea029/design002

**Verdict: APPROVED**

---

## Checklist

### Feasibility and Completeness

- **Starting point specified:** `baseline/` — correct.
- **Files changed:** `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — all within the allowed set.
- **Invariant files untouched:** none modified.

### `pelvis_utils.py`

- Identical to Design 001: `recover_abs_joints_batched` added at end of file with exact code. No issues.

### `pose3d_transformer_head.py`

- Identical to Design 001 — all four new kwargs, attribute storage, and `loss()` block are the same code. Stated explicitly.
- **Design 002 path:** `abs_joint_pelvis_grad_scale=1.0` (default, not < 1.0) → `else` branch executes. `abs_joint_axis_weights=[2.0, 1.0, 1.0]` → `register_buffer('abs_axis_weights', ...)` stores a `(3,)` float32 buffer. The per-axis weighting line `abs_loss_raw = abs_loss_raw * self.abs_axis_weights.view(1, 1, 3)` executes correctly.
- Effect of per-axis weighting on `.mean()` is explained clearly (X-column scaled ×2 before global average).
- `register_buffer` handles device placement — no manual `.to(device)` needed.

### `config.py`

- Three kwargs added: `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`, `abs_joint_axis_weights=[2.0, 1.0, 1.0]`. List-of-float literal — valid MMEngine config value. No Python imports. Constraint satisfied.
- Full head dict shown with all existing kwargs preserved.
- `abs_joint_pelvis_grad_scale` omitted from config → defaults to `1.0` — correct for this design.

### Invariants

Same as Design 001 — all preserved.

### Edge Cases

- Same as Design 001. Additionally: `abs_axis_weights` buffer will be on the correct device via `register_buffer` — confirmed.

### No Issues Found

The Builder can implement this without any guessing. Differences from Design 001 are limited to one additional config kwarg and the `abs_axis_weights` buffer path, both fully specified.
