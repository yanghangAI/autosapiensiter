# Code Review — idea008/design001

**Verdict: APPROVED**

---

## Review Check

`python scripts/cli.py review-check-implementation runs/idea008/design001` — PASSED.

---

## Files Changed vs. Design Specification

Design specifies changes to `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` should be unchanged.

- `code/pose3d_transformer_head.py` — changed. MATCHES design.
- `code/config.py` — changed. MATCHES design.
- `code/pelvis_utils.py` — diff against baseline: IDENTICAL. Correct.
- `code/train.py` — diff against baseline: IDENTICAL. Correct; train.py is invariant.

---

## Design Fidelity Checks

### pose3d_transformer_head.py

1. **`num_body_queries: int = 22` kwarg added after `dropout`** — PRESENT (line 151). Stored as `self.num_body_queries` (line 172). CORRECT.
2. **`self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)`** — PRESENT (line 184). Changed from `num_joints` to `num_body_queries`. CORRECT.
3. **`self.num_joints = 70` preserved** — PRESENT (line 171). Correct for `predict()` compatibility.
4. **`forward()`: decode over 22 body queries** — queries broadcast from `self.joint_queries.weight` (22 entries); `decoded` is `(B, 22, hidden_dim)`. CORRECT.
5. **`body_joints = self.joints_out(decoded)` → `(B, 22, 3)`** — PRESENT (line 253). CORRECT.
6. **Zero-pad to `(B, 70, 3)`** — `torch.zeros(B, self.num_joints - self.num_body_queries, 3, ...)` concatenated with `body_joints` (lines 256–258). No-gradient by default. CORRECT.
7. **`pelvis_token = decoded[:, 0, :]`** — PRESENT (line 260). Unchanged. CORRECT.
8. **Return dict keys unchanged: `joints`, `pelvis_depth`, `pelvis_uv`; `joints.shape == (B, 70, 3)`** — CORRECT.
9. **`loss()`: body joint loss restricted to `_BODY = list(range(0, 22))`** — PRESENT (lines 304–307). Unchanged from baseline. CORRECT.
10. **No hand auxiliary loss in design001** — correctly absent. CORRECT.
11. **`_DecoderLayer` unchanged** — structure identical to baseline. CORRECT.
12. **`_init_head_weights`**: initialises `self.joint_queries.weight` and output projections. No `hand_proj` to initialise. CORRECT.
13. **Constructor signature** matches design spec exactly. CORRECT.

### config.py

1. **`num_body_queries=22` added to `model.head` dict** — PRESENT (line 136). Integer literal. MMEngine-compliant. CORRECT.
2. **All other config values identical to baseline** — seed 2026, batch 4, accum 8, persistent_workers=False, optimizer, LR schedule, data pipeline, hooks all match baseline. CORRECT.
3. **`output_dir` set to `…/idea008/design001`** — CORRECT.

---

## Invariant Checks

- `persistent_workers=False` in both dataloaders — PRESERVED.
- Backbone, data preprocessor, metric, transforms — NOT modified (not in code/).
- `train.py` wrapper — IDENTICAL to baseline.
- `pelvis_utils.py` — IDENTICAL to baseline.
- Seed `2026`, batch `4`, accumulation `8` — PRESERVED.
- MMEngine config: no Python `import` statements; `num_body_queries=22` is a literal. COMPLIANT.

---

## Test Output

- Test ran to completion: 1 epoch, no Python errors or tracebacks.
- `metrics.csv`: `epoch=1, composite_val=462.00, mpjpe_body_val=447.86, mpjpe_pelvis_val=490.72` — all fields present. Values are high (only 1 warmup epoch with low LR), which is expected.
- `iter_metrics.csv`: 81 iterations logged with `loss_joints_train`, `loss_depth_train`, `loss_uv_train`. No NaN/Inf. Loss values decreasing across epoch. CORRECT.
- SLURM log confirms training proceeded normally with no runtime errors.

---

## Minor Observations (non-blocking)

- The module docstring still describes "self-attention over 70 joint queries" — a stale comment. This does not affect correctness.

---

## Summary

All required design changes are present and correctly implemented. All invariants are preserved. Test run completed without errors and produced valid outputs. APPROVED.
