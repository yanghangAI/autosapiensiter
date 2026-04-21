# Code Review — idea008/design002

**Verdict: APPROVED**

---

## Review Check

`python scripts/cli.py review-check-implementation runs/idea008/design002` — PASSED.

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

1. **`num_body_queries: int = 22` and `hand_aux_loss_weight: float = 0.1` kwargs** — PRESENT (lines 151, 154). Both stored as instance attributes (lines 173, 174). CORRECT.
2. **`self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)`** — PRESENT (line 186). CORRECT.
3. **`self.num_joints = 70` preserved** — PRESENT (line 172). CORRECT.
4. **`self.hand_proj = nn.Linear(num_body_queries * hidden_dim, (num_joints - num_body_queries) * 3)`** — PRESENT (lines 197–198). Evaluates to `Linear(5632, 144)`. Dimensions computed dynamically from kwargs. CORRECT.
5. **`_init_head_weights`: `hand_proj` initialised with `trunc_normal_(std=0.02)` weight and zero bias** — PRESENT (lines 218–219). CORRECT.
6. **`forward()`: `body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)`** — PRESENT (line 265). CORRECT.
7. **`hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)`** — PRESENT (lines 266–267). CORRECT.
8. **`joints = torch.cat([body_joints, hand_joints], dim=1)` → `(B, 70, 3)`** — PRESENT (line 269). CORRECT.
9. **`pelvis_token = decoded[:, 0, :]`** — PRESENT (line 271). Unchanged. CORRECT.
10. **`loss()`: auxiliary hand loss `loss/hand_aux/train = 0.1 * loss_joints_module(pred_joints[:, _HAND], gt_joints[:, _HAND])`** — PRESENT (lines 325–327). Uses `self.hand_aux_loss_weight` (0.1) and reuses `self.loss_joints_module`. `_HAND = list(range(22, 70))`. CORRECT.
11. **Body joint loss unchanged with `_BODY = list(range(0, 22))`** — PRESENT (lines 315–318). CORRECT.
12. **`_DecoderLayer` unchanged** — CORRECT.
13. **Constructor signature** matches design spec. CORRECT.

### config.py

1. **`num_body_queries=22` and `hand_aux_loss_weight=0.1` in `model.head` dict** — PRESENT (lines 136, 139). Both are literal values. MMEngine-compliant. CORRECT.
2. **All other config values identical to baseline** — seed 2026, batch 4, accum 8, persistent_workers=False, optimizer, LR schedule, data pipeline, hooks all match baseline. CORRECT.
3. **`output_dir` set to `…/idea008/design002`** — CORRECT.

---

## Invariant Checks

- `persistent_workers=False` in both dataloaders — PRESERVED.
- Backbone, data preprocessor, metric, transforms — NOT modified.
- `train.py` wrapper — IDENTICAL to baseline.
- `pelvis_utils.py` — IDENTICAL to baseline.
- Seed `2026`, batch `4`, accumulation `8` — PRESERVED.
- MMEngine config: no Python `import` statements; all new values are literals. COMPLIANT.
- Auxiliary hand loss uses existing `self.loss_joints_module` — no new loss module instantiated. CORRECT per constraint 5 in design.

---

## Test Output

- Test ran to completion: 1 epoch, no Python errors or tracebacks.
- `metrics.csv`: `epoch=1, composite_val=479.05, mpjpe_body_val=449.61, mpjpe_pelvis_val=538.82` — all columns present. Values are high (only 1 warmup epoch). Expected.
- SLURM log confirms `loss/hand_aux/train: 0.043019` is being logged during training, confirming the auxiliary loss is active. CORRECT.
- `iter_metrics.csv`: 81 iterations logged. Loss values are numerically stable (no NaN/Inf). CORRECT.
- Note: `iter_metrics.csv` columns do not include `loss_hand_aux_train`. This is because `MetricsCSVHook` tracks a fixed set of columns defined elsewhere; the hand_aux loss is visible in the SLURM log and does not affect CSV correctness.

---

## Minor Observations (non-blocking)

- Module docstring still describes "self-attention over 70 joint queries" — stale comment from baseline, not a functional issue.

---

## Summary

All required design changes are present and correctly implemented. Linear hand recovery module and auxiliary loss with weight 0.1 confirmed. All invariants preserved. Test run completed without errors. APPROVED.
