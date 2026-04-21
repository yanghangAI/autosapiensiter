# Code Review — idea008/design003

**Verdict: APPROVED**

---

## Review Check

`python scripts/cli.py review-check-implementation runs/idea008/design003` — PASSED.

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

1. **`num_body_queries: int = 22` and `hand_aux_loss_weight: float = 0.3` kwargs** — PRESENT (lines 151, 154). Stored as `self.num_body_queries` and `self.hand_aux_loss_weight` (lines 173, 174). Default of 0.3 matches design. CORRECT.
2. **`self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)`** — PRESENT (line 186). CORRECT.
3. **`self.num_joints = 70` preserved** — PRESENT (line 172). CORRECT.
4. **2-layer MLP `hand_proj`**: `nn.Sequential(Linear(num_body_queries * hidden_dim, hidden_dim), GELU(), Linear(hidden_dim, num_hand * 3))`** — PRESENT (lines 197–202). Evaluates to `Linear(5632, 256) → GELU → Linear(256, 144)`. `num_hand` computed dynamically as `num_joints - num_body_queries`. Bottleneck dim is `hidden_dim` (256) per design constraint 5. GELU activation per design constraint 6. CORRECT.
5. **`_init_head_weights`: each `nn.Linear` in `hand_proj` initialised with `trunc_normal_(std=0.02)` weight and zero bias** — PRESENT (lines 222–226). Iterates with `isinstance(layer, nn.Linear)`. CORRECT.
6. **`forward()`: `body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)`** — PRESENT (line 272). CORRECT.
7. **`hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)`** — PRESENT (lines 273–274). CORRECT.
8. **`joints = torch.cat([body_joints, hand_joints], dim=1)` → `(B, 70, 3)`** — PRESENT (line 276). CORRECT.
9. **`pelvis_token = decoded[:, 0, :]`** — PRESENT (line 278). Unchanged. CORRECT.
10. **`loss()`: auxiliary hand loss `loss/hand_aux/train = 0.3 * loss_joints_module(pred_joints[:, _HAND], gt_joints[:, _HAND])`** — PRESENT (lines 332–334). Uses `self.hand_aux_loss_weight` (0.3) and reuses `self.loss_joints_module`. `_HAND = list(range(22, 70))`. CORRECT.
11. **Body joint loss unchanged with `_BODY = list(range(0, 22))`** — PRESENT (lines 322–325). CORRECT.
12. **`_DecoderLayer` unchanged** — CORRECT.
13. **Constructor signature** matches design spec. CORRECT.

### config.py

1. **`num_body_queries=22` and `hand_aux_loss_weight=0.3` in `model.head` dict** — PRESENT (lines 136, 139). Both are literal values. MMEngine-compliant. CORRECT.
2. **All other config values identical to baseline** — seed 2026, batch 4, accum 8, persistent_workers=False, optimizer, LR schedule, data pipeline, hooks all match baseline. CORRECT.
3. **`output_dir` set to `…/idea008/design003`** — CORRECT.

---

## Invariant Checks

- `persistent_workers=False` in both dataloaders — PRESERVED.
- Backbone, data preprocessor, metric, transforms — NOT modified.
- `train.py` wrapper — IDENTICAL to baseline.
- `pelvis_utils.py` — IDENTICAL to baseline.
- Seed `2026`, batch `4`, accumulation `8` — PRESERVED.
- MMEngine config: no Python `import` statements; all new values are literals. COMPLIANT.
- Auxiliary hand loss uses existing `self.loss_joints_module` — no new loss module instantiated. CORRECT per design constraint 7.
- MLP activation is `nn.GELU()`, not ReLU — CORRECT per design constraint 6.
- Bottleneck dimension is `hidden_dim` (256) — CORRECT per design constraint 5.

---

## Test Output

- Test ran to completion: 1 epoch, no Python errors or tracebacks.
- `metrics.csv`: `epoch=1, composite_val=450.28, mpjpe_body_val=439.90, mpjpe_pelvis_val=471.36` — all columns present. Values are high (only 1 warmup epoch). Expected.
- SLURM log confirms `loss/hand_aux/train: 0.060898` is being logged during training, confirming the auxiliary loss is active with the stronger 0.3 weight. CORRECT.
- `iter_metrics.csv`: 81 iterations logged. Loss values are numerically stable. No NaN/Inf. CORRECT.

---

## Minor Observations (non-blocking)

- Module docstring still describes "self-attention over 70 joint queries" — stale comment inherited from baseline, not a functional issue.
- Memory usage at 10634 MB (vs. 10611 for design001, 10623 for design002), consistent with the MLP's ~1.47M extra parameters. Within 1080 Ti budget.

---

## Summary

All required design changes are present and correctly implemented. 2-layer MLP hand recovery (`Linear(5632,256) → GELU → Linear(256,144)`) and auxiliary loss weight 0.3 confirmed in code and active in training logs. All invariants preserved. Test run completed without errors. APPROVED.
