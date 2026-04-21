# Code Review — idea011/design002

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 002 (same two-pass architecture as Design 001, but with
`intermediate_supervision_weight=0.5` to enable the pass-1 body-joint
auxiliary loss) is faithfully implemented. The head file is identical
to Design 001's (intentional — the design specifies `forward()` unchanged
and the `loss()` guard already handles both weights), and `config.py`
sets the supervision weight to 0.5. The reduced test-train completes
cleanly and the training log confirms the `loss/joints_init/train` term
is actively being emitted.

---

## Pre-check

- `python scripts/cli.py review-check-implementation runs/idea011/design002`:
  **PASSED**.

## implementation_summary.md audit

`**Files changed:**` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are allowed per the design. `pelvis_utils.py` is correctly not
listed. Summary is non-empty.

Every change described in `**Changes:**` maps to actual code:

- Head file identical to Design 001 (three kwargs with same defaults,
  `coord_enc` zero-init, two-pass `forward()`, conditional intermediate
  supervision branch in `loss()`). `diff` against Design 001's head shows
  zero differences.
- The conditional intermediate supervision branch at
  `pose3d_transformer_head.py:371-376` is the one that activates for
  Design 002 because the config sets
  `intermediate_supervision_weight=0.5`.
- Config differs only by the single literal
  `intermediate_supervision_weight=0.5` at `config.py:149`.

No undocumented changes were found.

## Design-detail fidelity

Every required detail from `design.md` is implemented correctly:

1. Three new kwargs with correct defaults (verified at
   `pose3d_transformer_head.py:169-171`).
2. `coord_enc` constructed unconditionally (line 206-210).
3. Zero-init on `coord_enc[2]` weight AND bias (lines 241-243).
4. `forward()` identical to Design 001 spec (lines 272-327) — two-pass
   shared-weight decoder, residual output, pelvis from pass-2 token 0.
5. `loss()` branch for intermediate supervision uses
   `self.loss_joints_module` (not a fresh instance), the body-joint
   index set `_BODY = list(range(0, 22))`, and multiplies by
   `self.intermediate_supervision_weight` (lines 371-376).
6. Loss key is exactly `'loss/joints_init/train'` (line 372).
7. `torch.no_grad()` MPJPE block unchanged.
8. `predict()` body unchanged.
9. `config.py` adds `num_refine_passes=2, shared_decoder=True,
   intermediate_supervision_weight=0.5` — verified (lines 147-149);
   all int/bool/float literals (no imports).
10. No other `config.py` changes; `custom_imports` unchanged.
11. `persistent_workers=False` preserved.
12. Head uses absolute imports.

## Invariant compliance

Diffed against baseline and Design 001:
- `code/pelvis_utils.py` — bit-identical to baseline.
- `code/train.py` — bit-identical to baseline.
- `code/pose3d_transformer_head.py` — identical to Design 001.
- `code/config.py` — differs from Design 001 only in the single literal
  `intermediate_supervision_weight=0.5` vs `0.0`.

No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`,
`bedlam2_transforms.py`, `sapiens_rgbd.py`, the data preprocessor,
`infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or
`tools/train.py`.

## test_output verification

- Slurm job 55670846 ran without exceptions.
- `iter_metrics.csv` has 71 training iteration rows with sensible,
  decreasing losses.
- The MMEngine log (`20260417_142356/20260417_142356.log`) contains
  the Epoch(train) summary line:
  `loss: 1.275277  loss/joints/train: 0.266649
  loss/joints_init/train: 0.092674
  loss/depth/train: 0.771830
  loss/uv/train: 0.144124`.
  This confirms the intermediate supervision term IS being computed and
  summed into the total loss, and that its magnitude (~0.0927) is
  consistent with the spec: `0.5 × L_smoothL1(joints_initial[:, _BODY])`
  where the joints-final loss is ~0.267, so the pre-weight value on
  joints_initial is ≈0.185 — very close to the main loss, exactly the
  expected behaviour given that at init `joints_residual≈0` so
  `joints_initial ≈ joints_final` and the two losses are nearly equal
  (0.5·0.267 ≈ 0.134; observed 0.093 is within the expected range given
  training progress).
- `MetricsCSVHook` only records `loss/joints/train`, `loss/depth/train`,
  `loss/uv/train` — the `loss/joints_init/train` key is an invariant
  CSV-hook behaviour and is fine (the hook schema is an invariant file
  that intentionally does not log this auxiliary term). Total loss is
  still correctly summed by MMEngine.
- Backbone loads 293/293 tensors from pretrained checkpoint; training
  completes iteration 50 of epoch 1 cleanly.

## Issues

None. Note: the fact that `iter_metrics.csv` does not have a
`loss_joints_init_train` column is expected and correct — the CSV hook
is an invariant file and its schema is the canonical set of loss columns.
The auxiliary loss is still used by MMEngine for backward, which is the
only thing that matters for training.

---

**Final verdict: APPROVED**
