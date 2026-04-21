# Code Review — idea011/design001

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 001 (two-pass coordinate-conditioned decoder, shared weights, no
intermediate supervision) is faithfully implemented. All required changes
from `design.md` are present in `code/pose3d_transformer_head.py` and
`code/config.py`. No invariant files were modified. The reduced test-train
ran without errors, loss curves are well-behaved, and the loss schema
matches the expected baseline (no `loss/joints_init/train` term because
`intermediate_supervision_weight=0.0`).

---

## Pre-check

- `python scripts/cli.py review-check-implementation runs/idea011/design001`:
  **PASSED**.

## implementation_summary.md audit

`**Files changed:**` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are allowed per the design. `pelvis_utils.py` is correctly not
listed. The summary is non-empty, so no "Builder did nothing" reject.

Every change described in `**Changes:**` maps to actual code:

- Three new kwargs with defaults `num_refine_passes=1, shared_decoder=True,
  intermediate_supervision_weight=0.0` — present at
  `pose3d_transformer_head.py:169-171`; stored as attributes at
  `pose3d_transformer_head.py:184-186`.
- `self.coord_enc = nn.Sequential(Linear(3, hidden_dim), GELU,
  Linear(hidden_dim, hidden_dim))` — present at
  `pose3d_transformer_head.py:206-210`, placed AFTER `self.decoder_layer`
  (line 199) and BEFORE `self.joints_out` (line 214), as specified.
- Zero-init on both weight AND bias of `self.coord_enc[2]`, with
  trunc-normal on `self.coord_enc[0]` — present at
  `pose3d_transformer_head.py:238-243`.
- Two-pass `forward()` with short-circuit for `num_refine_passes <= 1`,
  residual joint output `joints_final = joints_1 + joints_residual`, and
  pelvis read from pass-2 `decoded_cur[:, 0, :]` — present at
  `pose3d_transformer_head.py:290-327`.
- `loss()` extended with conditional `loss/joints_init/train` guarded by
  `self.intermediate_supervision_weight > 0.0 and 'joints_initial' in
  pred` — present at `pose3d_transformer_head.py:371-376`. For Design 001
  the weight is 0.0, so the branch is not taken (confirmed by the training
  log showing only `loss/joints/train`, `loss/depth/train`,
  `loss/uv/train`).
- `torch.no_grad()` MPJPE block unchanged (reads `pred['joints']`).
- `predict()` body unchanged.

No undocumented changes were found.

## Design-detail fidelity

Every required detail from `design.md` is implemented correctly:

1. Kwarg signature ordering: after `loss_weight_uv`, before `init_cfg` — verified.
2. Default values preserve baseline behaviour: `num_refine_passes=1`,
   `shared_decoder=True`, `intermediate_supervision_weight=0.0` — verified.
3. `coord_enc` built unconditionally — verified (line 206).
4. `coord_enc[2]` weight AND bias both zero-initialised — verified
   (lines 241-243).
5. Short-circuit branch `if self.num_refine_passes <= 1` returns
   baseline-equivalent dict with pelvis from pass-1 token 0 — verified
   (lines 290-300).
6. Residual formulation `joints_cur = joints_cur + joints_residual` —
   verified (line 314). Final `joints_final` is the summed residual, not
   the raw pass-2 absolute prediction.
7. `joints_out` is shared between pass 1 and pass 2 (single
   `self.joints_out` module called on `decoded_1` and `decoded_next`) —
   verified.
8. No `.detach()` in the forward path — verified (`joints_cur` is fed
   into `coord_enc` without detachment).
9. Pelvis depth/UV read from `decoded_cur[:, 0, :]` (pass-2 token 0) in
   the refinement path — verified (line 318).
10. `forward()` returns dict with keys `joints`, `joints_initial`,
    `pelvis_depth`, `pelvis_uv` — verified.
11. `config.py` adds `num_refine_passes=2, shared_decoder=True,
    intermediate_supervision_weight=0.0` — verified (lines 147-149);
    all int/bool/float literals (no imports).
12. `custom_imports` list and pretrained weights unchanged — verified.
13. `persistent_workers=False` preserved in both dataloaders —
    verified (lines 178, 193).
14. Head uses absolute imports — verified (lines 31-35).

## Invariant compliance

Diffed against baseline:
- `code/pelvis_utils.py` — bit-identical to baseline.
- `code/train.py` — bit-identical to baseline.
- Only `pose3d_transformer_head.py` and `config.py` are modified.

No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`,
`bedlam2_transforms.py`, `sapiens_rgbd.py`, the data preprocessor,
`infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or
`tools/train.py`.

The training log in the `test_output` shows the backbone still loads
293/293 tensors from the pretrained Sapiens checkpoint, confirming the
backbone code path is unchanged.

## test_output verification

- Slurm job 55670841 ran without exceptions. No `Traceback`, `Error`,
  or `Exception` strings in stdout.
- `iter_metrics.csv` has 81 training iteration rows (epoch 1 completed)
  with all three loss columns populated with sensible values (loss
  decreasing from ~0.38 at iter 1 to ~0.20 at iter 81 for
  `loss_joints_train`).
- The MMEngine log (`20260417_142316/20260417_142316.log`) confirms
  head weights were initialised (including `head.coord_enc.0.weight`,
  `head.coord_enc.2.weight`, and biases) and that training proceeded
  through validation (`Epoch(val) [1][50/76]`).
- The log line `Epoch(train) [1][50/81] ... loss/joints/train: 0.270038
  loss/depth/train: 0.767458 loss/uv/train: 0.144068` confirms only the
  three baseline-shape loss keys are emitted (no `loss/joints_init/train`),
  matching the Design 001 spec.

## Issues

None.

---

**Final verdict: APPROVED**
