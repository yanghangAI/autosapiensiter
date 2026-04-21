# Code Review — idea011/design003

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 003 (two-pass decoder with INDEPENDENT pass-2 weights via
`self.decoder_layer_2`, plus intermediate supervision) is faithfully
implemented. The head correctly gates `decoder_layer_2` behind
`shared_decoder=False AND num_refine_passes >= 2`, uses it in the
refinement pass via the `else` branch in `forward()`, and keeps the
intermediate supervision term active. The reduced test-train ran to
completion and the MMEngine log confirms the new decoder layer's
parameters are registered and that the auxiliary loss is being emitted.

---

## Pre-check

- `python scripts/cli.py review-check-implementation runs/idea011/design003`:
  **PASSED**.

## implementation_summary.md audit

`**Files changed:**` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are allowed per the design. `pelvis_utils.py` correctly not
listed. Summary is non-empty.

Every change described in `**Changes:**` maps to actual code:

- Three new `__init__` kwargs with baseline-preserving defaults —
  `pose3d_transformer_head.py:169-171`, stored at lines 184-186.
- `self.coord_enc` (Sequential Linear-GELU-Linear) with zero-init on
  final Linear — lines 206-210 and 247-252.
- Conditional `self.decoder_layer_2 = _DecoderLayer(hidden_dim,
  num_heads, dropout)` only when `shared_decoder=False AND
  num_refine_passes >= 2` — lines 217-219. Constructor args match
  `self.decoder_layer`.
- `forward()` routes pass 2 through `self.decoder_layer_2` via the
  `else` branch (line 321) when `shared_decoder=False`; the shared-weights
  branch still exists (line 319) for future or alternate configs.
- `loss()` emits `loss/joints_init/train` when
  `intermediate_supervision_weight > 0.0` — lines 380-385.
- `predict()` unchanged.

No undocumented changes were found.

## Design-detail fidelity

Every required detail from `design.md` is implemented correctly:

1. Three new kwargs and defaults match Design 002 (verified).
2. `coord_enc` built unconditionally with zero-init (verified).
3. `self.decoder_layer_2` construction is correctly gated by
   `(not self.shared_decoder) and self.num_refine_passes >= 2` (line 217).
   When `shared_decoder=True` (Design 001/002 configs), the attribute is
   NOT created, preserving their parameter count.
4. `_DecoderLayer(hidden_dim, num_heads, dropout)` args match
   `self.decoder_layer` exactly (verified, line 218).
5. No custom init on `self.decoder_layer_2`; default PyTorch init is
   used — matches baseline treatment of `self.decoder_layer`.
6. `forward()` short-circuit on `num_refine_passes <= 1` still present
   (lines 299-309).
7. Residual formulation `joints_cur = joints_cur + joints_residual`
   (line 323), final = `joints_cur`.
8. Shared `self.joints_out` for pass 1 and pass 2 residual readout.
9. No `.detach()` in the forward path.
10. `forward()` returns dict with `joints`, `joints_initial`,
    `pelvis_depth`, `pelvis_uv`.
11. Intermediate supervision loss key is exactly `'loss/joints_init/train'`
    (line 381) and uses body joints 0-21.
12. `predict()` body unchanged.
13. `config.py` adds `num_refine_passes=2`, `shared_decoder=False`,
    `intermediate_supervision_weight=0.5` — verified (lines 147-149);
    all int/bool/float literals (no imports).
14. `paramwise_cfg` unchanged; new `coord_enc` and `decoder_layer_2`
    params fall under default head LR (1e-4) — confirmed by the
    `paramwise_options` dump in the slurm log showing head params with
    `lr=1e-4, lr_mult` not listed (no backbone tag).
15. `persistent_workers=False` preserved.
16. Head uses absolute imports.

## Invariant compliance

Diffed against baseline:
- `code/pelvis_utils.py` — bit-identical to baseline.
- `code/train.py` — bit-identical to baseline.
- Only `pose3d_transformer_head.py` and `config.py` are modified.

No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`,
`bedlam2_transforms.py`, `sapiens_rgbd.py`, the data preprocessor,
`infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or
`tools/train.py`.

## test_output verification

- Slurm job 55670847 ran without exceptions. No `Traceback`, `Error`,
  or `Exception` strings in stdout.
- `iter_metrics.csv` has 66 training iteration rows with
  sensible, decreasing losses. Early losses are slightly higher than
  Designs 001/002 (as expected — the fresh `decoder_layer_2` starts
  from random init and has to learn more), but convergence is healthy
  (joints loss drops from 0.494 at iter 1 to 0.268 at iter 39).
- The MMEngine log (`20260417_142405/20260417_142405.log`) confirms
  `head.decoder_layer_2.self_attn.in_proj_weight`,
  `head.decoder_layer_2.cross_attn.*`, `head.decoder_layer_2.ffn.*`,
  and all three layer-norms are present in the init summary —
  confirming the independent decoder layer is constructed and
  registered in the parameter tree.
- The Epoch(train) summary line shows all four loss keys:
  `loss: 1.817985  loss/joints/train: 0.347960
  loss/joints_init/train: 0.104254
  loss/depth/train: 1.241403
  loss/uv/train: 0.124368`.
  The intermediate supervision term IS active and being summed into the
  total loss.
- Backbone loads 293/293 pretrained tensors; training progresses through
  iteration 50 of epoch 1.

## Issues

None.

---

**Final verdict: APPROVED**
