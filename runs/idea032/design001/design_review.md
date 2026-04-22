# Design 001 — Design Review

**Verdict:** APPROVED

**Date:** 2026-04-22 17:01 UTC

## Summary

Design 001 adds an auxiliary zero-init `Linear(256, 1)` head on projected spatial tokens that regresses bilinearly-downsampled input metric depth at the 40x24 feature grid, supervised by SmoothL1(beta=0.1) with foreground mask 0.1 m < d < 30 m, weighted by λ=0.1.

## Checks Passed

- **Starting point:** `baseline/` — explicit.
- **Files to modify:** only `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — all allowed.
- **Invariants preserved:** `bedlam2_transforms.py`, `bedlam2_dataset.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `infra/*`, `train.py`, `tools/train.py` are untouched. Body-only joint loss (0-21), persistent_workers=False, absolute imports, and MMEngine-config literal-only rule are all respected.
- **Algorithmic spec:** complete. Exact code blocks given for:
  - `downsample_depth_map` helper in `pelvis_utils.py`
  - module-level `_rgbd_capture_pre_hook` registered once (with idempotent guard)
  - nine new `__init__` kwargs with defaults
  - `forward()` insertion point (after `spatial = spatial + pos_enc`, before `queries = ...`)
  - `loss()` insertion point (after UV-loss line, before `with torch.no_grad():` block)
- **Config values explicit:** `use_aux_depth=True, aux_depth_loss_weight=0.1, aux_depth_log_space=False, aux_depth_grad_weight=0.0, aux_depth_valid_min=0.1, aux_depth_valid_max=30.0, aux_depth_denorm_scale=20.0, feat_h=40, feat_w=24` — all literals.
- **Denorm scale verified against infra:** `_DEPTH_MAX_METERS = 20.0` in `mmpose/datasets/transforms/bedlam2_transforms.py:87`; `PackBedlamInputs` divides depth by that constant. Multiplying captured input channel by 20.0 correctly recovers metric depth.
- **Feature grid verified:** `img_h=640, img_w=384` in baseline `config.py`; backbone stride 16 gives 40x24. Matches `feat_h=40, feat_w=24`.
- **Zero-init correctness:** weight and bias zero-init ensures step-0 prediction is zero and gradient into `spatial` is zero at step 0, so main losses are numerically unchanged at initialization.
- **Edge cases covered:** empty-mask fallback (`pred.sum() * 0.0`), validation-time forward safe (predict() never reads `_aux_depth_pred`), preemption/resume safe (hook re-registered on module import, not persisted in checkpoint), zero-fill pixels excluded by lower bound 0.1 m.

## Infrastructure Concern (Noted, Not Blocking)

The design uses `torch.nn.modules.module.register_module_forward_pre_hook` to intercept any 4-D float tensor with `shape[1]==4` entering any module. This is an unusual but non-invasive mechanism that avoids editing invariant transforms/dataset/preprocessor/estimator. The filter (4D + C==4 + floating) is strict enough that only the RGBD backbone input should match under normal operation. The design documents the overwrite-staleness behaviour and device-keyed storage correctly. No blocker, but the Builder should be aware this is a global side-effect; if a future change introduces any other 4D C=4 float tensor (e.g., in an augmentation batch), the hook could capture it. Acceptable given current invariants.

## Required Preservations (Repeated for Builder)

- Hook registration guard `_RGBD_CAPTURE_HOOK_REGISTERED` must be present.
- `predict()` path unchanged; must not depend on `_aux_depth_pred`.
- `self._aux_depth_pred = None` initialized at end of `__init__` and cleared after consumption in `loss()`.
- No `import` statements in `config.py`.

## Verdict

APPROVED. The Builder has sufficient detail (exact code, exact insertion points, exact kwargs, exact config snippet) to implement without guessing.
