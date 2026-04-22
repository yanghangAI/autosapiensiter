# Code Review — idea032 / design001

**Verdict:** APPROVED

## Checks

- `review-check-implementation` passed.
- `implementation_summary.md` lists three files changed (`pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py`) — all experimentable; no invariant files touched. `train.py` in `code/` is byte-identical to baseline.
- `pelvis_utils.py`: `import torch.nn.functional as F` added; `downsample_depth_map(depth_map, feat_h, feat_w)` appended exactly as specified (bilinear interpolate + squeeze(1)). Existing `recover_pelvis_3d` / `compute_mpjpe_abs` untouched.
- `pose3d_transformer_head.py`:
  - `import torch.nn.functional as F` added.
  - `pelvis_utils` import extended to include `downsample_depth_map as _downsample_depth_map`.
  - Module-level `_LAST_RGBD_INPUT` dict and `_rgbd_capture_pre_hook` added with the required filter (`dim()==4`, `shape[1]==4`, floating-point), `.detach()` applied, idempotent registration via `_RGBD_CAPTURE_HOOK_REGISTERED` guard on `torch.nn.modules.module.register_module_forward_pre_hook`.
  - `__init__` signature extended with the nine aux-depth kwargs in the specified order; each is stored on `self`; `self.aux_depth_head = nn.Linear(hidden_dim, 1)` built when `use_aux_depth=True` with `nn.init.zeros_` on weight and bias; `self._aux_depth_pred = None` initialised.
  - `forward()`: right after `spatial = spatial + pos_enc`, the aux prediction is computed and stored on `self._aux_depth_pred` with shape `(B, feat_h, feat_w)`; else `None`.
  - `loss()`: the aux-depth block is placed after the UV-loss line and before the `with torch.no_grad():` MPJPE block as specified. It reads `_LAST_RGBD_INPUT[device]`, extracts channel 3, denormalises by `aux_depth_denorm_scale=20.0`, downsamples via the helper, casts GT to the prediction dtype (AMP-safe extra), applies the log-space branch when enabled, masks to `(0.1, 30.0)`, uses `F.smooth_l1_loss(beta=0.1)` with an empty-mask `pred.sum()*0.0` fallback, includes the optional gradient term behind `aux_depth_grad_weight > 0`, writes `losses['loss/aux_depth/train'] = aux_depth_loss_weight * recon_loss`, and clears `self._aux_depth_pred = None`. Defensive `losses['loss/aux_depth/train'] = self._aux_depth_pred.sum() * 0.0` fallback present when no RGBD capture is available.
  - `predict()` is unchanged; body-only `_BODY = list(range(0, 22))` restriction preserved.
- `config.py`: the nine literals `use_aux_depth=True, aux_depth_loss_weight=0.1, aux_depth_log_space=False, aux_depth_grad_weight=0.0, aux_depth_valid_min=0.1, aux_depth_valid_max=30.0, aux_depth_denorm_scale=20.0, feat_h=40, feat_w=24` are appended to the `head=dict(...)` block. No Python `import` introduced.
- `test_output/slurm_test_55985500.out`: training ran one epoch without error; log line shows `loss/aux_depth/train: 0.721559` alongside the existing joint/depth/uv losses; checkpoint saved; `[test] Finished.`
- Invariants preserved: body-only joint loss, `persistent_workers=False` (not modified), zero-init on aux head, absolute imports, no config imports, `predict()` unchanged.

No discrepancies between `implementation_summary.md`, `design.md`, and the code.
