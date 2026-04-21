# Design Review — idea012 / design001

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 001 specifies a minimal upper-triangular pairwise L1 distance-matrix
auxiliary loss on the 22 body joints (231 pairs) with `dist_loss_weight=0.5`,
`dist_loss_mode='abs'`, `dist_loss_eps=1e-3`. The design is complete, explicit,
and implementable without guesswork.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files are modified: `pose3d_transformer_head.py`,
  `config.py`. `pelvis_utils.py` explicitly unchanged.
- [x] Exact algorithmic change: `torch.cdist(pred_body, pred_body, p=2)` on
  body-only slice `_BODY = list(range(0, 22))`; upper-triangular gather via
  `torch.triu_indices(22, 22, offset=1, device=pred_body.device)`; plain
  `(d_pred - d_gt).abs().mean()` for mode `'abs'`; scaled by
  `self.dist_loss_weight` into key `'loss/dist_matrix/train'`.
- [x] Insertion point in `loss()` unambiguous: after the three existing loss
  assignments, before the `with torch.no_grad():` MPJPE block.
- [x] `__init__` signature change precise: three new kwargs
  (`dist_loss_weight: float = 0.0`, `dist_loss_mode: str = 'abs'`,
  `dist_loss_eps: float = 1e-3`) placed after `loss_weight_uv` and before
  `init_cfg`, with defaults that preserve baseline behaviour, plus an assert
  validating `dist_loss_mode ∈ {'abs','bone_weighted','log'}`.
- [x] Exact config values: `dist_loss_weight=0.5`, `dist_loss_mode='abs'`,
  `dist_loss_eps=1e-3` appended to the `head=dict(...)` block. Only
  literals, MMEngine-config compliant (no `import` required).
- [x] `forward()` and `predict()` explicitly unchanged. Training-only loss.
- [x] Invariants preserved: `persistent_workers=False`, body-only loss slice,
  `custom_imports`, absolute imports in head file, seed 2026, batch size 4,
  gradient accumulation 8, LR schedule unchanged, hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`, `bedlam2_dataset.py`,
  `bedlam2_transforms.py`, backbone, data preprocessor, `infra/*`,
  `train.py`, `tools/train.py`.
- [x] Edge cases handled: diagonal excluded via `offset=1` avoiding the
  `cdist` zero-distance gradient pitfall; `torch.triu_indices(..., device=...)`
  avoids host-device copies; explicit `.mean()` not `.sum()` so loss scale is
  batch-size-independent; `if self.dist_loss_weight > 0.0:` guard makes the
  baseline (weight=0.0) bit-identical.
- [x] Expected output correctly described: four loss keys including
  `'loss/dist_matrix/train'`; `MetricsCSVHook` auto-picks up the new key.
- [x] Zero new learnable parameters; zero buffers in Design 001.

## Minor observations (non-blocking)

- The `loss()` block contains a three-branch `if/elif/else` on
  `self.dist_loss_mode`. The `'bone_weighted'` branch references
  `self.bone_weights`, which is NOT created in Design 001. Because
  `dist_loss_mode='abs'` is fixed in config, the branch is unreachable at
  runtime, and the design document explicitly notes this. The Builder should
  keep the branch as written (for Designs 002/003 superset compatibility) —
  it will not execute under Design 001's configuration. Acceptable.

## Verdict

APPROVED — Builder can implement without guessing.
