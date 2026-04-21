# Design Review — idea012 / design003

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 003 specifies the same upper-triangular pairwise auxiliary loss as
Design 001 but applied to `log(d + eps)` with `eps=1e-3` (1 mm), producing a
scale-invariant / proportion-aware loss. The design is complete, explicit,
and implementable without guesswork.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files are modified: `pose3d_transformer_head.py`,
  `config.py`. `pelvis_utils.py` explicitly unchanged.
- [x] Exact algorithmic change: same `torch.cdist` + `torch.triu_indices(22,
  22, offset=1, device=...)` setup as Design 001; mode `'log'` computes
  `L_dist = (torch.log(d_pred + eps) - torch.log(d_gt + eps)).abs().mean()`
  with `eps = self.dist_loss_eps = 1e-3`. Scaled by
  `self.dist_loss_weight=0.5` into key `'loss/dist_matrix/train'`.
- [x] `__init__` signature: three new kwargs (`dist_loss_weight`,
  `dist_loss_mode`, `dist_loss_eps`) placed after `loss_weight_uv` and
  before `init_cfg`, with defaults preserving baseline behaviour. Assert on
  mode. `bone_parents` kwarg is optional (defaults to `None`) per Design 002
  superset compatibility; `self.bone_weights` resolves to `None` in this
  mode.
- [x] Exact config values: `dist_loss_weight=0.5`, `dist_loss_mode='log'`,
  `dist_loss_eps=1e-3`. Only literals, MMEngine-config compliant.
- [x] Numerical safety: `eps` added INSIDE each `torch.log(...)` — documented
  rationale (avoids `log(0)` at coincident joints). `eps=1e-3` bounds
  `1/(d+eps) ≤ 1000`; `clip_grad max_norm=1.0` provides additional safety.
  Justification for NOT using smaller `eps` (e.g., `1e-6`) is given.
- [x] `torch.abs` at 0 subgradient behaviour (picks 0 automatically) noted.
- [x] `forward()` and `predict()` explicitly unchanged. Training-only loss.
- [x] `if/elif/else` structure documented; the `elif 'bone_weighted'` branch
  is unreachable in Design 003 (mode `'log'` selected in config), so
  `self.bone_weights` being `None` never triggers an AttributeError.
- [x] Invariants preserved: `persistent_workers=False`, body-only loss slice,
  `custom_imports`, absolute imports in head file, seed 2026, batch size 4,
  gradient accumulation 8, LR schedule unchanged, hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`, `bedlam2_dataset.py`,
  `bedlam2_transforms.py`, backbone, data preprocessor, `infra/*`,
  `train.py`, `tools/train.py`.
- [x] Zero new learnable parameters; zero buffers in Design 003.

## Verdict

APPROVED — Builder can implement without guessing.
