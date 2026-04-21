# Code Review Log — idea012/design003

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Log-scaled / scale-invariant pairwise L1 distance-matrix auxiliary loss
(mode `'log'`, `dist_loss_weight=0.5`, `dist_loss_eps=1e-3`) is
faithfully implemented. `pose3d_transformer_head.py` adds the three new
kwargs (`dist_loss_weight=0.0`, `dist_loss_mode='abs'`,
`dist_loss_eps=1e-3`) with correct defaults and placement, stores them
as attributes, asserts `dist_loss_mode in ('abs', 'bone_weighted',
'log')`, and sets the sentinel `self.bone_weights = None` (Design 003
uses mode `'log'` so no buffer is created). In `loss()`, after the
three baseline losses and before the `torch.no_grad()` MPJPE block, the
guarded three-branch block is added; the `'log'` branch computes
`(torch.log(d_pred + eps) - torch.log(d_gt + eps)).abs().mean()` with
`eps = self.dist_loss_eps = 1e-3` (inside both logs as required) and
stores the result with the exact key `'loss/dist_matrix/train'`,
multiplied by `self.dist_loss_weight`. `forward()`, `predict()`, and
`_init_head_weights` are unchanged. `config.py` sets
`dist_loss_weight=0.5`, `dist_loss_mode='log'`, `dist_loss_eps=1e-3`
as literals. `pelvis_utils.py` and `train.py` are bit-identical to
baseline. `review-check-implementation` passed. The reduced test-train
completed 81 iterations + validation cleanly; the MMEngine log shows
`loss/dist_matrix/train: 0.787721`, finite and positive with no NaN/Inf
and no gradient explosion (`grad_norm: 8.47`, within `clip_grad
max_norm=1.0`-adjusted range). No blocking issues.
