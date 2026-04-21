# Code Review Log — idea012/design001

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Upper-triangular pairwise L1 distance-matrix auxiliary loss (mode `'abs'`,
`dist_loss_weight=0.5`) is faithfully implemented.
`pose3d_transformer_head.py` adds the three new kwargs
(`dist_loss_weight=0.0`, `dist_loss_mode='abs'`, `dist_loss_eps=1e-3`)
with the specified defaults, stores them as attributes, asserts
`dist_loss_mode in ('abs', 'bone_weighted', 'log')`, and appends the
guarded auxiliary-loss block inside `loss()` after the three baseline
losses and before the `torch.no_grad()` MPJPE block. The block uses
`torch.cdist(..., p=2)` + `torch.triu_indices(22, 22, offset=1, device=...)`
to gather the 231 upper-tri pairs, computes `(d_pred - d_gt).abs().mean()`
for the `'abs'` branch, and stores the result under the exact key
`'loss/dist_matrix/train'`. The three-branch `if/elif/else` skeleton is
present (unreachable `'bone_weighted'` and `'log'` branches match Designs
002/003). `forward()`, `predict()`, and `_init_head_weights` are
unchanged. `config.py` sets `dist_loss_weight=0.5`, `dist_loss_mode='abs'`,
`dist_loss_eps=1e-3` as literals inside the `head=dict(...)` block.
`pelvis_utils.py` and `train.py` are bit-identical to baseline.
`review-check-implementation` passed. The reduced test-train completed
81 iterations + validation cleanly; the MMEngine log shows all four loss
keys including `loss/dist_matrix/train: 0.257460` on the first printed
step, confirming the new loss backprops correctly. Note: `iter_metrics.csv`
lacks a column for the new loss because `infra/metrics_csv_hook.py` has a
hardcoded column list, but this hook is an invariant file and the primary
validation metrics in `metrics.csv` are unaffected; not a blocking issue.
