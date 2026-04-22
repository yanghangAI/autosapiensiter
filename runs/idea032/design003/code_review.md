# Code Review — idea032 / design003

**Verdict:** APPROVED

## Checks

- `review-check-implementation` passed.
- `implementation_summary.md` lists the three experimentable files; no invariant files modified; `code/train.py` unchanged from baseline.
- `pose3d_transformer_head.py` is identical to design001's head (shared scaffolding, per design spec). The gradient-consistency branch gated by `if self.aux_depth_grad_weight > 0:` is present inside `loss()`, computes `dx_pred/dy_pred` and `dx_tgt/dy_tgt` on the (log-space) `pred` and `target` tensors, then `grad_loss = (dx_pred - dx_tgt).abs().mean() + (dy_pred - dy_tgt).abs().mean()`, and adds `self.aux_depth_grad_weight * grad_loss` to `recon_loss` before the outer λ multiplication. Grad term is correctly computed in the same log-space as `target` (invariant 13) and is unmasked (invariant 14).
- `pelvis_utils.py` matches design001/002.
- `config.py` sets `use_aux_depth=True, aux_depth_loss_weight=0.3, aux_depth_log_space=True, aux_depth_grad_weight=0.5, aux_depth_valid_min=0.1, aux_depth_valid_max=30.0, aux_depth_denorm_scale=20.0, feat_h=40, feat_w=24` — exactly as specified. All literals; no imports. Resulting aux loss is `0.3 * (recon + 0.5 * grad)` in log-space.
- `test_output/slurm_test_*.out`: training ran one epoch; `loss/aux_depth/train: 0.566436` (higher than design002's 0.5436, consistent with the added grad term); checkpoint saved; `[test] Finished.`
- Invariants preserved: body-only joint loss, `predict()` unchanged, zero-init on aux head, no invariant files modified, config uses only literals.

No discrepancies.
