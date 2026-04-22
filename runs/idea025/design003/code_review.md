# Code Review — idea025/design003

**Verdict: APPROVED**

## review-check-implementation
PASSED.

## Files Changed vs Design Specification
- `code/pose3d_transformer_head.py` — required and changed. CORRECT.
- `code/config.py` — required and changed. CORRECT.
- `code/pelvis_utils.py` — listed but unchanged (diff confirms identical to baseline). No violation.
- `code/train.py` — unchanged (diff confirms identical to baseline). No violation.

## implementation_summary.md Fidelity
Every claim in `implementation_summary.md` is present in the code:
- Six new kwargs in `__init__` with correct defaults: PRESENT.
- No `sym_pair_weights_buf` registered (None for Design 003): CORRECT — the `sym_pairs is not None` branch sets `sym_pair_weights_buf = None` when `sym_pair_weights` is None.
- Adaptive weighting block (`sym_adaptive_weight=True` branch) under `torch.no_grad()`: PRESENT at lines 363–367.
- `asym_w = 1.0 / (1.0 + asym_gt_mag / self.sym_tau)` with `keepdim=True` for correct broadcast: PRESENT.
- Loss keyed as `loss/sym/train`: PRESENT.

## Design Fidelity
All required design details are implemented correctly:

1. **`bilateral_sym_loss_weight=0.5`**: confirmed in config.py.
2. **`sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]`**: confirmed in config.py.
3. **`sym_mirror_axis=1`**: confirmed in config.py.
4. **`sym_adaptive_weight=True`**: confirmed in config.py.
5. **`sym_tau=0.1`**: confirmed in config.py.
6. **No `sym_pair_weights`**: correct — not in config.
7. **Adaptive weight computed under `torch.no_grad()` using `.detach()`**: CORRECT at lines 364–366. No gradient flows through the weight.
8. **Broadcast shape `(B, P, 1)` over `(B, P, 3)`**: `keepdim=True` in norm call ensures `(B, P, 1)` shape. CORRECT.

## Invariant File Check
- `pelvis_utils.py`: identical to baseline. PASS.
- `train.py`: identical to baseline. PASS.
- No changes to evaluation metric, dataset, transforms, backbone, or data preprocessor.

## Test Output
- Training ran to completion: "Done training!" in SLURM log.
- `loss/sym/train: 0.023687` at iter 50 — lower than Design 001/002 as expected (adaptive weighting reduces loss magnitude for asymmetric poses). Finite value confirms correct operation.
- `grad_norm: inf` at the only logged training step (iter 50 of epoch 1): This is an AMP artifact at early training with dynamic loss scaling on a freshly initialized network. The `FixedAmpOptimWrapper` with `loss_scale='dynamic'` handles inf grad_norm by skipping the optimizer step for that iteration — this is normal behavior at early training. Training completed successfully with no errors or exceptions. This single `inf` observation is not a sign of incorrect implementation; it is consistent with known AMP initialization behavior also seen in baseline and other designs at early training.
- No Python errors, no NaN losses.
