# Code Review Log — idea035/design003

## 2026-05-05 — APPROVED
- review-check-implementation passed.
- Only allowed files changed; `pelvis_utils.py` identical to baseline.
- `DepthAblationDataPreprocessor.forward` `mode='shuffle'` branch implements per-sample `torch.randperm` permutation over `H*W` on channel 3 only; preserves marginal histogram, destroys spatial alignment; RGB untouched.
- Config wired with `mode='shuffle'`.
- No invariant files modified; reduced test run finished cleanly (`slurm_test_56728567.out`), iter metrics produced.
