# Code Review Log — idea014 / design001

- 2026-04-17: APPROVED. review-check-implementation passed. Head kwargs, conditional `depth_out` allocation, `log_bin_centres` non-persistent buffer, soft-argmax expectation in `forward()`, and SORD soft-CE in `loss()` all match design.md. No aux regression loss emitted (Design 001 value `depth_aux_reg_weight=0.0`). Test SLURM job 55739450 completed; metrics.csv and iter_metrics.csv present with expected three loss keys. No invariant files modified.
