# Code Review Log — idea014 / design003

- 2026-04-17: APPROVED. review-check-implementation passed. Adaptive bin mode allocates `depth_bins_head`, computes per-sample widths via softmax × range + cumsum + midpoints, per-sample `sigma_log` from median adjacent log-centre spacing, and detaches SORD target so width head is trained solely by SmoothL1 aux. `depth_aux_reg_weight=0.3` active; `loss/depth_reg/train` emitted. SLURM job 55739454 completed without crashes. No invariant files modified.
