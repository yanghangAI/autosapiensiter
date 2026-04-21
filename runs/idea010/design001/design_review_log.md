# Design Review Log — idea010/design001

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Auxiliary 2D reprojection Smooth-L1 loss on body joints 0-21, lambda=0.5. Changes confined to `pelvis_utils.py` (new `project_joints_to_2d` helper), `pose3d_transformer_head.py` (new `reproj_loss_weight` kwarg and loss block after UV loss), and `config.py` (head kwarg addition). Projection formula matches the baseline `recover_pelvis_3d` convention; `X>=0.01` clamp; smooth_l1 beta=0.05; loss key `'loss/reproj/train'`. All invariants preserved. Implementation-ready.
