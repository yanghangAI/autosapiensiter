# Design Review Log — idea011/design002

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Two-pass coordinate-conditioned decoder, shared weights, WITH intermediate supervision on pass-1 body joints (weight=0.5). Architecture and `forward()` identical to design001; `loss()` gains `loss/joints_init/train = 0.5 * SoftWeightSmoothL1(joints_initial[:, 0-21], gt_joints[:, 0-21])` via the shared `self.loss_joints_module`. Pelvis supervision stays on pass-2 outputs only. Config adds `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.5`. Defaults preserve baseline. Only `pose3d_transformer_head.py` and `config.py` touched. All invariants preserved. Implementation-ready.
