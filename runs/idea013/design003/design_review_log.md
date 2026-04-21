# Design Review Log — idea013 / design003

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea013 (Kinematic Chain Bone-Vector Output Parameterization)
- Design: 003 (Bone-vector head + five decoupled per-limb `Linear(256, 3)`
  output projections; `per_limb_heads=True`, bone-length auxiliary off)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py, bedlam_metric.py,
  bedlam2_dataset.py, bedlam2_transforms.py, sapiens_rgbd.py, data
  preprocessor, infra/*, train.py wrapper, tools/train.py).
- Key acceptance points:
  - Shared kinematic-parametrization spec (kwargs, buffer, scale-init,
    `_forward_kinematics`, assertions) reused from Design 001.
  - Per-limb head construction: `nn.ModuleList([nn.Linear(hidden_dim, 3)
    for _ in range(5)])` as `self.body_limb_heads`; original
    `self.joints_out` kept for hand tokens.
  - `limb_index` registered as non-persistent long buffer; five
    `_limb_idx_{0..4}` non-persistent long buffers registered in
    `__init__` to avoid per-forward allocation.
  - `per_limb_heads=True` requires `kinematic_parametrization=True`
    (assert in `__init__`); `num_limbs == 5` asserted.
  - `_limb_token_lists` covers the 22 body indices exactly — spine 6,
    left_leg 4, right_leg 4, left_arm 4, right_arm 4 = 22.
  - `_init_head_weights` trunc-normal-initialises each per-limb head
    (`std=0.02`) and in-place scales each `.weight` by `1/math.sqrt(21)`
    inside a single `torch.no_grad()` block, alongside the original
    `joints_out.weight` scaling.
  - `forward()` per-limb routing: `decoded.new_zeros(B, 22, 3)` canvas;
    `decoded.index_select(1, idx)` gather; `body_bone_vecs
    .index_copy_(1, idx, bone_vecs_limb)` scatter — autograd-safe,
    five iterations, negligible overhead.
  - Hand tokens pass through unchanged `self.joints_out`; recovered
    body coordinates concatenated with hand coords along `dim=1` →
    `(B, 70, 3)`.
  - Pelvis depth/UV pathway unchanged — `pelvis_token = decoded[:, 0, :]`.
  - `loss()` unchanged (three loss keys; bone-length auxiliary off);
    `predict()` untouched.
  - Config values: `kinematic_parametrization=True`,
    `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
    `bone_length_loss_weight=0.0`, `per_limb_heads=True`,
    `limb_index=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4]`.
    Literals only — MMEngine-config compliant.
  - Extra learnable params: 4 × (256 × 3 + 3) = 3084 floats; <0.001% of
    model. Param growth correctly attributed to "four *additional* heads
    beyond the baseline's single shared head."
  - Zero impact on the evaluation metric; tensor shapes and dtypes
    identical to baseline at inference.
- Verdict: APPROVED.
