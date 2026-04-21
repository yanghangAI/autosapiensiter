# Design Review Log — idea013 / design001

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea013 (Kinematic Chain Bone-Vector Output Parameterization)
- Design: 001 (Bone-vector head, minimal — single shared head,
  `1/sqrt(21)` scale-init, no bone-length auxiliary)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py, bedlam_metric.py,
  bedlam2_dataset.py, bedlam2_transforms.py, sapiens_rgbd.py, data
  preprocessor, infra/*, train.py wrapper, tools/train.py).
- Key acceptance points:
  - `__init__` adds five new kwargs with baseline-preserving defaults
    (all `False`/`0.0`/`None`); placed after `loss_weight_uv` and before
    `init_cfg`. Backward compatible.
  - `bone_parents` registered as non-persistent long buffer; Python int
    list cached as `self._bone_parents_list` for host-side indexing to
    avoid per-iteration host/device syncs.
  - Topological-ordering assertion (`parent[child] < child`) guarantees
    the single in-order `for child in range(1, 22)` forward-kinematics
    loop is correct.
  - `_forward_kinematics` clones its input, overwrites root to zero, and
    writes into distinct slots — autograd-safe, no aliasing.
  - `_init_head_weights` scales `joints_out.weight` in-place by
    `1/math.sqrt(21)` inside `torch.no_grad()` when
    `kinematic_parametrization=True`. Bias unchanged (already zero).
  - `forward()` splits body/hand slices and concatenates
    `[body_rr, hand_coords]` along `dim=1` — `(B, 70, 3)` shape
    preserved.
  - `loss()` body term reads the recovered coordinate tensor; the
    optional bone-length block is guarded by `> 0.0` so Design 001
    (`weight=0.0`) skips it entirely — same three loss keys as baseline.
  - `predict()` untouched.
  - Config values: `kinematic_parametrization=True`,
    `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
    `bone_length_loss_weight=0.0`, `per_limb_heads=False`,
    `limb_index=None`. Literals only — MMEngine-config compliant.
  - Parent list identical to idea012 (already validated there).
  - Zero new learnable parameters. Buffer `bone_parents` ~176 bytes.
  - Forward-kinematics compute: 21 adds + 1 small clone = <0.2 ms on
    1080 Ti. Memory/speed negligible.
- Verdict: APPROVED.
