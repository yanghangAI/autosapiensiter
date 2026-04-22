## 2026-04-22 16:28 UTC — Design Review

Reviewer: Reviewer agent (design review mode).

Result: APPROVED.

Checked:
- idea.md read for context.
- design.md fully specifies: Design Description, starting point `baseline/`, exhaustive list of files to change (only the three experimentable files), exact code patches for `pelvis_utils.py` helpers, exact `__init__`/`forward`/`loss` modifications in `pose3d_transformer_head.py`, exact config kwargs/values.
- Verified against `runs/baseline/code/pose3d_transformer_head.py`:
  - `spatial = feat.flatten(2).transpose(1, 2)` (row-major, H outer, W inner) — matches design.
  - `pelvis_token = decoded[:, 0, :]`; `self.uv_out = nn.Linear(hidden_dim, 2)` — matches the gated replacement described.
  - `pred['pelvis_uv']` shape and SmoothL1 loss — design preserves both.
- Invariants: no changes to metric, dataset, transforms, backbone, preprocessor, infra, train.py.
- MMEngine config constraint: all new kwargs are literals.

No infrastructure/automation bugs encountered.
