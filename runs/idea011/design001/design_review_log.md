# Design Review Log — idea011/design001

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Two-pass coordinate-conditioned decoder with shared weights, no intermediate supervision. Adds zero-init `coord_enc: Linear(3,256)->GELU->Linear(256,256)` in `pose3d_transformer_head.py` that feeds pass-1 joints back into pass-2 queries; residual readout `joints_final = joints_1 + joints_residual`; pelvis read from pass-2 token 0. Config adds `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0`. Defaults (`num_refine_passes=1, shared_decoder=True, intermediate_supervision_weight=0.0`) preserve baseline behaviour; `coord_enc[2]` zero-init ensures init-time baseline-match. Only `pose3d_transformer_head.py` and `config.py` touched; `pelvis_utils.py` untouched. All invariants preserved. Implementation-ready.
