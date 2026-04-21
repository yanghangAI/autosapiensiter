# Code Review Log — idea011/design001

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Two-pass coordinate-conditioned decoder with shared weights and no
intermediate supervision is faithfully implemented. `pose3d_transformer_head.py`
adds the three new kwargs with correct defaults (`num_refine_passes=1`,
`shared_decoder=True`, `intermediate_supervision_weight=0.0`), builds
`self.coord_enc` (Linear-GELU-Linear) with zero-init on the final Linear's
weight AND bias, implements the `forward()` two-pass / residual-output
logic with the `num_refine_passes <= 1` short-circuit, and extends `loss()`
with a correctly guarded `loss/joints_init/train` branch that is inactive
for Design 001 (weight 0.0). `config.py` sets `num_refine_passes=2`,
`shared_decoder=True`, `intermediate_supervision_weight=0.0` — all
int/bool/float literals. `pelvis_utils.py` and `train.py` are
bit-identical to baseline. `review-check-implementation` passed. The
reduced test-train produced 81 iterations with sensible, decreasing
losses, completed validation, and emitted only the three baseline-shape
loss keys (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`) as
expected.
