# Code Review — idea028/design003

**Verdict: APPROVED**

## review-check-implementation
Passed.

## Files Changed vs. Design
- `code/pose3d_transformer_head.py` — required and changed. Correct.
- `code/config.py` — required and changed. Correct.
- `code/pelvis_utils.py` — present (no changes, consistent with design). Correct.

## Implementation Fidelity

### `pose3d_transformer_head.py`
- `_PelvisCrossAttnDecoder` class defined at module level, before `Pose3dTransformerHead`. Correct.
- Four new kwargs (`use_decoupled_pelvis`, `pelvis_hidden_dim`, `pelvis_num_heads`, `num_body_queries`) in `__init__`, with correct defaults. Correct.
- Instance attributes `self.use_decoupled_pelvis`, `self.pelvis_hidden_dim`, `self.num_body_queries` stored. `self.num_joints = num_joints` (=70) also stored, separate from `num_body_queries=22`. Correct.
- `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` — changed from `num_joints` to `num_body_queries`, producing a 22-embedding table for body-only decoder. Correct.
- Conditional block after `self.uv_out`: `pelvis_coord_queries` and `pelvis_decoder` instantiated when `use_decoupled_pelvis=True`. Correct.
- `_init_head_weights()`: `trunc_normal_` on `joint_queries.weight` (now shape (22, hidden_dim)), existing loop covers output projections, conditional `trunc_normal_` on `pelvis_coord_queries`. Correct.
- `forward()`:
  - Queries broadcast: `(B, num_body_queries, hidden_dim)` = `(B, 22, hidden_dim)`. Correct.
  - Decoder operates on 22 queries. Correct.
  - `body_joints = self.joints_out(decoded)` → `(B, 22, 3)`. Correct.
  - Zero-pad: `torch.zeros(B, self.num_joints - self.num_body_queries, 3, ...)` = 48 hand joints zeros. `torch.zeros` has `requires_grad=False`. Correct.
  - `joints = torch.cat([body_joints, pad], dim=1)` → `(B, 70, 3)`. Correct.
  - Conditional pelvis path identical to design001 (since `use_decoupled_pelvis=True`): dedicated pelvis decoder from `pelvis_coord_queries`. Correct.
- Return dict: `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)`. Correct.
- `loss()`: `_BODY = list(range(0, 22))` — unchanged, covers exactly the 22 active body joints. Correct.
- `predict()`: `self.num_joints = 70` used for `keypoint_scores` shape. Correct.

### `config.py`
- Head dict contains `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=8`, `num_body_queries=22`. All literal values, no Python imports. Correct.
- All other config values identical to baseline. Correct.

## Invariant Verification
No modifications to: `pelvis_utils.py`, `train.py`, evaluation metric, dataset, transforms, backbone, data preprocessor, infra files. Confirmed.

## Test Output
- All three loss components present in `iter_metrics.csv` (72 iterations, 1 epoch). No NaN or inf values.
- SLURM log confirms: clean startup, weights loaded (293/293 backbone tensors), 1 epoch trained, checkpoint saved, "[test] Finished." — no errors.
- `loss/joints/train` ~0.17–0.23 (slightly lower than design001/002, expected with 22-query body-only decoder), `loss/depth/train` ~1.7–3.7, `loss/uv/train` ~0.05–0.30. All finite, no divergence.
