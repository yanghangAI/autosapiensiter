# Code Review — idea028/design001

**Verdict: APPROVED**

## review-check-implementation
Passed.

## Files Changed vs. Design
- `code/pose3d_transformer_head.py` — required and changed. Correct.
- `code/config.py` — required and changed. Correct.
- `code/pelvis_utils.py` — present (no changes, consistent with design specifying no changes). Correct.
- `code/train.py` — present but listed in no design-specified changes. Confirmed it is the unmodified invariant wrapper.

## Implementation Fidelity

### `pose3d_transformer_head.py`
- `_PelvisCrossAttnDecoder` is defined at module level, after `_build_2d_sincos_pos_enc` and before `_DecoderLayer` — acceptable (the design says "after `_DecoderLayer`", but placement before `_DecoderLayer` still satisfies the module-level accessibility requirement; the class is defined before `Pose3dTransformerHead` uses it). The `_DecoderLayer` ordering note is a comment about proximity, not a correctness constraint — the module compiles and runs correctly.
- Module body matches design exactly: `nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)`, `nn.LayerNorm(embed_dim)`, pre-norm residual forward.
- Four new kwargs (`use_decoupled_pelvis`, `pelvis_hidden_dim`, `pelvis_num_heads`, `num_body_queries`) added after `loss_weight_uv`, before `init_cfg`. Defaults match design.
- Instance attributes `self.use_decoupled_pelvis`, `self.pelvis_hidden_dim`, `self.num_body_queries` stored correctly.
- `self.joint_queries = nn.Embedding(num_joints, hidden_dim)` — correct for design001 (`num_body_queries=70`, same as `num_joints=70`).
- Conditional block after `self.uv_out`: `pelvis_coord_queries` (Embedding(2, pelvis_hidden_dim)) and `pelvis_decoder` (`_PelvisCrossAttnDecoder(pelvis_hidden_dim, pelvis_num_heads, dropout)`) instantiated when `use_decoupled_pelvis=True`. Correct.
- `_init_head_weights()`: existing loop covers `depth_out`/`uv_out`; conditional `trunc_normal_` on `pelvis_coord_queries.weight` when `use_decoupled_pelvis`. Correct.
- `forward()`: joint decoder path unchanged (70 queries, `_DecoderLayer`). Conditional pelvis path: `pelvis_coord_queries.weight.unsqueeze(0).expand(B,-1,-1)`, `pelvis_decoder(pelvis_qs, spatial)`, `depth_out` from `[:, 0, :]`, `uv_out` from `[:, 1, :]`. Exact match with design.
- Return dict keys `joints`, `pelvis_depth`, `pelvis_uv` — identical to baseline.
- `loss()` and `predict()` unchanged. Correct.

### `config.py`
- Head dict contains `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=8`, `num_body_queries=70`. All literal values, no Python imports. Correct.
- All other config values (optimizer, LR schedule, batch size 4, accum 8, seed 2026, `persistent_workers=False`, `resume=True`, `max_keep_ckpts=1`) are identical to baseline.

## Invariant Verification
No modifications to: `pelvis_utils.py`, `train.py`, evaluation metric, dataset, transforms, backbone, data preprocessor, infra files. Confirmed.

## Test Output
- All three loss components present in `iter_metrics.csv` (72 iterations, 1 epoch). No NaN or inf values.
- SLURM log confirms: clean startup, weights loaded (293/293 backbone tensors), 1 epoch trained, checkpoint saved, "[test] Finished." — no errors or exceptions.
- `loss/joints/train` ~0.19–0.25, `loss/depth/train` ~1.5–3.7, `loss/uv/train` ~0.04–0.30 — reasonable finite values, no divergence.
