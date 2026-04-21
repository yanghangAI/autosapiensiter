# Code Review — idea019/design001

**Verdict: APPROVED**

---

## Review Summary

### 1. review-check-implementation
Passed with exit code 0.

### 2. Files Changed
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.
- `code/config.py` — required by design.

Both files are among those permitted. No unexpected files modified. `pelvis_utils.py` and `train.py` are unchanged and match baseline (verified by diff).

### 3. Implementation vs. Design Fidelity

**`_DeformableDecoderLayer` class:**
- Added after `_DecoderLayer`, before `Pose3dTransformerHead` — correct placement.
- `__init__` signature matches design: `embed_dim`, `num_heads=8`, `dropout=0.1`, `num_points=8`, `deform_hidden_dim=64`, `num_queries=70`. `assert embed_dim % num_heads == 0` present.
- `offset_net`: `Linear(embed_dim, deform_hidden_dim) → GELU → Linear(deform_hidden_dim, num_points*2)` — matches exactly.
- `ref_points`: `nn.Parameter(torch.full((num_queries, 2), 0.5))` — correct.
- `attn_weight_net`, `value_proj`, `out_proj`, FFN, norms, dropouts — all match design spec.
- `_sample_spatial_features`: offsets scaled by 0.1, clamped to [0,1], converted to [-1,1], reshaped to `(B, Nq*K_s, 1, 2)`, AMP dtype cast `grid = grid.to(spatial_grid.dtype)` present, `grid_sample` with `mode='bilinear', padding_mode='border', align_corners=True` — all required details present.
- `forward`: pre-norm self-attn → deformable cross-attn (norm2 → sample → attn_weight → value_proj → out_proj → dropout) → FFN (norm3) — matches design.

**`Pose3dTransformerHead.__init__`:**
- New kwargs: `deform_num_points=0`, `deform_hidden_dim=64`, `num_body_queries=70`, `num_decoder_layers=1`, `hand_aux_loss_weight=0.0`, `aux_body_loss_weight=0.0` — all present with correct defaults.
- All kwargs stored as instance attributes.
- `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` — uses `num_body_queries`, not `num_joints`.
- `decoder_layers` built as `nn.ModuleList` for both deform and non-deform paths. The design called for a `decoder_layer = self.decoder_layers[0]` backward-compat alias; the implementation instead omits it with comment "no separate self.decoder_layer alias to avoid duplicate params." This is a minor deviation from the design spec, but the deviation is sound (avoids double-counting parameters) and the forward path uses `decoder_layers` throughout — no code path references `decoder_layer` (singular). Acceptable deviation.
- `has_hand_proj`, `has_intermediate_sup` guards present and inactive for Design 001 (`num_body_queries=70`, `aux_body_loss_weight=0.0`).

**`_init_head_weights`:**
- Near-zero init for `offset_net[-1].weight/bias` and `attn_weight_net.weight/bias` — present.
- `trunc_normal_(value_proj.weight, std=0.02)` + zero bias, same for `out_proj` — present.
- Intermediate supervision head init present under `has_intermediate_sup` guard.
- Hand proj init present under `has_hand_proj` guard.

**`forward()`:**
- Spatial feature processing: flatten → `input_proj` → add pos_enc → reshape to 2D grid when deformable — correct.
- Deformable path: iterates `decoder_layers`, collects intermediate outputs (for Design 003), sets `self._intermediate_decoded`.
- Non-deformable path (baseline fallback): iterates `decoder_layers` over flat spatial tokens.
- For Design 001: `joints = body_joints` (no `has_hand_proj`), output `(B, 70, 3)` — correct.
- `pelvis_token = decoded[:, 0, :]` — correct.
- Output dict shape `{'joints': (B,70,3), 'pelvis_depth': (B,1), 'pelvis_uv': (B,2)}` — unchanged.

**`loss()`:**
- `_BODY = list(range(0, 22))` — body joint loss restricted to indices 0–21.
- `has_intermediate_sup` guard — inactive for Design 001.
- `hand_aux_loss_weight > 0` guard — inactive for Design 001.
- Both conditions correctly inactive.

**`config.py`:**
- All required literals present: `deform_num_points=8`, `deform_hidden_dim=64`, `num_body_queries=70`, `num_decoder_layers=1`, `hand_aux_loss_weight=0.0`, `aux_body_loss_weight=0.0`. `in_channels=1024` hardcoded.
- No Python `import` statements inside any dict.
- All other config values (optimizer, LR schedule, data pipeline, hooks, backbone, seed, batch size, accumulation) identical to baseline.

### 4. Invariant Files
- `pelvis_utils.py`: diff vs. baseline shows no changes.
- `train.py`: diff vs. baseline shows no changes.
- Backbone, dataset, transforms, metric, data preprocessor, infra files — not present in `code/` (not touched).

### 5. Test Output
- `slurm_test_55859332.out`: Training ran to completion. "Done training!" reached. No errors or exceptions.
- Loss log at iter 50/72: `loss/joints/train: 0.196`, `loss/depth/train: 2.597`, `loss/uv/train: 0.144` — all finite, sensible magnitudes. No NaN or Inf values observed.
- `iter_metrics.csv`: 72 rows recorded for epoch 1. All loss values finite.
- GPU: RTX 2080 Ti (correct partition), memory ~8616 MB — within budget.
- AMP (FixedAmpOptimWrapper) active. No dtype mismatch errors under grid_sample.
- Model loaded 293/293 backbone tensors. Head randomly initialised. No load errors.

### 6. Notable Observations
- The `decoder_layer` (singular) backward-compat alias was omitted by the Builder. The design explicitly required it. However, no code path in the head file references `decoder_layer` (singular) after construction — the omission has no functional consequence. No regression risk.
- Test ran cleanly with no OOM, no NaN losses, no runtime errors.
