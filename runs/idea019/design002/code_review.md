# Code Review — idea019/design002

**Verdict: APPROVED**

---

## Review Summary

### 1. review-check-implementation
Passed with exit code 0.

### 2. Files Changed
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.
- `code/config.py` — required by design.

Both files permitted. No unexpected files modified. `pelvis_utils.py` and `train.py` unchanged and matching baseline (verified by diff).

### 3. Implementation vs. Design Fidelity

The `pose3d_transformer_head.py` for Design 002 is identical to Design 001's file (verified by diff: no differences). This is correct because the design explicitly states "The Builder must implement `_DeformableDecoderLayer` exactly as specified in Design 001" and all Design 002 differences are controlled entirely through config kwargs.

**Design-002-specific behaviour gated by config:**
- `num_body_queries=22`: `joint_queries = nn.Embedding(22, 256)`. The `has_hand_proj = (22 < 70) = True` guard activates `hand_proj = Linear(22*256, 48*3)` = `Linear(5632, 144)` — matches design.
- `hand_aux_loss_weight=0.1`: activates `loss/hand_aux/train` in `loss()` over `_HAND = list(range(22, 70))` — matches design.
- `aux_body_loss_weight=0.0`, `num_decoder_layers=1`: `has_intermediate_sup = False` — correct for Design 002.

**`forward()` with `num_body_queries=22`:**
- `decoded` shape: `(B, 22, 256)`.
- `body_joints = joints_out(decoded)`: `(B, 22, 3)`.
- `body_flat = decoded.reshape(B, 22*256=5632)`.
- `hand_joints = hand_proj(body_flat).reshape(B, 48, 3)`.
- `joints = cat([body_joints, hand_joints], dim=1)`: `(B, 70, 3)` — output shape invariant maintained.
- `pelvis_token = decoded[:, 0, :]` — query 0 remains pelvis token.

**`loss()` for Design 002:**
- `_HAND = list(range(self.num_body_queries, self.num_joints))` = `range(22, 70)` — 48 indices.
- `loss/hand_aux/train = 0.1 * loss_joints_module(pred['joints'][:, _HAND], gt_joints[:, _HAND])` — reuses `loss_joints_module`, no new module. Matches design.

**`config.py`:**
- `num_body_queries=22`, `num_decoder_layers=1`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0` — all correct literals.
- All other values (optimizer, schedule, seed, batch, accumulation, data pipeline, hooks) identical to baseline.

### 4. Invariant Files
Same as Design 001 — unchanged. `pelvis_utils.py` diff vs. baseline: clean.

### 5. Test Output
- `slurm_test_55859333.out`: Training ran to completion. "Done training!" reached. No errors.
- Loss log at iter 50/72: `loss/joints/train: 0.201`, `loss/depth/train: 2.595`, `loss/uv/train: 0.146`, `loss/hand_aux/train: 0.039` — all finite. Auxiliary hand loss appearing as expected.
- GPU memory: 8629 MB — slightly above Design 001 (8616 MB), consistent with larger `hand_proj` parameter. No OOM.
- AMP active. No dtype errors.
- Model loaded correctly.

### 6. Notable Observations
- `loss/hand_aux/train: 0.039` at iter 50 of epoch 1 is at 0.1× scale relative to joints loss, as expected.
- The shared `pose3d_transformer_head.py` file correctly handles all three designs through conditional guards, driven by config kwargs — clean and correct implementation strategy.
