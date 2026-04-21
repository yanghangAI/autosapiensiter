# Code Review — idea018/design003

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea018/design003` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are within the allowed set. `pelvis_utils.py` verified identical to baseline (no diff).

---

## Implementation vs. Design Checklist

### `_DecoderLayer.forward()` — optional `attn_logit_bias` argument

Design requires: identical to Designs 001 and 002 modification.

Code at lines 104–142: Identical expansion pattern. All requirements met.

### `Pose3dTransformerHead.__init__()` — combined kwargs and modules

Design requires:
- `num_body_queries: int = 70`
- `hand_aux_loss_weight: float = 0.0`
- `depth_gate_type: str = 'none'`
- `depth_gate_sigma: float = 1.0`
- `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` (replaces `nn.Embedding(num_joints, hidden_dim)`)
- `hand_proj: Linear(num_body_queries * hidden_dim, (num_joints - num_body_queries) * 3)` if `num_body_queries < num_joints`; else `hand_proj = None`
- `depth_probe_global`, `depth_probe_token`, `depth_gate_sigma_buf` when `depth_gate_type == 'gaussian'`
- `self.num_body_queries` and `self.hand_aux_loss_weight` stored as attributes

Code at lines 170–247:
- All four new kwargs present with correct defaults (lines 185–188)
- `self.num_body_queries = num_body_queries` at line 201, `self.hand_aux_loss_weight = hand_aux_loss_weight` at line 202
- `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` at line 212 — correct replacement
- `self.num_joints = num_joints` remains 70 (line 198) — `predict()` will work
- `hand_proj` conditional at lines 223–230 — guard `num_body_queries < num_joints`, uses dynamic computation (`num_body_queries * hidden_dim`, `(num_joints - num_body_queries) * 3`), no hardcoded 48 or 22
- Depth gate modules at lines 236–245 — identical to Design 001 (`gaussian` type)

All requirements met.

### `_init_head_weights()` — trunc-normal init for hand_proj, zero-init for depth probes

Design requires:
- `trunc_normal_(hand_proj.weight, std=0.02)` and `zeros_(hand_proj.bias)` when `hand_proj is not None`
- `zeros_()` for all four depth probe weights/biases when `depth_gate_type == 'gaussian'`

Code at lines 253–270: Both blocks present in correct order (hand_proj init, then depth probe zero-init). Guards `self.hand_proj is not None` and `self.depth_gate_type == 'gaussian'` correct.

### `forward()` — depth gate + 22-query decoder + body/hand split

Design requires:
- Gate computation identical to Design 001 (`gaussian` type, `depth_gate_sigma_buf`, no z_hat caching)
- `queries` from `self.joint_queries.weight` — shape `(B, 22, hidden_dim)` when `num_body_queries=22`
- `body_joints = self.joints_out(decoded)` — shape `(B, 22, 3)`
- When `hand_proj is not None`: `body_flat = decoded.reshape(B, num_body_queries * hidden_dim)`, `hand_joints = hand_proj(body_flat).reshape(B, num_hand, 3)`, `joints = cat([body_joints, hand_joints], dim=1)` — shape `(B, 70, 3)`
- `pelvis_token = decoded[:, 0, :]` — token 0 of decoded output

Code at lines 305–344:
- Gate computed at lines 310–318 (identical to Design 001)
- No `_depth_probe_z_hat` caching — correct (no auxiliary probe loss in Design 003)
- `queries` from `self.joint_queries.weight` — produces `(B, 22, hidden_dim)` in active config
- `body_joints = self.joints_out(decoded)` at line 325
- `body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)` — uses `self.num_body_queries` and `self.hidden_dim`, not hardcoded (line 329)
- `num_hand = self.num_joints - self.num_body_queries` dynamically computed (line 330) — no hardcoded 48
- `joints = torch.cat([body_joints, hand_joints], dim=1)` at line 332 — produces `(B, 70, 3)`
- `pelvis_token = decoded[:, 0, :]` at line 336 — correct

All requirements met.

### `loss()` — auxiliary hand loss

Design requires:
- After `loss/uv/train`: `if self.hand_aux_loss_weight > 0.0 and self.hand_proj is not None:`
- `_HAND = list(range(self.num_body_queries, self.num_joints))`
- `losses['loss/hand_aux/train'] = hand_aux_loss_weight * loss_joints_module(pred['joints'][:, _HAND], gt_joints[:, _HAND])`

Code at lines 390–393: Exactly this pattern. `_HAND` computed dynamically. Loss key is `'loss/hand_aux/train'`. Uses `self.loss_joints_module` — correct (hand joints are positional coordinates like body joints).

### `config.py` — head kwargs

Design requires: `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0` as literals.

Config at lines 162–165: All four present as int/float/str literals. All other config sections match baseline.

---

## Invariant File Check

- `pelvis_utils.py`: identical to baseline (diff = empty)
- `train.py`: identical to baseline
- No changes to invariant infra files

---

## Output Shape Check

`forward()` always returns `joints (B, 70, 3)` — guaranteed by `cat([body_joints (B,22,3), hand_joints (B,48,3)])`. `pelvis_depth (B, 1)` and `pelvis_uv (B, 2)` unchanged. `predict()` uses `self.num_joints = 70` — correct.

---

## Test Output Check

SLURM log (`slurm_test_55859006.out`):
- Training ran successfully to completion ("Done training!")
- Epoch 1 loss line: `loss: 2.893733  loss/joints/train: 0.220412  loss/depth/train: 2.475702  loss/uv/train: 0.147381  loss/hand_aux/train: 0.050237  grad_norm: inf`
- Four loss keys present: three baseline keys plus `loss/hand_aux/train` — exactly as designed
- `loss/hand_aux/train: 0.050237` is non-zero and finite — `hand_proj` is active and receiving gradients
- No `loss/depth_probe/train` key — correct (Design 003 has no auxiliary probe loss)
- `iter_metrics.csv`: 72 rows, all three tracked columns populated. `loss/hand_aux/train` not tracked by `MetricsCSVHook` — expected.
- Checkpoint saved at epoch 1. Training started and ended cleanly.
- Memory: 8630 MB — slightly higher than Design 001 (8629 MB) due to `hand_proj` parameters; well within the 24G SLURM budget.

---

## Issues Found

None. The implementation faithfully matches every required detail in design.md. The compositional combination of 22-query body-only decoder and fixed-sigma Gaussian depth gate is correctly implemented.
