# Design Review — idea018/design003

**Verdict: APPROVED**

---

## Review Summary

Design 003 combines the 22-query body-only decoder (from idea008/design002) with the fixed-sigma Gaussian depth gate (from idea018/design001). The design is complete, unambiguous, and implementation-ready. It is the most complex of the three designs but is fully specified.

---

## Checklist

### Feasibility
- The composition of 22-query decoder + depth gate is architecturally clean. The gate `(B, 960)` broadcasts over 22 query dimension uniformly — physically justified (all body joints in roughly the same depth plane).
- `hand_proj: Linear(22*256, 48*3) = Linear(5632, 144)` is feasible. Parameter count (810,576) is noted and acceptable.
- The cross-attention shape with 22 queries: `(B*num_heads, 22, 960) = (32, 22, 960)` for batch=4, heads=8 — correct and consistent with `_DecoderLayer.forward()` expansion logic.
- `decoded.reshape(B, self.num_body_queries * self.hidden_dim)` requires `self.hidden_dim` to be accessible — it is, since baseline sets `self.hidden_dim = hidden_dim` in `__init__`.
- `self.num_joints` remains 70 (set by `BaseHead` or explicitly) — confirmed by constraint 4. The `predict()` method uses `self.num_joints` for `keypoint_scores` shape, which must remain 70.

### Completeness
- Starting point: `baseline/` — specified.
- Files changed: only `pose3d_transformer_head.py` and `config.py` — within the allowed set.
- `pelvis_utils.py`: explicitly no changes.
- All five modification points covered: `_DecoderLayer.forward()`, `__init__()`, `_init_head_weights()`, `forward()`, `loss()`.
- Config additions: `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0` — all int/float/str literals, compliant.
- Joint embedding change: `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` — replacing the baseline's `nn.Embedding(num_joints, hidden_dim)`. Explicit.
- Body joint loss: `_BODY = list(range(0, 22))` unchanged — still body-only (first 22 joints).
- Hand auxiliary loss uses `list(range(self.num_body_queries, self.num_joints))` dynamically — correct.
- `hand_proj` guard `if num_body_queries < num_joints` ensures baseline-compatible fallback when `num_body_queries=70`.

### Explicitness
- The `forward()` replacement of the joints output block is fully specified: `body_joints → hand_joints via hand_proj → cat → joints (B, 70, 3)`.
- `pelvis_token = decoded[:, 0, :]` — uses token 0 of the 22-query decoded output. Explicitly justified (constraint 6).
- `_init_head_weights()` specifies `trunc_normal_(hand_proj.weight, std=0.02)` and `zeros_(hand_proj.bias)` — consistent with baseline's output projection init.
- The `_HAND = list(range(self.num_body_queries, self.num_joints))` in `loss()` is computed dynamically — correct.
- Constraint 3 makes explicit that `num_hand = num_joints - num_body_queries` must NOT be hardcoded as 48.
- Loss keys: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train` — four total; consistent with the expected training log.
- No `_depth_probe_z_hat` caching needed (no auxiliary probe loss in Design 003, as `depth_probe_loss_weight` is not a kwarg here).
- AMP: both `hand_proj` and depth gate operations are float16-safe.

### Invariant Compliance
- Invariant files not touched.
- `persistent_workers=False` preserved.
- Output `joints` shape: `(B, 70, 3)` — guaranteed by `cat([body_joints, hand_joints], dim=1)` with 22+48=70.
- `self.num_joints = 70` preserved — `predict()` will work correctly.
- MMEngine config no-import constraint satisfied.

### Issues Found

**Potential issue — `self.num_joints` initialization order:** Design 003 adds `num_body_queries` as a new constructor kwarg, and replaces `self.joint_queries = nn.Embedding(num_joints, hidden_dim)` with `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)`. The `self.num_joints` attribute is set by the baseline `__init__` before the embedding line. The Builder must verify that `self.num_joints` is set to `num_joints` (70) in `__init__` — in the baseline this is `self.num_joints = num_joints` on line 170. This is inherited and the design does not change it. **Non-blocking — just verify.**

**Potential issue — `queries` variable name in `forward()`:** The baseline uses `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)`. With `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)`, this produces `(B, 22, hidden_dim)` when `num_body_queries=22`. The variable name `queries` is still used but now has shape `(B, 22, hidden_dim)`. The design is explicit about this: `decoded: (B, num_body_queries, hidden_dim) = (B, 22, 256)`. **Non-blocking — consistent.**

---

## Notes for Builder

1. Do NOT hardcode `48` or `22` anywhere — always compute as `num_joints - num_body_queries` and `num_body_queries` dynamically.
2. `self.num_joints` must remain `70` (set by the baseline's `self.num_joints = num_joints` line). Do not override it.
3. `hand_proj` is only created when `num_body_queries < num_joints`. When `num_body_queries=70` (the default), `hand_proj is None` and the `else` branch in `forward()` is taken — exactly the baseline.
4. The `_DecoderLayer.forward()` modification is identical to Designs 001 and 002. If all three designs share the same head file in different code directories, this change must be applied to each independently.
5. `loss/hand_aux/train` loss must use `self.hand_aux_loss_weight > 0.0 and self.hand_proj is not None` as the guard condition.
6. The combined depth gate attn_mask shape for Design 003 is `(B*num_heads, 22, 960) = (32, 22, 960)` — different from Design 001's `(32, 70, 960)`. The expansion in `_DecoderLayer.forward()` uses `Nq = q.shape[1]` dynamically, so this is handled automatically.
