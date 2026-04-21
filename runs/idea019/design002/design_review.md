# Design Review — idea019/design002

**Verdict: APPROVED**

**Reviewed:** 2026-04-21

---

## Summary

Design 002 specifies deformable sparse cross-attention (K_s=8) with a 22-query body-only decoder and linear hand recovery via `Linear(22*256, 48*3)`. It composes two independently-validated improvements (idea008 22-query body decoder + idea019 deformable sampling) and provides a complete, unambiguous implementation spec.

---

## Checklist

### Design Description
Present and accurate. ✓

### Starting Point
`baseline/` — explicitly stated. ✓

### Files to Change
- `pose3d_transformer_head.py` — fully specified. ✓
- `config.py` — fully specified. ✓
- `pelvis_utils.py` — no changes (stated explicitly). ✓
- No invariant files touched. ✓

### Algorithmic Changes

**`_DeformableDecoderLayer`**: Defers to Design 001 spec; no changes needed to the class itself. ✓

**`Pose3dTransformerHead.__init__` changes**:
- All new kwargs with types and defaults (same set as Design 001). ✓
- `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` = `nn.Embedding(22, 256)`. ✓
- `has_hand_proj` guard with `self.hand_proj = nn.Linear(num_body_queries * hidden_dim, (num_joints - num_body_queries) * 3)` = `Linear(5632, 144)`. ✓
- Input/output dimensions computed dynamically from kwargs (not hardcoded). ✓
- `decoder_layers = nn.ModuleList(...)` — same structure as Design 001. ✓
- `decoder_layer = decoder_layers[0]` alias. ✓
- All instance attributes stored. ✓

**`_init_head_weights`** changes:
- Deformable init block (same as Design 001). ✓
- `hand_proj` init: `trunc_normal_(std=0.02)` for weight, `zeros_` for bias. ✓
- Guarded by `has_hand_proj`. ✓

**`forward()`** changes:
- Full implementation provided. ✓
- 22-query deformable path: `decoded` shape `(B, 22, 256)`. ✓
- Body joints: `self.joints_out(decoded)` → `(B, 22, 3)`. ✓
- Hand recovery: `decoded.reshape(B, 22*256)` → `hand_proj` → `reshape(B, 48, 3)`. ✓
- `torch.cat([body_joints, hand_joints], dim=1)` → `(B, 70, 3)`. ✓
- `has_hand_proj` guard around hand recovery path. ✓
- `pelvis_token = decoded[:, 0, :]` — query 0 of 22 body queries. ✓
- Output dict shape `{'joints': (B,70,3), 'pelvis_depth': (B,1), 'pelvis_uv': (B,2)}`. ✓

**`loss()`** changes:
- Auxiliary hand loss added after existing body/depth/UV losses. ✓
- Key: `'loss/hand_aux/train'`. ✓
- `_HAND = list(range(self.num_body_queries, self.num_joints))` = `range(22, 70)`. ✓
- Reuses `self.loss_joints_module` (no new loss module). ✓
- Guarded by `hand_aux_loss_weight > 0.0 and has_hand_proj`. ✓

### Config Values
All int/float/str literals. No Python import statements. ✓
- `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0`. ✓
- `deform_num_points=8`, `deform_hidden_dim=64`, `num_decoder_layers=1`. ✓
- `in_channels=1024` hardcoded literal. ✓
- All other config (optimizer, LR, hooks) identical to baseline. ✓

### Invariants Preserved
- Output `joints` shape `(B, 70, 3)` via cat. ✓
- `self.num_joints = 70` preserved. ✓
- `pelvis_token = decoded[:, 0, :]`. ✓
- Body joint loss restricted to indices 0-21. ✓
- `persistent_workers=False`. ✓
- Backbone, metric, transforms, data preprocessor, infra files untouched. ✓

### No Issues Found

The design is complete. A Builder can implement it without guessing.
