# Design Review — idea006/design002

**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## Checklist

### Feasibility
PASS. The change builds on the same mechanism as design001 with an additional module-level helper function `_build_skeleton_attn_bias` that uses only `torch` (already imported). No new dependencies. No new imports.

### Completeness
PASS. All required details are specified:
- Starting point: `baseline/` — explicit.
- Files to change: `pose3d_transformer_head.py` and `config.py` — explicit.
- Exact algorithmic change: `_build_skeleton_attn_bias` helper with hardcoded edge list; `_DecoderLayer.__init__` updated with optional `attn_bias_init`; `_DecoderLayer.forward` updated with `attn_mask=self.attn_bias`; `Pose3dTransformerHead.__init__` updated with `attn_bias_type` param and conditional init logic.
- Exact parameter values: `adjacent_val=+0.5`, `pelvis_diag_val=-0.5`, all other entries `0.0`, shape `(70, 70)`, 4900 parameters.
- Config change: `attn_bias_type='skeleton_init'` added to head dict as string literal.
- All constraints and invariants specified.

### Explicitness
PASS. Exact code provided for every change:
- `_build_skeleton_attn_bias` fully written with all edge lists.
- Exact new `_DecoderLayer.__init__` signature and body additions.
- Exact `Pose3dTransformerHead.__init__` changes with before/after config dict.
- Builder has zero ambiguity.

### Index Range Verification
PASS. Left hand covers joints 22–44 (23 joints, max index 44 < 70). Right hand covers joints 45–67 (23 joints, max index 67 < 70). Jaw=68, head_top=69. Total = 22 + 23 + 23 + 2 = 70 joints. All indices within [0, 69].

### Edge Count Verification
PASS. body_edges=21, left_hand_edges=23 (4 chains of 4 + 3 metacarpal = 19... wait — 4 finger chains × 4 edges + 1 thumb chain × 4 edges + 3 metacarpal edges = 4×4 + 4 + 3 = 16+4+3 = 23), right_hand_edges=23 (mirror), face_edges=2. Total = 21+23+23+2 = 69. Matches table.

### Bidirectionality
PASS. The constraint explicitly requires both `bias[i,j]` and `bias[j,i]` set to `+0.5` for all edges, and the helper code does this correctly with lines `bias[i, j] = adjacent_val; bias[j, i] = adjacent_val`.

### `.float().clone()` Requirement
PASS. Constraint #6 explicitly requires `_DecoderLayer.__init__` to call `.float().clone()` on `attn_bias_init` before wrapping in `nn.Parameter`. The code in section 1a shows this: `self.attn_bias = nn.Parameter(attn_bias_init.float().clone())`.

### Invariant Compliance
PASS. Only `pose3d_transformer_head.py` and `config.py` change. `config.py` change is a string literal — no Python imports. No invariant files touched.

### Placement of Helper
PASS. Constraint #1 specifies `_build_skeleton_attn_bias` must be at module level, after imports and before `_DecoderLayer`. This is explicit.

---

## Notes
- The `attn_bias_type='none'` fallback path sets `_bias_init = None`, causing `_DecoderLayer` to fall back to `torch.zeros(num_joints, num_joints)` — matching baseline behaviour. This fallback is correctly specified and required for compatibility.
- Design 002 handles `'zero_init'` mode in `Pose3dTransformerHead` as well, producing `torch.zeros`. This is consistent and does not conflict with Design 001 (which is a separate independent code copy per design).
