# Design Review — idea028/design003

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- Combines design001 (decoupled pelvis decoder) with idea008/design001 (22-query body-only joint decoder). Both mechanisms are individually validated. The interaction between them is correctly described: they operate on separate query sets with no cross-dependencies.
- All code snippets are explicit and complete. No guessing required.

### Completeness
- Starting point: `baseline/` — specified.
- Files changed: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` unchanged.
- New kwargs: all four specified with exact names, types, defaults.
- `joint_queries` embedding change from `nn.Embedding(num_joints, hidden_dim)` to `nn.Embedding(num_body_queries, hidden_dim)` is explicitly stated with before/after.
- Zero-padding logic is fully specified: `torch.zeros(B, self.num_joints - self.num_body_queries, 3, device=..., dtype=...)` then `torch.cat([body_joints, pad], dim=1)`. Arithmetic: 70 - 22 = 48 padding joints confirmed.
- Forward shape annotation: `(B, 22, hidden_dim)` → decoded → `(B, 22, 3)` body joints → cat pad → `(B, 70, 3)`.
- Pelvis decoder path identical to design001 (`use_decoupled_pelvis=True`).
- `_init_head_weights()` addition: same as design001; `trunc_normal_` on `pelvis_coord_queries.weight`. Note that `self.joint_queries.weight` initialization in the existing baseline loop still applies, now to shape `(22, hidden_dim)` — the design explicitly calls this out as handled correctly.
- `loss()`: unchanged; `_BODY = list(range(0, 22))` covers all 22 active body output joints. Zero-padded indices 22–69 are never referenced in the loss.
- `predict()`: unchanged; `self.num_joints = 70` still correct.
- `config.py`: full head dict given with `num_body_queries=22`, `use_decoupled_pelvis=True`, all as literals.

### Explicitness
- Constraint 3 explicitly separates `self.num_body_queries = 22` from `self.num_joints = 70` — both must coexist as instance attributes. This is clearly stated.
- Constraint 5 specifies the exact zero-pad formula using `self.num_joints - self.num_body_queries`. No hardcoded magic numbers in the spec.
- Constraint 13: "pelvis token `decoded[:, 0, :]` is NOT read for pelvis output in this design" — Builder is explicitly warned against using the baseline path.
- `requires_grad` behavior for `torch.zeros` is called out (defaults to False) — no special detach needed. Accurate.
- `pelvis_hidden_dim=256` must equal `hidden_dim=256` — same constraint as design001/002, confirmed valid.

### Invariants
- No modification to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or `train.py`.
- `persistent_workers=False`, seed 2026, batch 4, accum 8 preserved.
- `self.num_joints = 70` preserved for `predict()`.

### Interaction Between the Two Mechanisms
- The 22-query path and the decoupled pelvis path are orthogonal: they share only the `spatial` tokens. No shared state or index conflicts.
- Body joint loss covers exactly indices 0–21 of `joints (B, 70, 3)`, which corresponds to the 22 body joint outputs. Zero-padded indices 22–69 carry no gradient.
- Pelvis losses flow only into `pelvis_coord_queries` and `pelvis_decoder` — no interaction with the joint decoder.

All required details are present and explicit. The design is unambiguously implementable.
