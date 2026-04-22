# Design Review — idea028/design001

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- The `_PelvisCrossAttnDecoder` class is fully specified with exact `__init__` and `forward` signatures, tensor shapes, and residual connection pattern. A Builder can implement it without guessing.
- `nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)` is valid; `256 % 8 == 0`.
- The conditional block in `__init__`, `_init_head_weights`, and `forward` are all explicitly spelled out. No ambiguity about control flow.

### Completeness
- Starting point: `baseline/` — specified.
- Files changed: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` unchanged. All within the allowed experimentable set.
- New module placement: "at module level, after `_DecoderLayer`" — unambiguous.
- New kwargs: all four specified with exact names, types, and defaults.
- Instance attribute storage: listed explicitly (`self.use_decoupled_pelvis`, `self.pelvis_hidden_dim`, `self.num_body_queries`).
- `forward()` change: exact before/after code given, with variable names and shapes annotated.
- `_init_head_weights()` change: exact addition given; existing loop covers `depth_out` and `uv_out` re-initialization, confirmed by reviewer inspection of baseline.
- `loss()` and `predict()`: explicitly unchanged.
- `config.py`: full head dict given with all four new kwargs as bool/int literals. No Python imports. Compliant with MMEngine constraint.

### Explicitness
- Constraint 5 notes `pelvis_hidden_dim` must equal `hidden_dim` for `depth_out`/`uv_out` shape compatibility — Builder is explicitly warned.
- Constraint 6 notes existing `_init_head_weights` loop covers `depth_out` and `uv_out` — confirmed against baseline (the loop `for m in [self.joints_out, self.depth_out, self.uv_out]` is present).
- Index ordering (0 = depth, 1 = UV) stated twice (constraint 4 and forward code) — unambiguous.
- `num_body_queries=70` for this design — joint embedding size is unchanged.

### Invariants
- No modification to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or `train.py`.
- `persistent_workers=False` preserved.
- `self.num_joints = 70` preserved.
- Seed 2026, batch 4, accum 8 preserved.

### Minor Notes (non-blocking)
- The design does not store `pelvis_num_heads` as a separate instance attribute (only `pelvis_hidden_dim` and `num_body_queries` are stored). This is fine because `pelvis_num_heads` is consumed solely during `__init__` construction of `_PelvisCrossAttnDecoder` and is not needed in `forward()`. Builder should note this is intentional.
- Design references `dropout` when constructing `_PelvisCrossAttnDecoder` — the `dropout` local variable is available in `__init__` from the parameter. No issue.

All required details are present and explicit. A Builder can implement this without any guessing.
