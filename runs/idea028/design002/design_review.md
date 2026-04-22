# Design Review — idea028/design002

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- Identical to design001 in all implementation respects except `pelvis_num_heads=4`. The design explicitly states this is the only runtime difference.
- `nn.MultiheadAttention(256, 4, ...)`: `256 % 4 == 0` — confirmed valid.
- Full code listings for `_PelvisCrossAttnDecoder`, `__init__`, `_init_head_weights`, and `forward()` are provided verbatim.

### Completeness
- Starting point: `baseline/` — specified.
- Files changed: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` unchanged.
- The single differentiating config change (`pelvis_num_heads=4`) is highlighted explicitly alongside a complete head dict.
- All other config values stated to be identical to baseline.
- `loss()` and `predict()`: explicitly unchanged.

### Explicitness
- Constraint 4 explicitly checks `256 % 4 == 0` and states the Builder must verify this. Constraint is satisfied and confirmed by the reviewer.
- Constraint 6: `pelvis_hidden_dim=256` must equal `hidden_dim=256` — same as design001, confirmed valid.
- Index ordering (0 = depth, 1 = UV) retained consistently from design001.
- `num_body_queries=70` for this design — joint embedding unchanged.
- Per-head dimension math provided: 64-dimensional projections (256/4) vs 32 in design001.

### Invariants
- No modification to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or `train.py`.
- `persistent_workers=False`, `self.num_joints = 70`, seed 2026, batch 4, accum 8 all preserved.

### Minor Notes (non-blocking)
- Same observation as design001: `pelvis_num_heads` is not stored as an instance attribute, which is correct since it is only used at construction time.

All required details are present and explicit. Trivially implementable given design001.
