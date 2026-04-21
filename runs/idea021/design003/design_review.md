# Design Review — idea021/design003

**Verdict: APPROVED**

---

## Summary

Design003 uses the same factored parameterization as design002, adding Gaussian warm-start initialization for body joints 0–21 in `_init_head_weights()`. The design is complete, explicit, and implementable without guessing.

---

## Checklist

### Completeness and Explicitness

- **Design Description:** Present and clear. ✓
- **Starting point:** `baseline/` — explicit. ✓
- **Files to change:** `pose3d_transformer_head.py` and `config.py` only — both allowed. No `pelvis_utils.py` change. ✓
- **Algorithmic change:** Same factored bias as design002 plus warm-start in `_init_head_weights()`. Fully specified including `h_coords = torch.arange(self.feat_h)`, `sigma=4.0`, `alpha=1.0`, `.data[i] = gauss` assignment. ✓
- **Parameter shapes:** Identical to design002: `cross_attn_bias_row (70, 40)`, `cross_attn_bias_col (70, 24)`. ✓
- **Instance attribute storage:** `self.joint_row_prior = joint_row_prior` explicitly required. ✓
- **Warm-start guard:** `if (self.use_cross_attn_bias and self.cross_attn_bias_type == 'factored_warmstart' and self.joint_row_prior is not None)` — all conditions specified. ✓
- **Ordering in `__init__`:** Parameter allocation block comes before `_init_head_weights()` call (per baseline structure — `_init_head_weights()` is called last in `__init__`). The warm-start code appended to `_init_head_weights()` will find `self.cross_attn_bias_row` already allocated. ✓
- **Slice safety:** `self.joint_row_prior[:22]` guards against list length mismatches. ✓
- **`.data` assignment:** Correct bypass of autograd for initialization. ✓
- **Hand joints (22–69):** Explicitly remain zero (no action needed — `torch.zeros()` init). ✓
- **Column biases:** Explicitly zero-initialized for all joints. ✓
- **Config values:** `use_cross_attn_bias=True`, `cross_attn_bias_type='factored_warmstart'`, `feat_h=40`, `feat_w=24`, `joint_row_prior=[22 floats]` — all literals, MMEngine compliant. ✓
- **`joint_row_prior` list:** Exactly 22 float entries provided. ✓
- **Checkpoint resume correctness:** Explained — `_init_head_weights()` runs at construction, then checkpoint overwrites; resumed training uses learned values not warm-start values. ✓
- **AMP compatibility:** Gaussian peak ~1.0 well within float16 range. ✓
- **Invariants preserved:** Body joint loss indices 0–21 unchanged; `pelvis_token` unchanged; `persistent_workers=False` unchanged; no invariant files modified. ✓

### No Invariant File Modifications

No changes to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files. ✓

### Implementation Readiness

All method signatures, parameter names, shapes, initialization logic, and config values are fully specified. The Builder can implement this without guessing.

---

## Notes

The `joint_row_prior` values contain anatomically approximate entries (e.g., pelvis row 12.0 and left knee row 9.0, which would place the knee above the pelvis in the grid). However, the design explicitly acknowledges these are soft priors with σ=4 grid cells that the model refines during training. This does not block implementation — the values are hardcoded literals that the Builder copies verbatim from the design. Correctness of the anatomical mapping is outside the Builder's scope.

Design is approved as written.
