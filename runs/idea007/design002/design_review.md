# Design Review — idea007 / design002

**Verdict: APPROVED**

---

## Checklist

### Feasibility
PASS. The change is identical to design001 at the structural level, with the addition of an analytically computed warm-start prior. All computations use only `torch` operations (`torch.arange`, element-wise arithmetic, `torch.exp`, `nn.Parameter`) — no external imports needed. The `rounding_mode='floor'` correction for integer row indices is explicitly stated and correct.

### Completeness
PASS. All required sections are present:
- Design Description: stated clearly.
- Starting point: `baseline/` — confirmed.
- Files to change: `pose3d_transformer_head.py` and `config.py` only.
- Exact algorithmic changes: new `_DecoderLayer.__init__` signature with `cross_attn_bias_init: str = 'zero'/'band_prior'`, full prior computation block specified with exact constants.
- Exact config values: `num_spatial=960`, `cross_routing_type='band_prior'` as plain literals.
- Training/loss/data/inference changes: none.
- Constraints and edge cases: all 11 constraints listed explicitly.

### Explicitness
PASS. The Builder is given:
- Exact new `_DecoderLayer.__init__` signature with all default values.
- Exact prior computation: `LOWER_BODY_JOINTS`, `UPPER_BODY_JOINTS` hardcoded lists, spatial grid dimensions `H'=40, W'=24`, Gaussian centers `lower_center=30.0`, `upper_center=10.0`, `sigma=5.0`, scale `[−0.5, +0.5]`.
- Explicit correction: use `.div(_W_prime, rounding_mode='floor')` instead of `//` on float tensors.
- Exact code block for prior computation (verbatim Python).
- Forward pass identical to design001, with assert and `attn_mask` pass-through.
- `cross_routing_type` mapping dict (`'none'` → `'zero'`, `'zero_init'` → `'zero'`, `'band_prior'` → `'band_prior'`) specified explicitly.
- Decoder layer construction code with all kwargs.
- Full updated `Pose3dTransformerHead.__init__` signature.
- Complete config.py head dict with all loss args filled in.

### Implementation Readiness
PASS. No ambiguity requiring guessing:
- Row-index computation: `torch.arange(num_spatial, dtype=torch.float32).div(_W_prime, rounding_mode='floor')` — exact.
- Gaussian formula: `torch.exp(-0.5 * ((row_idx - center) / sigma) ** 2)` — exact.
- Bias scaling: `g - 0.5` gives range `~[−0.5, +0.5]` — explicit.
- Joint group index lists hardcoded as Python list literals in the code block.
- `init_bias` created as `torch.zeros(num_joints, num_spatial)`, then rows filled per group, then wrapped in `nn.Parameter` — explicit.
- `_init_head_weights` exclusion stated.
- `cross_routing_type='none'` backward-compatibility stated.

### Invariant Compliance
PASS. No invariant files touched. Config uses only string and integer literals — compliant with MMEngine no-imports rule.

### Cross-Design Consistency
PASS. design002 starts from `baseline/` independently and introduces the `cross_attn_bias_init` kwarg to `_DecoderLayer`. When `cross_attn_bias_init='zero'`, the behaviour is equivalent to design001 — this is consistent.

---

## Notes
- The `LOWER_BODY_JOINTS` and `UPPER_BODY_JOINTS` lists cover exactly indices 0–21 (22 body joints), with all 22 assigned to one group or the other and no overlaps. Builder should verify this during implementation.
- The design notes that `init_bias` is passed directly to `nn.Parameter(init_bias)`, which copies the data into the parameter — the temporary tensor does not need to be retained. This is correct.
- The `cross_routing_type='none'` mapping to `'zero'` ensures that if a user omits `cross_routing_type`, zero init is used, recovering baseline-identical behaviour — correct and safe default.
