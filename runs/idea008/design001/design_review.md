# Design Review — idea008 / design001

**Verdict: APPROVED**

---

## Checklist

### Feasibility
PASS. Reducing `nn.Embedding` from 70 to 22 queries and zero-padding the output is a minimal, low-risk change. No new modules, no new loss terms, no shape mismatches.

### Completeness
PASS. All required sections are present: Design Description, Starting Point, Files to Change (with line-level precision), Constraints and Invariants, Expected Behavior.

### Explicitness
PASS. Every detail is nailed down:
- `num_body_queries: int = 22` kwarg placement (after `dropout`), storage as `self.num_body_queries`.
- `nn.Embedding(num_body_queries, hidden_dim)` — exact replacement line.
- Zero-pad via `torch.zeros(B, self.num_joints - self.num_body_queries, 3, device=..., dtype=...)` — shape, device, dtype, and gradient behavior all explicit.
- `pelvis_token = decoded[:, 0, :]` — explicitly confirmed unchanged.
- `_init_head_weights` — explicitly confirmed no new modules to initialize, so no change needed.
- Full constructor signature provided verbatim.
- `config.py` snippet provided verbatim with `num_body_queries=22` as an integer literal.
- `pelvis_utils.py` — no changes, explicitly stated.

### Implementation Readiness
PASS. The Builder can implement this without guessing. All tensor shapes, variable names, device/dtype propagation, and config values are given explicitly.

### Invariant Compliance
PASS.
- `persistent_workers=False` — not touched.
- `self.num_joints = 70` — explicitly preserved.
- `_BODY = list(range(0, 22))` — not changed.
- `_DecoderLayer` — not changed.
- Backbone, metric, dataset, transforms, data preprocessor, infra files, train.py — none touched.
- MMEngine config constraint: `num_body_queries=22` is an integer literal, no import required.

### Cross-Design Consistency
PASS. Design 001 is the diagnostic baseline for the idea. Designs 002 and 003 build on it by adding hand recovery modules; Design 001 correctly uses zero-padding as the simplest diagnostic variant.

---

## Issues Found
None.
