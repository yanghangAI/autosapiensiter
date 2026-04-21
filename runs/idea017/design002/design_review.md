# Design Review — idea017/design002

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- Design 002 is a strict superset of Design 001: the only runtime difference is `aux_body_loss_weight=0.4` (was `0.0`), which activates the intermediate body loss branch already specified in Design 001.
- The intermediate loss branch iterates over `self._intermediate_outputs[:-1]`, which for `num_decoder_layers=2` yields exactly 1 element (layer-1 output). The loss key `loss/joints_aux_0/train` at weight 0.4 is correct.
- Gradient flow described (from `loss/joints_aux_0/train` → `joints_out` → layer-1 → `joint_queries` → backbone) matches the idea001/design002 pattern known to train stably.
- No new architectural components introduced. All new code paths are already present in the Design 001 spec.

### Completeness
- Starting point: `baseline/` — specified.
- Files to change: `pose3d_transformer_head.py` and `config.py` — both fully specified. `pelvis_utils.py` explicitly unchanged.
- Full constructor signature reproduced (identical to Design 001).
- `__init__` body: identical to Design 001 with explicit note that `aux_body_loss_weight=0.4` activates the intermediate loss branch.
- `_init_head_weights()`: identical to Design 001.
- `forward()`: identical to Design 001 (full code reproduced).
- `loss()`: the intermediate body loss block is now active; exact code given with correct loss key `loss/joints_aux_0/train`.
- `config.py` head dict: `aux_body_loss_weight=0.4` is explicit. All other values are literals. No Python import statements.
- Additional constraints 13–16 beyond Design 001 are documented.

### Explicitness
- Constraint 13 is critical and correctly stated: `self._intermediate_outputs[:-1]` (not `self._intermediate_outputs`) must be used so the final layer output is not double-counted. For `num_decoder_layers=2` this yields exactly 1 element.
- Constraint 14 (shared `joints_out`) is intentional and documented.
- Constraint 15 (no key collision with baseline losses) is verified: `loss/joints_aux_0/train` is a new key.
- Constraint 16 (weight 0.4 as float literal in config) is explicit.
- Expected losses logged: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train`, `loss/joints_aux_0/train` — fully enumerated.

### Invariant Compliance
- Same as Design 001: no changes to invariant files, `pelvis_utils.py` unchanged, loss restricted to body joints 0-21 for main body loss, `persistent_workers=False` preserved, no Python imports in config.

### Issues / Notes
- None. Design 002 is well-specified. The sole code difference from Design 001 is `aux_body_loss_weight=0.4` in the config, and the design makes this clear. The intermediate loss code path was already fully specified in Design 001 and is merely activated here.
