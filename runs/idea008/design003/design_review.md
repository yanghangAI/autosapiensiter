# Design Review — idea008 / design003

**Verdict: APPROVED**

---

## Checklist

### Feasibility
PASS. The 2-layer MLP `nn.Sequential(Linear(5632,256), GELU, Linear(256,144))` is within 1080 Ti memory budget (~1.47M parameters, explicitly verified in the Parameter Budget section). No new loss modules, no new data requirements, no changes to the decoder.

### Completeness
PASS. All required sections are present: Design Description, Starting Point, Files to Change (with line-level precision), Parameter Budget, Constraints and Invariants, Expected Behavior, Risk Notes.

### Explicitness
PASS. Every detail is nailed down:
- New kwargs: `num_body_queries: int = 22` and `hand_aux_loss_weight: float = 0.3`, placed after `dropout`, stored explicitly.
- MLP definition using `nn.Sequential` with exact layer types, dimensions computed dynamically from kwargs, and `nn.GELU()` activation (not ReLU — explicitly called out in constraints).
- Bottleneck dimension is `hidden_dim` (256) — reuses existing kwarg, not a new hyperparameter.
- Weight init loop: `for layer in self.hand_proj: if isinstance(layer, nn.Linear):` — this correctly skips the `nn.GELU()` element inside the Sequential and initializes only the two Linear layers.
- `forward()`: identical pattern to Design 002 — `body_flat`, `self.hand_proj(body_flat).reshape(B, num_hand, 3)`, `torch.cat` — all explicit.
- `loss()`: auxiliary loss key `'loss/hand_aux/train'`, `_HAND = list(range(22, 70))`, weight `self.hand_aux_loss_weight = 0.3` — fully specified.
- Full constructor signature provided verbatim with `hand_aux_loss_weight: float = 0.3`.
- `config.py` snippet provided with `hand_aux_loss_weight=0.3` as a float literal.
- `pelvis_utils.py` — no changes, explicitly stated.

### Implementation Readiness
PASS. The Builder can implement this without guessing. The only difference from Design 002 is `nn.Sequential` instead of a bare `nn.Linear`, the init loop, and the loss weight of 0.3. All three differences are fully specified.

### Invariant Compliance
PASS.
- `persistent_workers=False` — not touched.
- `self.num_joints = 70` — explicitly preserved.
- `_BODY = list(range(0, 22))` — unchanged.
- Auxiliary loss reuses existing `self.loss_joints_module` — no new loss module created.
- `_DecoderLayer` — not changed.
- Backbone, metric, dataset, transforms, data preprocessor, infra files, train.py — none touched.
- MMEngine config constraint: `num_body_queries=22` (int literal), `hand_aux_loss_weight=0.3` (float literal). No imports required.

### Cross-Design Consistency
PASS. Design 003 is the richest variant in the idea: same 22-query decoder as Designs 001 and 002, hand recovery upgraded from single Linear to 2-layer MLP, auxiliary loss weight stepped up from 0.1 to 0.3. The progression is coherent and internally consistent across all three designs.

---

## Issues Found
None.
