# Design Review — idea008 / design002

**Verdict: APPROVED**

---

## Checklist

### Feasibility
PASS. Adding `nn.Linear(5632, 144)` as `self.hand_proj` and computing hand joints via a linear projection is straightforward. The auxiliary loss reuses the existing `loss_joints_module` instance; no new loss module is required.

### Completeness
PASS. All required sections are present: Design Description, Starting Point, Files to Change (with line-level precision), Parameter Budget, Constraints and Invariants, Expected Behavior.

### Explicitness
PASS. Every detail is nailed down:
- New kwargs: `num_body_queries: int = 22` and `hand_aux_loss_weight: float = 0.1`, placed after `dropout`, stored explicitly.
- `self.hand_proj = nn.Linear(num_body_queries * hidden_dim, (num_joints - num_body_queries) * 3)` — computed dynamically from kwargs, not hard-coded.
- Weight init: `nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)` and `nn.init.zeros_(self.hand_proj.bias)` — exact calls given.
- `forward()`: `body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)` — uses `self.hidden_dim` which is stored in baseline; `hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)`; concatenation via `torch.cat` — all explicit.
- `loss()`: auxiliary loss key `'loss/hand_aux/train'`, `_HAND = list(range(22, 70))`, weight applied as `self.hand_aux_loss_weight * self.loss_joints_module(...)` — fully specified.
- Full constructor signature provided verbatim.
- `config.py` snippet provided with `num_body_queries=22` and `hand_aux_loss_weight=0.1` as literals.
- `pelvis_utils.py` — no changes, explicitly stated.

### Implementation Readiness
PASS. The Builder can implement this without guessing. Variable names, tensor shapes, init calls, loss key names, and config values are all explicit.

### Invariant Compliance
PASS.
- `persistent_workers=False` — not touched.
- `self.num_joints = 70` — explicitly preserved.
- `_BODY = list(range(0, 22))` — unchanged.
- Auxiliary loss reuses existing `self.loss_joints_module` — no new loss module created.
- `_DecoderLayer` — not changed.
- Backbone, metric, dataset, transforms, data preprocessor, infra files, train.py — none touched.
- MMEngine config constraint: both new values are literals (int and float). No imports required.

### Cross-Design Consistency
PASS. Design 002 builds directly on Design 001's 22-query decoder; the only additions are `hand_proj` and the auxiliary loss. The kwarg `hand_aux_loss_weight` defaults to 0.1, consistent with the idea spec.

---

## Issues Found
None.
