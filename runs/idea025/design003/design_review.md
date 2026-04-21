# Design Review — idea025/design003

**Verdict: APPROVED**

---

## Checklist

### Completeness and Explicitness

- [x] **Design Description** present and clear: adaptive GT-magnitude weighting, λ=0.5, 8 verified pairs, no static per-pair weights, `sym_tau=0.1 m`.
- [x] **Starting point** explicitly stated: `baseline/`.
- [x] **Files to change** explicitly listed: `pose3d_transformer_head.py` and `config.py` only.
- [x] **Algorithm** fully specified: asymmetry vectors computed as in Design 001; adaptive weight `asym_w = 1 / (1 + ||asym_gt||_2 / tau)` computed per (B, P) under `torch.no_grad()` with `.detach()`; broadcast `(B, P, 1)` over `sym_loss (B, P, 3)`.
- [x] **Adaptive weight range** documented: strictly in (0, 1]; no division-by-zero since denominator ≥ 1.0.
- [x] **`torch.no_grad()` scope** explicit: both `asym_gt_mag` and `asym_w` computed inside the context — no gradient through weighting.
- [x] **`sym_tau` units** documented: metres, matching root-relative joint coordinate space (post-SubtractRootJoint); 0.1 m = 100 mm is appropriate.
- [x] **Exact constructor kwargs** specified: same six kwargs; `sym_adaptive_weight=True`, `sym_tau=0.1` are the differentiating values.
- [x] **`__init__` insertion point** explicit: after `self.loss_weight_uv = loss_weight_uv`.
- [x] **`loss()` insertion point** explicit: after `losses['loss/uv/train']`, before `with torch.no_grad():`.
- [x] **Config values** fully literal: `sym_adaptive_weight=True` (bool), `sym_tau=0.1` (float), no Python imports. MMEngine-compliant.
- [x] **`sym_pair_weights` omitted from config**: default `None` is correct — no static pair weights for Design 003.
- [x] **`sym_pair_weights_buf = None`** explicitly noted — `if self.sym_pair_weights_buf is not None:` branch skipped.
- [x] **AMP note** provided: optional `.float()` cast for `asym_gt_mag` if float16 GT tensors. Advisory only; not a blocking ambiguity since `gt_joints` is loaded as float32 in practice.

### Joint Index Verification

Same 8 verified pairs as Designs 001/002 — confirmed against `bedlam2_transforms.py::_FLIP_PAIRS`. Correct.

### Mirror Convention

`sym_mirror_axis=1` — same as Designs 001/002, correct.

### Invariant Compliance

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, infra files, or training infrastructure. Compliant.

### No Ambiguities for Builder

All code snippets for `__init__` and `loss()` are complete and exact. The adaptive weighting path is fully specified with shapes, scope, and broadcast semantics. Config kwargs are fully specified. Builder can implement without guessing.
