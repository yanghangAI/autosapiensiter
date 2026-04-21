# Design Review — idea025/design002

**Verdict: APPROVED**

---

## Checklist

### Completeness and Explicitness

- [x] **Design Description** present and clear: distal-limb-focused bilateral symmetry loss, λ=0.5, 8 verified pairs with per-pair weights.
- [x] **Starting point** explicitly stated: `baseline/`.
- [x] **Files to change** explicitly listed: `pose3d_transformer_head.py` and `config.py` only.
- [x] **Algorithm** fully specified: same as Design 001 plus static per-pair weight tensor broadcast as `(1, P, 1)` over `(B, P, 3)`.
- [x] **Per-pair weights** fully specified: 8-element list `[0.5, 1.0, 2.0, 2.0, 0.5, 1.0, 1.5, 2.0]`, one entry per pair in matching order. Length (8) equals number of pairs (8).
- [x] **Weight rationale** documented per pair: hip=0.5, knee=1.0, ankle=2.0, foot=2.0, collar=0.5, shoulder=1.0, elbow=1.5, wrist=2.0.
- [x] **Exact constructor kwargs** specified: same six kwargs as Design 001; `sym_pair_weights` is the differentiating addition.
- [x] **`__init__` insertion point** explicit: after `self.loss_weight_uv = loss_weight_uv`.
- [x] **`loss()` insertion point** explicit: after `losses['loss/uv/train']`, before `with torch.no_grad():`.
- [x] **Config values** fully literal: MMEngine-compliant, no Python imports.
- [x] **`sym_pair_weights_buf` registration**: registered as persistent buffer (`register_buffer`), not plain attribute — device-safe.
- [x] **`sym_adaptive_weight` and `sym_tau` omitted from config**: defaults (`False`, `0.1`) are correct for Design 002.
- [x] **Constraint**: `len(sym_pairs) == len(sym_pair_weights)` (both 8) — explicitly called out.

### Joint Index Verification

Same 8 verified pairs as Design 001 — confirmed against `bedlam2_transforms.py::_FLIP_PAIRS`. Correct.

### Mirror Convention

`sym_mirror_axis=1` — same as Design 001, correct.

### Deviation from idea.md

idea.md assumed 10 pairs and listed 10-element pair weights. Design 002 correctly re-maps to 8 pairs and 8 weights preserving the overall distal-upweighting intent. The deviation is justified and complete.

### Invariant Compliance

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, infra files, or training infrastructure. Compliant.

### No Ambiguities for Builder

All code snippets for `__init__` and `loss()` are complete and exact (identical to Design 001 plus the `sym_pair_weights_buf` branch). Config kwargs are fully specified. Builder can implement without guessing.
