# Design Review — idea025/design001

**Verdict: APPROVED**

---

## Checklist

### Completeness and Explicitness

- [x] **Design Description** present and clear: uniform bilateral symmetry loss, λ=0.3, 8 verified pairs.
- [x] **Starting point** explicitly stated: `baseline/`.
- [x] **Files to change** explicitly listed: `pose3d_transformer_head.py` and `config.py` only.
- [x] **Algorithm** fully specified: asymmetry vector computation, mirror convention (Y-axis, index 1), smooth-L1 (beta=0.05), mean reduction, `bilateral_sym_loss_weight` scaling, loss key `'loss/sym/train'`.
- [x] **Exact constructor kwargs** specified with types and defaults: all six kwargs listed, defaults reproduce baseline when `bilateral_sym_loss_weight=0.0`.
- [x] **`__init__` insertion point** explicit: after `self.loss_weight_uv = loss_weight_uv`.
- [x] **`loss()` insertion point** explicit: after `losses['loss/uv/train']`, before `with torch.no_grad():`.
- [x] **Config values** fully literal (int/float/list-of-lists): MMEngine-compliant, no Python imports.
- [x] **`gt_joints` shape** addressed: confirmed full `(B, 70, 3)`; indices 1–21 valid.
- [x] **Buffer registration** specified: `sym_pairs_buf` registered via `register_buffer`; `sym_pair_weights_buf = None` for Design 001.
- [x] **Device handling** addressed: `mirror` tensor constructed on `pred['joints'].device`.
- [x] **AMP compatibility** noted: `torch.where` safe with float16.

### Joint Index Verification

The 8 symmetric pairs `[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]` are confirmed correct against `bedlam2_transforms.py::_FLIP_PAIRS`:
```
(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)
```
The correction from idea.md's assumed 10 pairs (which wrongly included eyes/heels outside body range 0–21) is justified and correct.

### Mirror Convention

`sym_mirror_axis=1` (negate Y) matches BEDLAM2 convention X=forward, Y=left, Z=up, confirmed by `pelvis_utils.py` and `bedlam2_transforms.py`.

### Invariant Compliance

- No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, infra files, or training infrastructure. Compliant.

### No Ambiguities for Builder

The design is self-contained. All code snippets are exact and complete. Builder can implement without guessing.
