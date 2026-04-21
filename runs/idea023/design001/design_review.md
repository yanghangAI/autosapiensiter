# Design Review — idea023/design001

**Verdict: APPROVED**

---

## Summary

Design 001 adds a zero-init `Linear(256, 22)` heatmap projector to produce per-joint soft attention weights over the 40×24 spatial token grid, soft-pools per-joint feature vectors added to body query embeddings (0–21) before the decoder, and supervises with hard one-hot cross-entropy at weight λ=0.1. All three target files are specified. Implementation instructions are explicit and complete.

---

## Checklist

### Feasibility
- All operations (Linear, softmax, bmm, F.cross_entropy) are standard PyTorch. No new dependencies.
- Memory and speed overhead are negligible (< 1 MB, < 0.5 ms per step).
- Zero-init on `heatmap_proj` ensures stable training start identical to idea003/design001.

### Completeness and Explicitness

**`pelvis_utils.py`:**
- `project_joints_to_grid_coords` is fully specified with exact signature, docstring, projection convention (BEDLAM2: `u = fx*(-Y/X) + cx`, `v = fy*(-Z/X) + cy`), and return shape `(J, 2)`.
- Clamp on `X` (min=0.01) prevents div-by-zero. Confirmed torch/np already imported.

**`pose3d_transformer_head.py`:**
- Import changes are precise: exact lines to replace are specified, `recover_pelvis_3d` import explicitly required in step 2d note.
- All new `__init__` kwargs listed with types and defaults.
- All new `self.*` attribute assignments specified.
- `heatmap_proj` zero-init explicitly required.
- `forward()` insertion point is unambiguous: "after `spatial = spatial + pos_enc`, before query broadcast."
- Zero-pad pattern (`torch.zeros(B, self.num_joints - 22, self.hidden_dim, ...)`) avoids the expand-copy issue — uses `num_joints - 22` not hardcoded 48 (invariant 10 confirmed).
- The original query broadcast line removal is explicitly called out.
- `loss()` insertion point is precise: "after `losses['loss/uv/train'] = ...`, before `with torch.no_grad():`."
- One-hot indexing: `h_idx * self.feat_w + w_idx` flat index is correct for H-major order.
- `F.cross_entropy(logits_i, target_idx)` shape check: `logits_i = (22, 960)`, `target_idx = (22,)` — correct for per-joint multi-class CE.
- Side-channel `_heatmap_logits` set to `None` after loss and in the else branch — both cases specified.

**`config.py`:**
- Complete head dict shown. All new kwargs are bool/int/float/str literals. No Python import statements. MMEngine constraint satisfied.

### Invariant Compliance
- `persistent_workers=False` — not touched.
- Joint loss restricted to body indices 0–21 — unchanged.
- Evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, `train.py` wrapper — none mentioned or modified.
- `predict()` does not access `_heatmap_logits` — side-channel is harmless; `predict()` calls `forward()` but never reads the loss attribute.

### Issues / Notes

**Minor (non-blocking):**
1. The design note in step 2d says `recover_pelvis_3d` should be imported "via `from pelvis_utils import recover_pelvis_3d`" and suggests using `_recover_pelvis_3d(...)`. The final combined import block at the bottom of step 2d correctly adds `recover_pelvis_3d as _recover_pelvis_3d`. However, the import block in step 2a (the initial replacement block) does not yet include `recover_pelvis_3d`. The Builder must use the final combined import block from step 2d (which supersedes step 2a). This is clearly stated in the design; no ambiguity for a careful reader.

2. The design uses `B` in the `forward()` insertion (`torch.zeros(B, self.num_joints - 22, ...)`) where `B` is already defined from `B, C, H, W = feat.shape`. Consistent.

3. The `loss()` code uses `recover_pelvis_3d` directly as `recover_pelvis_3d(...)` in the inline text of step 2d but uses `_recover_pelvis_3d(...)` in the import alias. The Builder must consistently use the `_recover_pelvis_3d` alias throughout.

None of these require guessing; the design is unambiguous.

---

## Conclusion

All design details are fully specified. The Builder can implement this without guessing. No invariant violations. **APPROVED.**
