# Design Review — idea023/design002

**Verdict: APPROVED**

---

## Summary

Design 002 adds the same heatmap-pooled query warm-start as design001, replacing hard one-hot cross-entropy with KL divergence against a Gaussian heatmap target (σ=2 grid cells) at weight λ=0.2. A new module-level helper `_build_gaussian_heatmap_target` is added to `pose3d_transformer_head.py`. All three target files are specified. Instructions are explicit and complete.

---

## Checklist

### Feasibility
- All operations are standard PyTorch. Gaussian construction is a vectorised `torch.exp` over `(J, H'W')` — no loops, numerically safe.
- `F.log_softmax` + soft-target cross-entropy `-(gt_hm * log_probs).sum(dim=-1).mean()` is the correct KL divergence formulation (constant entropy of target dropped).
- Gaussian normalisation uses `.clamp(min=1e-6)` to prevent div-by-zero when joints are far outside the grid.

### Completeness and Explicitness

**`pelvis_utils.py`:**
- Identical to design001 — fully specified.

**`pose3d_transformer_head.py`:**
- Imports identical to design001 final combined block: `_compute_mpjpe_abs`, `_project_joints_to_grid_coords`, `_recover_pelvis_3d`. Complete.
- `_build_gaussian_heatmap_target` placement is specified: "before `_DecoderLayer` class definition." Signature, docstring, and full implementation provided. Uses `indexing='ij'` — correct for H-major order matching `feat.flatten(2).transpose(1,2)`.
- All `__init__` kwargs and `self.*` assignments identical to design001 — complete.
- `heatmap_proj` zero-init — specified.
- `forward()` block is identical to design001 (scalar temperature, `heatmap_learnable_temp=False`) — correct and fully specified.
- `loss()` uses `_build_gaussian_heatmap_target` for soft target; KL loss: `-(gt_hm * log_probs).sum(dim=-1).mean()` — sum over spatial, mean over 22 joints, then divide by batch. Reduction is explicit.
- Side-channel `_heatmap_logits` cleared after loss and in else branch — specified.

**`config.py`:**
- Complete head dict shown. `heatmap_target='gaussian'`, `heatmap_sigma=2.0`, `heatmap_loss_weight=0.2`. All values are literals. MMEngine constraint satisfied.

### Invariant Compliance
- `persistent_workers=False` — not touched.
- Body-only joint loss — unchanged.
- No invariant files modified.
- `predict()` side-channel safety — same as design001.

### Issues / Notes

**Minor (non-blocking):**
1. The design does not explicitly state the `B` variable availability in `forward()` — but since the code block starts "after `spatial = spatial + pos_enc`" and `B` is extracted from `feat.shape` earlier in baseline's `forward()`, this is unambiguous.

2. The `loss()` per-sample GT absolute joint construction (`gt_abs_joints = gt_joints[i, :22] + gt_pelvis_3d`) assumes `gt_pelvis_3d` has shape `(1, 3)` from `_recover_pelvis_3d(...[i:i+1], ...)`. Broadcasting `(22, 3) + (1, 3)` → `(22, 3)` is correct.

3. `_build_gaussian_heatmap_target` receives `grid_coords` which may have out-of-bounds values. The Gaussian naturally handles this (decays to near-zero values), and normalisation ensures a valid probability distribution even for OOB joints. No issue.

---

## Conclusion

All design details are fully specified. The Builder can implement this without guessing. No invariant violations. **APPROVED.**
