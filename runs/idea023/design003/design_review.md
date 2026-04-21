# Design Review — idea023/design003

**Verdict: APPROVED**

---

## Summary

Design 003 extends design002 (Gaussian KL heatmap loss, λ=0.2) with a learnable per-joint softmax temperature `nn.Parameter(torch.ones(22))` passed through `F.softplus`, allowing each of the 22 body joints to learn its own attention sharpness. The heatmap loss operates on raw logits (before temperature), so the supervision target is independent of the attention sharpness. All three target files are specified. Instructions are explicit and complete.

---

## Checklist

### Feasibility
- `nn.Parameter(torch.ones(22))` with `F.softplus` is standard PyTorch. `F.softplus` maps ℝ → (0, +∞), ensuring positivity. AMP-safe (softplus does not overflow for typical values).
- Temperature shape `(1, 22, 1)` broadcasts correctly with logits `(B, 22, H'W')`.
- 22 extra parameters — negligible overhead.

### Completeness and Explicitness

**`pelvis_utils.py`:**
- Identical to design001/002 — fully specified.

**`pose3d_transformer_head.py`:**
- Imports: same combined block as design002 — complete.
- `_build_gaussian_heatmap_target`: identical to design002 — complete.
- `__init__` kwargs: identical parameter set to design001/002, with `heatmap_learnable_temp=True` in config. The `heatmap_temperature` scalar is still accepted as a kwarg (used when `heatmap_learnable_temp=False`); this is consistent and correct.
- `heatmap_temp` parameter initialization: `nn.Parameter(torch.ones(22))`. The design explicitly documents that `F.softplus(1.0) ≈ 1.313`, slightly above 1.0. The Builder is instructed to use `torch.ones(22)` for simplicity. Acceptable warm-start.
- `heatmap_proj` zero-init — specified.
- `forward()` block: the `if self.heatmap_learnable_temp:` branch uses `F.softplus(self.heatmap_temp).view(1, 22, 1)` — shape and broadcasting are explicitly documented. The `else` branch uses `self.heatmap_temperature` scalar. Both paths fully specified.
- `loss()` block: identical to design002 — operates on raw `_heatmap_logits[i]` before temperature. This is explicitly noted and justified (loss supervises raw logit space independently of attention sharpness). Correct.
- Side-channel `_heatmap_logits` cleared after loss and in else branch — specified.

**`config.py`:**
- Complete head dict shown. `heatmap_learnable_temp=True` is the only diff from design002. `heatmap_temperature=1.0` retained as a literal (unused at runtime when learnable temp is active, included for signature consistency). MMEngine constraint satisfied.

### Invariant Compliance
- `persistent_workers=False` — not touched.
- Body-only joint loss — unchanged.
- No invariant files modified.
- `predict()` side-channel safety — same as design001/002.

### Issues / Notes

**Minor (non-blocking):**
1. The design notes two options for the `heatmap_temp` initial raw value: `torch.ones(22)` (effective temp ≈ 1.31) or `torch.full((22,), 0.5413)` (exact effective temp = 1.0). The Builder is explicitly directed to use `torch.ones(22)`. No ambiguity.

2. `heatmap_temperature=1.0` is passed in config but unused when `heatmap_learnable_temp=True`. The design explicitly states this is intentional for config/signature consistency. The Builder must ensure `forward()` checks `self.heatmap_learnable_temp` first before using the scalar. Clearly stated in invariant 7 and in the config section note.

3. The `loss()` code uses `_heatmap_logits[i].T` — raw logits before any temperature scaling. The design explicitly justifies this in invariant 9: "temperature affects how sharply the pooling attention acts, not the loss target." Correct.

---

## Conclusion

All design details are fully specified. The Builder can implement this without guessing. No invariant violations. **APPROVED.**
