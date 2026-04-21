
---
## 2026-04-21 — Code Review

**Verdict: REJECTED**

Gaussian loss reduction is `-(gt_hm * log_probs).sum()` (sum over both joints and spatial dims) instead of the design-specified `-(gt_hm * log_probs).sum(dim=-1).mean()` (sum over spatial, mean over joints). This inflates the heatmap loss by ~22×, making it 29.8 at init vs. expected ~1.37, swamping 3D regression. Fix: change `.sum()` to `.sum(dim=-1).mean()` at line 431 of `pose3d_transformer_head.py`.

---
## 2026-04-21 — Code Review (Revision)

**Verdict: APPROVED**

Applied the required fix: `-(gt_hm * log_probs).sum()` → `-(gt_hm * log_probs).sum(dim=-1).mean()` at line 431. Sanity test confirms `loss/heatmap/train: 1.356` ≈ expected ~1.37 at init (SLURM job 55860281). All other files and implementation unchanged.
