# Design Review — idea034 / design002 (Variant B: Multi-Scale Sinusoidal 3D PE)

**Verdict: APPROVED**

## Checks
- **Design Description:** present — fixed multi-scale sinusoidal 3D basis (σ ∈ {0.25, 1, 4, 16} m) + zero-init Linear projection.
- **Starting point:** `baseline/` — explicit.
- **Files to modify:** only `pose3d_transformer_head.py`, `pelvis_utils.py`, `config.py` — compliant.
- **Algorithmic changes:** `_SinusoidalMetric3DPE` class body given verbatim with explicit concat order, tensor reshape semantics (`basis_dim = 6*K`), and zero-init on `proj`. Depth/K extraction, unprojection helper, `forward/loss/predict` hook points all reference design001's spec identically — no ambiguity (design explicitly instructs "reuse if design001 is implemented first; otherwise add with same signature and body").
- **Config values:** `use_metric_pe_3d=True`, `metric_pe_variant='sinusoidal'`, `metric_pe_sigmas=(0.25, 1.0, 4.0, 16.0)`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0`. Tuple of floats is a bare literal — MMEngine-compatible (no import required).
- **Init / numerical safety:** zero-init on `proj.weight`/`proj.bias` → baseline-identical step 0. Sinusoidal basis is bounded in [-1, 1] so the pure-zero depth/missing-sample path remains NaN-free.
- **Invariant preservation:** same as design001 — evaluation/dataset/transforms/backbone/preprocessor/infra/`train.py` untouched. `recover_pelvis_3d` unchanged.
- **Edge cases:** wrap-around of the finest σ across image-edge coordinates is noted and mitigated by coarser scales; NaN/Inf guard inherited from `unproject_grid_to_metric_3d`.

All parameters (σ list, basis dim, reshape order, init target) are exact; Builder has no guesswork.
