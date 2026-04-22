# Design Review — idea034 / design001 (Variant A: MLP Metric 3D PE, additive)

**Verdict: APPROVED**

## Checks
- **Design Description:** present, concrete (unproject per-cell depth through per-sample K, MLP embed, zero-init additive).
- **Starting point:** `baseline/` — explicit.
- **Files to modify:** only `pose3d_transformer_head.py`, `pelvis_utils.py`, `config.py`. Invariant files listed as untouched. Compliant with the allowed-file whitelist.
- **Algorithmic changes:** fully specified — depth extraction (mirrors idea004), K extraction, `unproject_grid_to_metric_3d` helper with explicit formula and sign convention matching `recover_pelvis_3d` (verified in `baseline/pelvis_utils.py` lines 14–46: `Y = -(u-cx)X/fx`, `Z = -(v-cy)X/fy`, `X = d`), `_Metric3DPE` class body given verbatim, forward/loss/predict hook points and ordering specified.
- **Config values:** `use_metric_pe_3d=True`, `metric_pe_variant='mlp_additive'`, `metric_pe_mlp_hidden=256`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0` — all bool/str/float literals, MMEngine-compatible.
- **Init / numerical safety:** zero-init of final Linear ensures baseline-identical step 0; fp32 cast for unprojection arithmetic; NaN/Inf guard via `torch.where`+clamp. AMP-safe.
- **Invariant preservation:** `recover_pelvis_3d` and `compute_mpjpe_abs` left unchanged; evaluation metric, dataset, transforms, backbone, data preprocessor, `train.py` wrapper, infra files untouched. Output dict keys/shapes and body-only loss unchanged. `persistent_workers=False` preserved.
- **Metadata availability:** `K` and `depth_npy_path` are confirmed in baseline `meta_keys` (config.py lines 173, 182); `depth_required=True` on `LoadBedlamLabels` — no upstream change needed.
- **Edge cases:** missing depth/K, variable crop size, pixel-centre offset all called out.

Builder has no guesswork: helper, module, hook point, init, dtype handling, and config values are all fully pinned.
