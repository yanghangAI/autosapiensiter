## Design Review — idea030/design001

**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-21

---

### Checklist

**1. Design Description present:** Yes — "Single-layer spatial encoder (8 heads, zero-init) inserted before decoder cross-attention."

**2. Starting-point path specified:** Yes — `baseline/`

**3. Files to change specified:** Yes — `pose3d_transformer_head.py` and `config.py` only. No changes to `pelvis_utils.py`, invariant files, or training infrastructure.

**4. Invariants respected:** Yes. No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `data_preprocessor`, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py` wrapper, or `tools/train.py`.

**5. Algorithmic changes specified exactly:** Yes.
- New `_EncoderLayer` class with full code listing (self-attn + FFN + pre-norm + residual + dropout, zero-init on `self_attn.out_proj` and `ffn[-2]`).
- Exact insertion point in `pose3d_transformer_head.py`: before `_DecoderLayer` class (line 77 in baseline).
- Exact new kwargs for `Pose3dTransformerHead.__init__` with defaults.
- Exact insertion point in `forward()`: after `spatial = spatial + pos_enc`, before `queries = self.joint_queries.weight.unsqueeze(0).expand(...)`. Verified against baseline `forward()` — these lines exist at lines 241 and 244 respectively.
- Exact `config.py` head dict snippet with all 5 new kwargs as literals.

**6. Config values and defaults specified:** Yes.
- `use_spatial_encoder=True`, `num_encoder_layers=1`, `encoder_num_heads=8`, `encoder_dropout=0.1`, `encoder_zero_init=True`.
- Default values in `__init__` signature: `use_spatial_encoder=False`, `num_encoder_layers=1`, `encoder_num_heads=8`, `encoder_dropout=0.1`, `encoder_zero_init=True`.
- All literals; no Python `import` statements in config.

**7. Implementation readiness:** The Builder can implement without guessing.
- `ffn[-2]` index is explicitly explained (second-to-last element in `nn.Sequential` = `nn.Linear(embed_dim*4, embed_dim)`; last = `nn.Dropout`). Correct per baseline `_DecoderLayer.ffn` structure.
- `self.self_attn.out_proj` is standard `nn.MultiheadAttention` attribute — no ambiguity.
- Pre-norm style is specified and consistent with existing `_DecoderLayer`.
- When `use_spatial_encoder=False`, no `spatial_encoder` attribute is created — correctly stated.

**8. Output shapes unchanged:** Confirmed — `joints (B,70,3)`, `pelvis_depth (B,1)`, `pelvis_uv (B,2)`.

**9. Loss/predict unchanged:** Confirmed — design explicitly states no changes to `loss()` or `predict()`.

**10. MMEngine config constraint satisfied:** Yes — all new config values are bool/int/float literals.

---

### No Issues Found

All required design elements are present, explicit, and consistent with the baseline code. The Builder can implement this design without ambiguity.
