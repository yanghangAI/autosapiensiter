# Design Review — idea019/design003

**Verdict: APPROVED**

**Reviewed:** 2026-04-21

---

## Summary

Design 003 specifies a two-layer deformable decoder (K_s=8, 22 body queries) with intermediate body joint supervision (weight 0.4) after layer 0 and auxiliary hand loss (weight 0.1). It extends Design 002 by adding `num_decoder_layers=2` and intermediate supervision infrastructure. The design is complete, explicit, and implementation-ready.

---

## Checklist

### Design Description
Present and accurate. ✓

### Starting Point
`baseline/` — explicitly stated. ✓

### Files to Change
- `pose3d_transformer_head.py` — fully specified. ✓
- `config.py` — fully specified. ✓
- `pelvis_utils.py` — no changes (stated explicitly). ✓
- No invariant files touched. ✓

### Algorithmic Changes

**`_DeformableDecoderLayer`**: Defers to Design 001 spec; no changes. ✓

**`Pose3dTransformerHead.__init__` changes**:
- Same new kwargs as Design 001/002. ✓
- `nn.Embedding(22, 256)` for joint queries. ✓
- `hand_proj` with `has_hand_proj` guard (same as Design 002). ✓
- `decoder_layers = nn.ModuleList([...] * 2)` — two independent `_DeformableDecoderLayer` instances. ✓
- Constraint 14 explicitly states both layers have independent parameters (not shared). ✓
- `decoder_layer = decoder_layers[0]` alias. ✓
- `has_intermediate_sup = (num_decoder_layers > 1 and aux_body_loss_weight > 0.0)` guard. ✓
- `intermediate_joints_out = nn.ModuleList([nn.Linear(hidden_dim, 3)] * (num_decoder_layers-1))` = 1 head for Design 003. ✓

**`_init_head_weights`** changes:
- Deformable init block (loops over all decoder layers, covering both). ✓
- Intermediate supervision head init: `trunc_normal_(std=0.02)` for weight, `zeros_` for bias with `is not None` guard. ✓
- `hand_proj` init same as Design 002. ✓

**`forward()`** changes:
- Collects intermediate decoded states in `intermediate_decoded` list. ✓
- Stores as `self._intermediate_decoded` for consumption by `loss()`. ✓
- Collection condition: `if i < len(self.decoder_layers) - 1` — correctly collects all but final layer. ✓
- Final `decoded = queries` is the output of the last layer. ✓
- Non-deformable else path sets `self._intermediate_decoded = []`. ✓
- Hand recovery and pelvis token identical to Design 002. ✓
- Output dict shape unchanged. ✓
- `predict()` safety: `self._intermediate_decoded` is populated but `loss()` not called during inference — acknowledged and stated safe. ✓

**`loss()`** changes:
- Intermediate body supervision loop:
  - Key: `f'loss/joints_inter{idx}/train'` (= `'loss/joints_inter0/train'` for Design 003). ✓
  - Applied to `self.intermediate_joints_out[idx](inter_decoded)[:, _BODY]` where `_BODY = range(0,22)`. ✓
  - Weight: `self.aux_body_loss_weight` = 0.4. ✓
  - Guarded by `has_intermediate_sup and hasattr(self, '_intermediate_decoded')`. ✓
  - Reuses `self.loss_joints_module`. ✓
- Auxiliary hand loss: same as Design 002. ✓
- Ordering: intermediate body loss before hand aux loss (both after base body/depth/UV losses). ✓

### Config Values
All int/float/str literals. No Python import statements. ✓
- `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`. ✓
- `deform_num_points=8`, `deform_hidden_dim=64`. ✓
- `in_channels=1024` hardcoded literal. ✓
- All other config identical to baseline. ✓

### Invariants Preserved
- Output `joints` shape `(B, 70, 3)`. ✓
- `self.num_joints = 70`. ✓
- `pelvis_token = decoded[:, 0, :]`. ✓
- Body joint loss restricted to indices 0-21 (including intermediate supervision). ✓
- `persistent_workers=False`. ✓
- Backbone, metric, transforms, data preprocessor, infra files untouched. ✓

### Guards and Safety
- `has_intermediate_sup` False when `num_decoder_layers=1` (Design 001/002 safe). ✓
- `has_hand_proj` False when `num_body_queries=70` (Design 001 safe). ✓
- `hasattr(self, '_intermediate_decoded')` guard in `loss()` for robustness. ✓
- Independent parameters per layer explicitly called out. ✓

### No Issues Found

The design is complete. A Builder can implement it without guessing.
