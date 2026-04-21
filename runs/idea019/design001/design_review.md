# Design Review — idea019/design001

**Verdict: APPROVED**

**Reviewed:** 2026-04-21

---

## Summary

Design 001 specifies a minimal diagnostic implementation of per-query deformable sparse cross-attention (K_s=8 sampling points) replacing standard dense cross-attention (70×960), with a single decoder layer and all 70 joint queries. The design is complete, explicit, and implementation-ready.

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

**`_DeformableDecoderLayer`**: Full class implementation provided, including:
- Constructor with all kwargs and their types/defaults. ✓
- `self_attn` (MHA, unchanged from baseline). ✓
- `offset_net`: `Linear(embed_dim, deform_hidden_dim) → GELU → Linear(deform_hidden_dim, num_points*2)`. ✓
- `ref_points`: `nn.Parameter(torch.full((num_queries, 2), 0.5))`. ✓
- `attn_weight_net`: `Linear(embed_dim, num_points)`. ✓
- `value_proj`, `out_proj`: `Linear(embed_dim, embed_dim)`. ✓
- FFN: identical to baseline structure. ✓
- Three LayerNorms, two Dropouts. ✓
- `_sample_spatial_features()`: bilinear interpolation via `F.grid_sample`, AMP cast `grid = grid.to(spatial_grid.dtype)` present. ✓
- `align_corners=True`, `padding_mode='border'` specified. ✓
- Offset scaling factor 0.1 explicitly specified. ✓
- `forward()`: pre-norm pattern, self-attn → deformable cross-attn → FFN. ✓

**`Pose3dTransformerHead.__init__` changes**:
- All new kwargs listed with types and defaults. ✓
- `self.decoder_layers = nn.ModuleList(...)` for both deformable and standard paths. ✓
- `self.decoder_layer = self.decoder_layers[0]` backward-compat alias. ✓
- `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` — for Design 001 num_body_queries=70, identical to baseline. ✓
- No `hand_proj` for Design 001 (num_body_queries=70, no guard needed). ✓
- All instance attributes stored. ✓

**`_init_head_weights`** changes:
- Near-zero init for `offset_net[-1]` weights and biases. ✓
- Near-zero init for `attn_weight_net`. ✓
- `trunc_normal_(std=0.02)` for `value_proj` and `out_proj`. ✓
- Bias init with `zeros_` for value_proj/out_proj, with `is not None` guard. ✓
- Rationale for stable warm-start clearly explained. ✓

**`forward()`** changes:
- Complete replacement with both deformable and standard paths. ✓
- Deformable path keeps spatial as 2D grid `(B, hidden_dim, H', W')`. ✓
- Standard path uses flat `(B, H*W, hidden_dim)`. ✓
- Both paths iterate `decoder_layers`. ✓
- Output shape `(B, 70, 3)` joints preserved. ✓
- `pelvis_token = decoded[:, 0, :]` preserved. ✓

**`loss()`** changes:
- Design specifies a `hand_aux_loss_weight > 0` guard must be added for forward compatibility. ✓ (Block body is never entered for Design 001 since weight=0.0.)

### Config Values
All int/float/str literals. No Python import statements. ✓
- `deform_num_points=8`, `deform_hidden_dim=64`, `num_body_queries=70`, `num_decoder_layers=1`, `hand_aux_loss_weight=0.0`, `aux_body_loss_weight=0.0` ✓
- `in_channels=1024` hardcoded literal ✓

### Invariants Preserved
- `persistent_workers=False` explicitly stated. ✓
- `self.num_joints=70`, output shapes unchanged. ✓
- Body joint loss restricted to indices 0-21. ✓
- `pelvis_token = decoded[:, 0, :]`. ✓
- Backbone, metric, transforms, data preprocessor, infra files untouched. ✓
- Seed 2026, batch 4, accumulation 8. ✓

### Edge Cases
- AMP dtype cast explicitly required and shown in `_sample_spatial_features`. ✓
- `offset_net[-1]` index verified against the 3-element `nn.Sequential` (indices 0, 1, 2 → final Linear is index 2). ✓
- `decoder_layers` (plural, ModuleList) used in forward, never `decoder_layer` singular in deformable path. ✓

### No Issues Found

The design is complete. A Builder can implement it without guessing.
