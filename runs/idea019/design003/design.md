# Design 003 — K_s=8 Deformable Sampling + 2-Layer Decoder + Intermediate Supervision

**Design Description:** Two-layer deformable decoder (K_s=8, 22 body queries); intermediate body joint supervision after layer 1 (weight 0.4); hand recovery via `Linear(22*hidden_dim, 48*3)`; auxiliary hand loss 0.1.

**Starting Point:** `baseline/`

---

## Overview

This design extends Design 002 by stacking two deformable decoder layers and adding intermediate supervision at layer 1. The VRAM savings from two mechanisms — 22 queries (vs. 70 baseline) and K_s=8 sparse sampling (vs. 960 tokens) — make two decoder layers feasible on the 2080 Ti without memory pressure.

Cross-attention VRAM comparison (rough element count):
- Baseline (1 layer, 70 queries, 960 tokens): B × 70 × 960 = 537,600 elements/layer
- Design 003 (2 layers, 22 queries, K_s=8): 2 × B × 22 × 8 = 352 elements — 1,527× smaller cross-attention footprint

Progressive refinement via two decoder layers mirrors the strategy of idea001/design001 (2-layer standard decoder, best stage-2 composite 224.52mm). The deformable sampling algorithm replaces dense cross-attention, and intermediate supervision at layer 1 prevents gradient vanishing at the first layer.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

This design builds on the same `_DeformableDecoderLayer` class from Design 001. The Builder must implement `_DeformableDecoderLayer` exactly as specified in Design 001. No changes to `_DeformableDecoderLayer` itself.

#### Changes to `Pose3dTransformerHead.__init__`

Same new kwargs as Design 001/002 (Builder implements once for all three designs):
- `deform_num_points: int = 0`
- `deform_hidden_dim: int = 64`
- `num_body_queries: int = 70`  (config passes 22)
- `num_decoder_layers: int = 1`  (config passes 2)
- `hand_aux_loss_weight: float = 0.0`  (config passes 0.1)
- `aux_body_loss_weight: float = 0.0`  (config passes 0.4)

All stored as instance attributes: `self.use_deform`, `self.num_body_queries`, `self.hand_aux_loss_weight`, `self.aux_body_loss_weight`, `self.num_decoder_layers`, `self.has_hand_proj`.

Joint queries: `nn.Embedding(num_body_queries, hidden_dim)` = `nn.Embedding(22, 256)`.

Hand projection (same as Design 002):
```python
self.has_hand_proj = (num_body_queries < num_joints)
if self.has_hand_proj:
    self.hand_proj = nn.Linear(
        num_body_queries * hidden_dim,          # 22 * 256 = 5632
        (num_joints - num_body_queries) * 3,    # 48 * 3 = 144
    )
```

Decoder module list (Design 003 specific — `num_decoder_layers=2`):
```python
self.use_deform = deform_num_points > 0
if self.use_deform:
    self.decoder_layers = nn.ModuleList([
        _DeformableDecoderLayer(
            hidden_dim, num_heads, dropout,
            num_points=deform_num_points,
            deform_hidden_dim=deform_hidden_dim,
            num_queries=num_body_queries,
        )
        for _ in range(num_decoder_layers)
    ])
# For Design 003: nn.ModuleList of 2 _DeformableDecoderLayer instances
self.decoder_layer = self.decoder_layers[0]  # backward-compat alias

# Intermediate supervision head (for layers 1..N-1, i.e., not the last layer)
# Only active when num_decoder_layers > 1 and aux_body_loss_weight > 0
self.has_intermediate_sup = (num_decoder_layers > 1 and aux_body_loss_weight > 0.0)
if self.has_intermediate_sup:
    # Separate joint output head for intermediate layer(s)
    # Shared architecture with self.joints_out (Linear(hidden_dim, 3))
    # Number of intermediate heads = num_decoder_layers - 1
    self.intermediate_joints_out = nn.ModuleList([
        nn.Linear(hidden_dim, 3)
        for _ in range(num_decoder_layers - 1)
    ])
# For Design 003: 1 intermediate head (after layer 0, before layer 1)
```

#### Changes to `_init_head_weights`

```python
# Deformable offset network near-zero init (same as Design 001/002)
if self.use_deform:
    for layer_mod in self.decoder_layers:
        nn.init.zeros_(layer_mod.offset_net[-1].weight)
        nn.init.zeros_(layer_mod.offset_net[-1].bias)
        nn.init.zeros_(layer_mod.attn_weight_net.weight)
        nn.init.zeros_(layer_mod.attn_weight_net.bias)
        nn.init.trunc_normal_(layer_mod.value_proj.weight, std=0.02)
        if layer_mod.value_proj.bias is not None:
            nn.init.zeros_(layer_mod.value_proj.bias)
        nn.init.trunc_normal_(layer_mod.out_proj.weight, std=0.02)
        if layer_mod.out_proj.bias is not None:
            nn.init.zeros_(layer_mod.out_proj.bias)

# Intermediate supervision head init
if self.has_intermediate_sup:
    for head in self.intermediate_joints_out:
        nn.init.trunc_normal_(head.weight, std=0.02)
        if head.bias is not None:
            nn.init.zeros_(head.bias)

# Hand projection init
if self.has_hand_proj:
    nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
    nn.init.zeros_(self.hand_proj.bias)
```

#### Changes to `forward()`

For Design 003, `forward()` must collect intermediate decoded states for intermediate supervision during training. However, `forward()` itself does NOT apply intermediate supervision losses (losses are computed in `loss()`). The intermediate outputs must be returned or stored.

**Implementation approach**: store intermediate decoded tensors as an instance attribute `self._intermediate_decoded` (a list) in the deformable multi-layer path, so `loss()` can access them. This attribute is set in `forward()` and consumed in `loss()`. It is not included in the returned dict.

```python
def forward(self, feats):
    feat = feats[-1]  # (B, C, H, W)
    B, C, H, W = feat.shape

    spatial_flat = feat.flatten(2).transpose(1, 2)   # (B, H*W, C)
    spatial_proj = self.input_proj(spatial_flat)      # (B, H*W, hidden_dim)
    pos_enc = self._get_pos_enc(H, W, feat.device)
    spatial_proj = spatial_proj + pos_enc

    if self.use_deform:
        spatial_grid = spatial_proj.transpose(1, 2).view(
            B, self.hidden_dim, H, W)                 # (B, hidden_dim, H', W')
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        # (B, 22, 256)
        intermediate_decoded = []
        for i, layer in enumerate(self.decoder_layers):
            queries = layer(queries, spatial_grid)
            if i < len(self.decoder_layers) - 1:
                # Collect intermediate output for supervision (all but the last layer)
                intermediate_decoded.append(queries)
        decoded = queries                              # (B, 22, 256) — final layer output
        self._intermediate_decoded = intermediate_decoded   # [(B, 22, 256)] for layer 0
    else:
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_proj)
        decoded = queries
        self._intermediate_decoded = []

    # Body joints from final decoded
    body_joints = self.joints_out(decoded)            # (B, 22, 3)

    # Hand recovery
    if self.has_hand_proj:
        body_flat = decoded.reshape(
            B, self.num_body_queries * self.hidden_dim)     # (B, 5632)
        num_hand = self.num_joints - self.num_body_queries  # 48
        hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)  # (B, 48, 3)
        joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)
    else:
        joints = body_joints

    pelvis_token = decoded[:, 0, :]                   # (B, 256)
    pelvis_depth = self.depth_out(pelvis_token)       # (B, 1)
    pelvis_uv = self.uv_out(pelvis_token)             # (B, 2)

    return {
        'joints': joints,
        'pelvis_depth': pelvis_depth,
        'pelvis_uv': pelvis_uv,
    }
```

**Important**: `self._intermediate_decoded` is only meaningful when called from `loss()` (training mode). During `predict()`, `forward()` is called standalone and `self._intermediate_decoded` will be populated but not used. The `predict()` method only uses the returned dict — this is correct and safe.

#### Changes to `loss()`

After the existing body joint loss, depth loss, and UV loss:

```python
# Intermediate layer body joint supervision (Design 003 only: aux_body_loss_weight=0.4)
if self.has_intermediate_sup and hasattr(self, '_intermediate_decoded'):
    _BODY = list(range(0, 22))
    for idx, inter_decoded in enumerate(self._intermediate_decoded):
        inter_body_joints = self.intermediate_joints_out[idx](inter_decoded)  # (B, 22, 3)
        losses[f'loss/joints_inter{idx}/train'] = (
            self.aux_body_loss_weight * self.loss_joints_module(
                inter_body_joints[:, _BODY], gt_joints[:, _BODY]))

# Auxiliary hand loss (weight 0.1)
if self.hand_aux_loss_weight > 0.0 and self.has_hand_proj:
    _HAND = list(range(self.num_body_queries, self.num_joints))  # range(22, 70)
    losses['loss/hand_aux/train'] = self.hand_aux_loss_weight * self.loss_joints_module(
        pred['joints'][:, _HAND], gt_joints[:, _HAND])
```

**Intermediate supervision detail**: `self.intermediate_joints_out[0]` is applied to `self._intermediate_decoded[0]` (output of decoder layer 0). This produces `(B, 22, 3)` intermediate body joint predictions. The loss is SmoothL1 over body indices 0–21 with weight 0.4. This loss is included in the `losses` dict so MMEngine sums it into the total backward loss.

Reuse `self.loss_joints_module` for intermediate supervision — no new loss module.

The existing `_BODY = list(range(0, 22))` in the base body loss and `_train_mpjpe` computations are unchanged.

---

### 2. `config.py`

Replace the `head=dict(...)` block with:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=1024,
    hidden_dim=256,
    num_joints=70,
    num_heads=8,
    dropout=0.1,
    deform_num_points=8,
    deform_hidden_dim=64,
    num_body_queries=22,
    num_decoder_layers=2,
    hand_aux_loss_weight=0.1,
    aux_body_loss_weight=0.4,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values (optimizer lr=1e-4, weight_decay=0.03, betas=(0.9,0.999), clip_grad max_norm=1.0, accumulative_counts=8, LinearLR warmup 3 epochs start_factor=0.333, CosineAnnealingLR, data pipeline, hooks, backbone, dataloaders with persistent_workers=False, batch_size=4) are **identical to baseline**.

---

### 3. `pelvis_utils.py`

No changes.

---

## Memory Feasibility Estimate (AMP, batch=4, 2080 Ti 10.57 GB)

With AMP (float16 activations):
- **Self-attention per layer**: (B=4, 8_heads, 22, 22) = 7,744 float16 elements × 2 bytes × 2 layers ≈ 31 KB
- **Deformable cross-attention per layer**: sampled tensor (B=4, 22, 8, 256) = 180,224 float16 elements × 2 bytes × 2 layers ≈ 720 KB
- **FFN intermediate per layer**: (B=4, 22, 1024) × 2 layers ≈ 180 KB
- **Intermediate body joint predictions**: (B=4, 22, 3) ≈ negligible
- **Backbone and data preprocessor**: dominant consumer, same as all designs

Total additional VRAM from two deformable layers: < 1 MB. No OOM risk. The 2080 Ti budget is easily met.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Output `joints` shape must be `(B, 70, 3)` from `forward()` — `hand_proj` + cat guarantees this.
3. `self.num_joints = 70` must remain for `predict()` output shape.
4. `pelvis_token = decoded[:, 0, :]` — query 0 of the 22 body queries is the pelvis token.
5. Body joint loss restricted to `_BODY = list(range(0, 22))` — unchanged.
6. Intermediate supervision also restricted to `_BODY` indices (not full 70).
7. `self._intermediate_decoded` stores the layer 0 output (shape `(B, 22, 256)`) for the 2-layer case. Length of list: `num_decoder_layers - 1 = 1`.
8. `self.intermediate_joints_out` is an `nn.ModuleList` of length `num_decoder_layers - 1 = 1`, each element `nn.Linear(256, 3)`.
9. `loss/joints_inter0/train` key in the losses dict — MMEngine sums this into the total loss for backprop.
10. `loss/hand_aux/train` key in the losses dict — MMEngine sums into total loss.
11. AMP compatibility: `grid = grid.to(spatial_grid.dtype)` cast required in `_DeformableDecoderLayer._sample_spatial_features`.
12. `in_channels=1024` is hardcoded literal in config.
13. `num_body_queries=22` passed to each `_DeformableDecoderLayer` must match `joint_queries.weight.shape[0]`.
14. Both `_DeformableDecoderLayer` instances in `decoder_layers` are independent (separate parameters — not shared weights). Each layer has its own `ref_points`, `offset_net`, `attn_weight_net`, `value_proj`, `out_proj`, `self_attn`, `ffn`, norms.
15. `has_intermediate_sup` is only True when `num_decoder_layers > 1 AND aux_body_loss_weight > 0`. Safe guard for Design 001/002 (both have `num_decoder_layers=1`).
16. `has_hand_proj` guard ensures the `hand_proj` module is not created for Design 001 (`num_body_queries=70`).
17. MMEngine config: all values are int/float/str literals. No `import` statements in head dict.
18. Backbone, data preprocessor, metric, transforms, pelvis_utils invariant.
19. Seed `2026`, batch size `4`, accumulation `8` — do not change.

---

## Expected Behavior After Change

- Two-stage progressive decoding: layer 0 produces an initial estimate, layer 1 refines it. Both layers attend to the same spatial grid but have independent learnable parameters (separate `ref_points` and `offset_net` weights), allowing specialisation: layer 0 may learn coarse body-region references, layer 1 refines to precise joint locations.
- Intermediate supervision loss `loss/joints_inter0/train` at weight 0.4 appears in training logs. This prevents the gradient vanishing problem at layer 0 that would arise if supervision only came from the final layer output.
- `loss/hand_aux/train` at weight 0.1 appears in training logs, providing auxiliary regularisation to body decoder (same as Design 002).
- Two independent sets of reference points (one per decoder layer) learn to specialise: layer 0 points may gravitate toward coarse body-region tokens; layer 1 points may refine toward precise per-joint locations.
- Memory cost of two layers: < 1 MB additional VRAM vs. Design 002 (single layer). No VRAM risk on 2080 Ti.
- `composite_val` target: < 320 at stage-1; < 210 at stage-2 (improving on best prior 224.52 — idea001/design001).
- `mpjpe_body_val` target: < 175 mm at stage-1; < 155 mm at stage-2.
- `mpjpe_rel_val` target: < 330 mm at stage-2.
- Output dict `{'joints': (B,70,3), 'pelvis_depth': (B,1), 'pelvis_uv': (B,2)}` — unchanged shape. All downstream metric code (BedlamMPJPEMetric, TrainMPJPEAveragingHook, MetricsCSVHook) receives identical-shape tensors.
