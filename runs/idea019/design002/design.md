# Design 002 — K_s=8 Deformable Sampling, 22 Body Queries, Linear Hand Recovery

**Design Description:** Deformable sparse cross-attention (K_s=8) on 22-query body-only decoder; hand joints (22–69) recovered via `Linear(22*hidden_dim, 48*3)`; auxiliary hand loss weight 0.1.

**Starting Point:** `baseline/`

---

## Overview

This design composes two independently-validated improvements, changing the algorithm at the cross-attention level:
1. **22-query body-only decoder** (from idea008/design002): removes hand-query contamination of self-attention and cross-attention. idea008/design002 achieved mpjpe_rel_val=333mm and mpjpe_abs=534mm at stage-2 — the best relative MPJPE across all 18 prior ideas.
2. **Deformable sparse cross-attention algorithm** (new in idea019): each of the 22 body queries predicts K_s=8 sampling locations on the 2D feature grid instead of attending densely over 960 tokens.

The 22 body queries each learn their own reference point and offset prediction, giving each body joint a spatially localised attention footprint aligned to its anatomical region. The pelvis token (query 0) learns to attend to depth-informative spatial regions.

Hand joints (indices 22–69) are recovered via a linear projection from the flattened 22 body query features: `Linear(22×256, 48×3)`. An auxiliary hand loss (weight 0.1) keeps this projection anchored in pose space and provides regularising gradient to the body decoder.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

This design builds on the same `_DeformableDecoderLayer` class introduced in Design 001. The Builder must implement `_DeformableDecoderLayer` exactly as specified in Design 001. No changes to `_DeformableDecoderLayer` itself are needed for Design 002.

#### Changes to `Pose3dTransformerHead.__init__`

New constructor kwargs (same set as Design 001, but different defaults and `num_body_queries=22`):
- `deform_num_points: int = 0`
- `deform_hidden_dim: int = 64`
- `num_body_queries: int = 70`  (config passes 22)
- `num_decoder_layers: int = 1`
- `hand_aux_loss_weight: float = 0.0`  (config passes 0.1)
- `aux_body_loss_weight: float = 0.0`

Store all as instance attributes: `self.use_deform`, `self.num_body_queries`, `self.hand_aux_loss_weight`, `self.aux_body_loss_weight`, `self.num_decoder_layers`.

Change joint query embedding to use `num_body_queries`:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
# For Design 002: nn.Embedding(22, 256)
```

Add hand projection layer (after `self.uv_out`):
```python
self.hand_proj = nn.Linear(
    num_body_queries * hidden_dim,          # 22 * 256 = 5632
    (num_joints - num_body_queries) * 3,    # 48 * 3 = 144
)
```

Only add `self.hand_proj` when `num_body_queries < num_joints` (i.e., when `num_joints - num_body_queries > 0`). Use a guard:
```python
self.has_hand_proj = (num_body_queries < num_joints)
if self.has_hand_proj:
    self.hand_proj = nn.Linear(
        num_body_queries * hidden_dim,
        (num_joints - num_body_queries) * 3,
    )
```

Decoder module list (same structure as Design 001):
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
else:
    self.decoder_layers = nn.ModuleList([
        _DecoderLayer(hidden_dim, num_heads, dropout)
        for _ in range(num_decoder_layers)
    ])
self.decoder_layer = self.decoder_layers[0]  # backward-compat alias
```

#### Changes to `_init_head_weights`

Add to existing init code:

```python
# Deformable offset network near-zero init (same as Design 001)
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

# Hand projection init
if self.has_hand_proj:
    nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
    nn.init.zeros_(self.hand_proj.bias)
```

#### Changes to `forward()`

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
        # (B, num_body_queries, hidden_dim) = (B, 22, 256)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_grid)
        decoded = queries                              # (B, 22, 256)
    else:
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_proj)
        decoded = queries

    # Body joints from body queries
    body_joints = self.joints_out(decoded)            # (B, 22, 3)

    # Hand recovery via linear projection from flattened body features
    if self.has_hand_proj:
        body_flat = decoded.reshape(
            B, self.num_body_queries * self.hidden_dim)     # (B, 22*256=5632)
        num_hand = self.num_joints - self.num_body_queries  # 48
        hand_joints = self.hand_proj(body_flat).reshape(
            B, num_hand, 3)                                 # (B, 48, 3)
        joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)
    else:
        joints = body_joints                          # (B, 70, 3) when num_body_queries=70

    pelvis_token = decoded[:, 0, :]                   # (B, 256) — query 0 = pelvis token
    pelvis_depth = self.depth_out(pelvis_token)       # (B, 1)
    pelvis_uv = self.uv_out(pelvis_token)             # (B, 2)

    return {
        'joints': joints,          # always (B, 70, 3)
        'pelvis_depth': pelvis_depth,
        'pelvis_uv': pelvis_uv,
    }
```

#### Changes to `loss()`

After the existing body joint loss, depth loss, UV loss, add:

```python
# Auxiliary hand loss (weight 0.1) — provides regularising gradient to body decoder
if self.hand_aux_loss_weight > 0.0 and self.has_hand_proj:
    _HAND = list(range(self.num_body_queries, self.num_joints))  # range(22, 70)
    losses['loss/hand_aux/train'] = self.hand_aux_loss_weight * self.loss_joints_module(
        pred['joints'][:, _HAND], gt_joints[:, _HAND])
```

Reuse `self.loss_joints_module` — do not create a new loss module.

The existing `_BODY = list(range(0, 22))` for body joint loss, `_train_mpjpe`, and `_train_mpjpe_abs` computations are unchanged.

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
    num_decoder_layers=1,
    hand_aux_loss_weight=0.1,
    aux_body_loss_weight=0.0,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values (optimizer lr=1e-4, weight_decay=0.03, clip_grad max_norm=1.0, accumulative_counts=8, LR schedule, data pipeline, hooks, backbone) are **identical to baseline**.

---

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Count Delta vs Baseline

New parameters vs. Design 001:
- `joint_queries`: reduced from 70×256 to 22×256 = **−12,288 parameters**
- `hand_proj`: Linear(5632, 144) = 5632×144 + 144 = **810,576 parameters**
- `_DeformableDecoderLayer` with `num_queries=22` instead of 70:
  - `ref_points`: 22×2 = 44 (vs. 70×2 = 140 in Design 001) = **−96 parameters**
  - All other layer params unchanged (offset_net, attn_weight_net, value_proj, out_proj share weights across queries)

Net vs. baseline:
- Much fewer self-attention parameters on 22 vs. 70 queries
- Addition of `hand_proj` (810,576)

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Output `joints` shape must be `(B, 70, 3)` — `hand_proj` path produces `(B, 48, 3)` and `cat` with body gives `(B, 70, 3)`.
3. `self.num_joints = 70` must remain for `predict()` output shape.
4. `pelvis_token = decoded[:, 0, :]` — query 0 is still the pelvis token (first of 22 body queries).
5. `_BODY = list(range(0, 22))` for body joint loss — only body joint indices supervised.
6. `_HAND = list(range(22, 70))` for auxiliary hand loss — exactly 48 indices.
7. `hand_proj` input dimension: `num_body_queries * hidden_dim = 22 * 256 = 5632`. Output: `(num_joints - num_body_queries) * 3 = 48 * 3 = 144`. These must be computed dynamically from kwargs in `__init__` (not hardcoded) to be robust.
8. `loss_joints_module` is reused for `loss/hand_aux/train` — no new loss module instantiation.
9. AMP compatibility: `grid = grid.to(spatial_grid.dtype)` cast required in `_sample_spatial_features`.
10. `in_channels=1024` is hardcoded as literal in config (not `embed_dim` variable).
11. MMEngine config: all values are int/float/str literals. No `import` statements.
12. `num_body_queries=22` passed to `_DeformableDecoderLayer` must match `joint_queries.weight.shape[0]` — assert recommended.
13. `has_hand_proj` guard ensures Design 001 (num_body_queries=70) does not create or call `hand_proj`.
14. Backbone, data preprocessor, metric, transforms, pelvis_utils invariant.
15. Seed `2026`, batch size `4`, accumulation `8` — do not change.

---

## Expected Behavior After Change

- Body decoder operates on 22 clean queries with no hand-query contamination. Self-attention among 22 queries is less polluted by off-topology (hand) queries.
- Each body query has its own learned spatial reference point and K_s=8 sampling locations. Over 20 epochs, reference points migrate toward anatomically relevant grid positions.
- Hand joints recovered via linear projection — not from independent hand-specific queries but from body features, providing structurally consistent (if lower fidelity) hand output that does not interfere with body cross-attention.
- `loss/hand_aux/train` appears in logs at ~0.1 × SmoothL1(hand), providing auxiliary gradient regularisation to the body decoder.
- Self-attention matrix: (B, 8, 22, 22) vs. baseline (B, 8, 70, 70) — much smaller, faster.
- Cross-attention: (B, 22, 8) attention weights vs. baseline (B, 8, 70, 960) — dramatically smaller.
- `composite_val` target: < 330 at stage-1 (best prior: 328.14 — idea013/design003); < 220 at stage-2.
- `mpjpe_rel_val` target: < 333 mm at stage-2 (matching or beating idea008/design002's 333mm best).
- Output dict `{'joints': (B,70,3), 'pelvis_depth': (B,1), 'pelvis_uv': (B,2)}` — unchanged shape. All metric code receives identical-shape tensors.
