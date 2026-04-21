**Design Description:** K=32 super-tokens + 2 decoder layers with auxiliary loss weight 0.4 on intermediate layer output.

**Starting Point:** `baseline/`

---

## Overview

Combine K=32 super-token pooling (same as Design A) with 2 stacked decoder layers that both cross-attend over the same K=32 super-tokens. The slot attention is computed once and reused by both decoder layers. The first decoder layer's output drives an auxiliary joint loss (weight 0.4) to prevent gradient vanishing at the intermediate layer. The second (final) decoder layer's output drives the primary joint, depth, and UV losses — identical to baseline.

**Memory rationale:** Baseline decoder cross-attention FLOPS ∝ 960 per layer. Design C uses 2 layers × 32 K/V → FLOPS ∝ 64 (33% of baseline per forward pass). The slot attention (one extra MHA over 960 K/V) costs ∝ 32 × 960 for the K=32 slot queries, but this is computed once, not per decoder layer. Net memory change vs. baseline is strongly negative.

**Algorithm:**
1. Compute K=32 super-tokens from 960 spatial tokens via slot-attention (same as Design A).
2. Run decoder layer 1: `decoded_1 = decoder_layers[0](queries, super_tokens)`.
3. Run decoder layer 2: `decoded_2 = decoder_layers[1](decoded_1, super_tokens)` (super_tokens reused, not recomputed).
4. Primary losses on `decoded_2` output (joints, depth, UV).
5. Auxiliary joint loss on `decoded_1` output: `aux_loss_weight * loss_joints(joints_layer1[:, _BODY], gt_joints[:, _BODY])`.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Constructor signature** — same four new parameters as Designs A and B:

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_super_tokens: int = 0,
    slot_pos_init: bool = False,
    num_decoder_layers: int = 1,
    aux_loss_weight: float = 0.0,
    loss_joints: ConfigType = ...,
    ...
):
```

Store all four: `self.num_super_tokens`, `self.slot_pos_init`, `self.num_decoder_layers`, `self.aux_loss_weight`.

**New modules in `__init__`** — same as Designs A and B:

```python
if self.num_super_tokens > 0:
    self.slot_queries = nn.Embedding(num_super_tokens, hidden_dim)
    self.slot_attn = nn.MultiheadAttention(
        hidden_dim, num_heads, dropout=dropout, batch_first=True)
    self.slot_norm = nn.LayerNorm(hidden_dim)
```

**Decoder layers** — same replacement as Designs A and B:

```python
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Remove `self.decoder_layer` (singular).

**`_init_head_weights()` changes:**
```python
if self.num_super_tokens > 0:
    nn.init.trunc_normal_(self.slot_queries.weight, std=0.02)
    # slot_pos_init is False for this design — no spatial block init
```

**`forward()` changes** — same as Design A, with `num_super_tokens=32` and `num_decoder_layers=2`:

```python
spatial = feat.flatten(2).transpose(1, 2)
spatial = self.input_proj(spatial)
pos_enc = self._get_pos_enc(H, W, feat.device)
spatial = spatial + pos_enc

if self.num_super_tokens > 0:
    S = self.slot_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 32, hidden_dim)
    S_normed = self.slot_norm(S)
    super_tokens, _ = self.slot_attn(S_normed, spatial, spatial)  # (B, 32, hidden_dim)
    spatial_for_decoder = super_tokens
else:
    spatial_for_decoder = spatial

queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

decoded = queries
intermediate_outputs = []
for layer in self.decoder_layers:
    decoded = layer(decoded, spatial_for_decoder)
    intermediate_outputs.append(decoded)

# intermediate_outputs[0] = decoded_1 (layer 1 output)
# intermediate_outputs[1] = decoded_2 (layer 2 output = final decoded)

joints = self.joints_out(decoded)          # (B, num_joints, 3) — from final layer
pelvis_token = decoded[:, 0, :]
pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)
```

Note: `forward()` returns only the final-layer output. `intermediate_outputs` is only used in `loss()`. The `forward()` return dict is unchanged.

**`loss()` changes** — this is the critical change for Design C. The `loss()` method must call the modified `forward()` and also compute the auxiliary loss from the intermediate layer:

The current `loss()` method calls `pred = self.forward(feats)`. Since `forward()` does not return intermediate outputs, `loss()` must directly call the forward computation steps to access `intermediate_outputs`. The cleanest implementation is to refactor the inner computation of `forward()` into a shared helper, OR to make `loss()` call a new internal method that returns both final and intermediate outputs.

**Recommended implementation:** Add an internal `_forward_with_intermediates()` method:

```python
def _forward_with_intermediates(
    self, feats: Tuple[torch.Tensor, ...]
) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
    """Forward pass returning both final outputs and intermediate decoder outputs.

    Returns:
        (pred_dict, intermediate_outputs):
            pred_dict: same as forward() return dict
            intermediate_outputs: list of (B, num_joints, hidden_dim) tensors,
                one per decoder layer, in order
    """
    feat = feats[-1]
    B, C, H, W = feat.shape

    spatial = feat.flatten(2).transpose(1, 2)
    spatial = self.input_proj(spatial)
    pos_enc = self._get_pos_enc(H, W, feat.device)
    spatial = spatial + pos_enc

    if self.num_super_tokens > 0:
        S = self.slot_queries.weight.unsqueeze(0).expand(B, -1, -1)
        S_normed = self.slot_norm(S)
        super_tokens, _ = self.slot_attn(S_normed, spatial, spatial)
        spatial_for_decoder = super_tokens
    else:
        spatial_for_decoder = spatial

    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

    decoded = queries
    intermediate_outputs = []
    for layer in self.decoder_layers:
        decoded = layer(decoded, spatial_for_decoder)
        intermediate_outputs.append(decoded)

    joints = self.joints_out(decoded)
    pelvis_token = decoded[:, 0, :]
    pelvis_depth = self.depth_out(pelvis_token)
    pelvis_uv = self.uv_out(pelvis_token)

    pred = {
        'joints': joints,
        'pelvis_depth': pelvis_depth,
        'pelvis_uv': pelvis_uv,
    }
    return pred, intermediate_outputs
```

Then update:
- `forward()` to call `pred, _ = self._forward_with_intermediates(feats); return pred`
- `loss()` to call `pred, intermediate_outputs = self._forward_with_intermediates(feats)` and then add the auxiliary loss:

```python
def loss(self, feats, batch_data_samples, train_cfg={}):
    pred, intermediate_outputs = self._forward_with_intermediates(feats)

    # --- GT extraction (unchanged from baseline) ---
    gt_joints = torch.cat([d.gt_instances.lifting_target for d in batch_data_samples], dim=0)
    if gt_joints.dim() == 4:
        gt_joints = gt_joints.squeeze(1)
    gt_joints = gt_joints.to(pred['joints'].device)

    gt_depth = torch.stack([d.gt_instance_labels.pelvis_depth for d in batch_data_samples])
    gt_depth = gt_depth.to(pred['pelvis_depth'].device)
    if gt_depth.dim() == 1:
        gt_depth = gt_depth.unsqueeze(-1)

    gt_uv = torch.cat([d.gt_instance_labels.pelvis_uv for d in batch_data_samples], dim=0)
    gt_uv = gt_uv.to(pred['pelvis_uv'].device)

    _BODY = list(range(0, 22))
    losses = dict()

    # Primary losses (final layer)
    losses['loss/joints/train'] = self.loss_joints_module(
        pred['joints'][:, _BODY], gt_joints[:, _BODY])
    losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
        pred['pelvis_depth'], gt_depth)
    losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
        pred['pelvis_uv'], gt_uv)

    # Auxiliary loss on intermediate decoder outputs (all layers except last)
    if self.aux_loss_weight > 0.0:
        for i, inter_decoded in enumerate(intermediate_outputs[:-1]):
            inter_joints = self.joints_out(inter_decoded)  # (B, num_joints, 3)
            losses[f'loss/joints_aux_{i}/train'] = self.aux_loss_weight * self.loss_joints_module(
                inter_joints[:, _BODY], gt_joints[:, _BODY])

    # MPJPE tracking (unchanged from baseline)
    with torch.no_grad():
        self._train_mpjpe = (
            (pred['joints'][:, _BODY] - gt_joints[:, _BODY]).norm(dim=-1).mean() * 1000.0)
        self._train_mpjpe_abs = _compute_mpjpe_abs(
            pred['joints'], gt_joints,
            pred['pelvis_depth'], gt_depth,
            pred['pelvis_uv'], gt_uv,
            batch_data_samples)

    return losses, pred
```

Key note: `self.joints_out` is shared between the primary loss (final layer) and auxiliary loss (intermediate layers). This is intentional and matches how idea001/design002 implemented auxiliary losses — the same output projection head is used to probe intermediate representations.

**`predict()` — no changes.** Calls `self.forward(feats)` which internally calls `_forward_with_intermediates` and discards intermediates.

### 2. `config.py`

In the `head=dict(...)` block:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_super_tokens=32,
    slot_pos_init=False,
    num_decoder_layers=2,
    aux_loss_weight=0.4,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values remain identical to baseline (same LR, weight decay, warmup, batch, seed, data pipeline, hooks).

---

## Constraints and Invariants to Preserve

1. **Joint loss scope:** `_BODY = list(range(0, 22))` — indices 0–21 only. Applied to BOTH primary and auxiliary joint losses.
2. **Pelvis token:** `decoded[:, 0, :]` from final decoder layer only — pelvis depth and UV are not computed for intermediate layers.
3. **`persistent_workers=False`** — do not change.
4. **No Python `import` in `config.py`** — `num_super_tokens=32`, `slot_pos_init=False`, `num_decoder_layers=2`, `aux_loss_weight=0.4` are all literals.
5. **Absolute imports in `pose3d_transformer_head.py`** — unchanged.
6. **Super-tokens are computed once** and reused by both decoder layers — the slot attention (`self.slot_attn`) is called exactly once per forward pass, and `spatial_for_decoder` (= super_tokens) is passed unchanged to both `decoder_layers[0]` and `decoder_layers[1]`.
7. **`self.joints_out` is shared** between primary loss (final `decoded`) and auxiliary loss (intermediate `decoded`). This is by design — no separate auxiliary output projection needed.
8. **Auxiliary loss key format:** `f'loss/joints_aux_{i}/train'` where `i` is the layer index (0-indexed, excludes last). For `num_decoder_layers=2`, this produces one key: `'loss/joints_aux_0/train'`.
9. **`_forward_with_intermediates` is an internal helper** — not part of the public API, not registered, not called by MMEngine hooks directly. `forward()` and `predict()` still work as before.
10. **Output tensor shapes unchanged:** `(B, 70, 3)`, `(B, 1)`, `(B, 2)`.
11. **`_DecoderLayer` class not modified.**
12. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py`.
13. **`slot_pos_init=False`** for this design — no block-averaged init. `_init_head_weights` uses `trunc_normal_(std=0.02)` for slot queries only.
14. **`intermediate_outputs[:-1]`** selects all decoder layer outputs except the last for auxiliary loss. For `num_decoder_layers=2`, this is exactly `intermediate_outputs[0]` (layer 1 output).

---

## Expected Behavior After Change

- Model computes 960 spatial tokens → K=32 super-tokens via slot attention (same as Design A).
- Two decoder layers both cross-attend over the same K=32 super-tokens.
- Layer 1 output drives auxiliary joint loss (weight 0.4, body joints only).
- Layer 2 output drives primary losses (joint + depth + UV, identical to baseline).
- Effective cross-attention cost: 2 layers × 32 K/V → proportional to 64 (vs. baseline 1 layer × 960 → proportional to 960). Net reduction: ~93% in cross-attention FLOPS.
- Total loss = primary_joint_loss + 0.4 × aux_joint_loss_layer1 + depth_loss + uv_loss.
- New parameters vs. baseline: slot_queries (8,192 scalars) + slot_attn (~1.3M params) + slot_norm + one additional `_DecoderLayer` (~3.2M params). Total ~4.5M extra parameters.
- Expected: `composite_val < 320` at stage-1 epoch 20. This is the primary design — combining token compression with decoder depth, targeting the best composite of all three designs.
