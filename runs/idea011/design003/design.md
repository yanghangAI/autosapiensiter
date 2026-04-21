# Design 003 — Two-pass Coordinate-Conditioned Decoder with Independent Pass-2 Weights and Intermediate Supervision

**Design Description:** Same two-pass coordinate-feedback architecture as Designs 001/002 (zero-init `coord_enc: Linear(3,256)→GELU→Linear(256,256)` feeding pass-1 joint coordinates back into pass-2 queries, residual joint output, pelvis read from pass-2 token 0) and intermediate supervision from Design 002 (weight=0.5 on pass-1 body joints); but give pass 2 its OWN decoder layer `self.decoder_layer_2 = _DecoderLayer(hidden_dim=256, num_heads=8, dropout=0.1)` with fresh weights, so pass 1 and pass 2 can specialize to the "rough pose" and "local refinement" sub-tasks respectively.

**Starting Point:** `baseline/`

---

## Overview

Designs 001 and 002 share the same `self.decoder_layer` between passes 1 and 2. Design 003 gives pass 2 its own `_DecoderLayer` instance (`self.decoder_layer_2`) while keeping the rest of the Design 002 setup intact:

- Two-pass forward (from Designs 001/002).
- Zero-init `coord_enc` providing coordinate feedback (same as Designs 001/002).
- Residual joint output `joints_final = joints_1 + joints_residual` (same as Designs 001/002).
- Pelvis depth/UV read from pass-2 token 0 (same as Designs 001/002).
- Intermediate supervision on `joints_initial` with weight 0.5 (same as Design 002).
- **Different from Designs 001/002:** pass 2 uses `self.decoder_layer_2`, not `self.decoder_layer`.

### Why independent decoder weights

idea001 (multi-layer decoder) regressed pelvis MPJPE by +14 to +19 mm because the extra self-attention over all 70 queries over-specialized the pelvis token 0 away from the absolute-regression task. With independent pass-2 weights, `self.decoder_layer_2` can learn different self-attention patterns — e.g., attention specialized for local refinement (smaller effective receptive field) and for re-concentrating pelvis-token features under the new residual objective, rather than simply repeating pass 1's attention pattern. The residual output formulation still makes the task easier than regressing from scratch: pass 2 only has to predict a small correction.

Parameter cost of one extra `_DecoderLayer` at `hidden_dim=256, num_heads=8`:
- 2× MultiheadAttention(256, 8) ≈ 2·(4·256·256) = 524,288 weights
- FFN: 256·1024 + 1024·256 ≈ 524,288 weights
- LayerNorms and biases: small
- **Total: ~1.1-1.3M additional parameters.** Well within 1080 Ti budget.

Plus the `coord_enc` (~67K). Overall overhead <0.5% relative to the 0.3B-parameter Sapiens backbone.

---

## Files to Change

1. `pose3d_transformer_head.py` — add `coord_enc`, conditionally build `self.decoder_layer_2`, modify `forward()` to use `self.decoder_layer_2` on pass 2 when `shared_decoder=False`, enable the intermediate supervision loss term.
2. `config.py` — add `num_refine_passes=2`, `shared_decoder=False`, `intermediate_supervision_weight=0.5` head kwargs.
3. `pelvis_utils.py` — **no change**.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. Imports

No new imports required.

#### 2. `Pose3dTransformerHead.__init__` — new parameters and second decoder layer

Add the same three kwargs as Designs 001/002:

```python
num_refine_passes: int = 1,
shared_decoder: bool = True,
intermediate_supervision_weight: float = 0.0,
```

Store as attributes:

```python
self.num_refine_passes = num_refine_passes
self.shared_decoder = shared_decoder
self.intermediate_supervision_weight = intermediate_supervision_weight
```

Build `coord_enc` unconditionally (same as Designs 001/002):

```python
self.coord_enc = nn.Sequential(
    nn.Linear(3, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
)
```

**NEW (Design 003):** conditionally build `self.decoder_layer_2` when `shared_decoder=False` AND `num_refine_passes >= 2`:

```python
if (not self.shared_decoder) and self.num_refine_passes >= 2:
    self.decoder_layer_2 = _DecoderLayer(hidden_dim, num_heads, dropout)
```

Placement: AFTER `self.decoder_layer = _DecoderLayer(...)` and AFTER `self.coord_enc = ...`, BEFORE `self.joints_out = nn.Linear(...)`.

Constraints:
- `self.decoder_layer_2` uses the SAME constructor args as `self.decoder_layer`: `_DecoderLayer(hidden_dim, num_heads, dropout)` with hidden_dim=256, num_heads=8, dropout=0.1. Do NOT change its internal structure.
- Its weights are initialized with PyTorch defaults (MultiheadAttention uses Xavier/Kaiming internally). No custom init is applied here; the baseline `_init_head_weights` covers only the output projections and query embeddings, not the `_DecoderLayer` internals. This matches the baseline behavior for `self.decoder_layer`.
- The `_init_head_weights` method itself does NOT need to reference `self.decoder_layer_2` — decoder layer init is not custom.
- When `shared_decoder=True` (Designs 001/002), `self.decoder_layer_2` is NOT built and MUST NOT be referenced.

#### 3. `_init_head_weights` — same as Designs 001/002

Identical to Designs 001/002. Zero-init the last Linear of `coord_enc`:

```python
def _init_head_weights(self) -> None:
    nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
    for m in [self.joints_out, self.depth_out, self.uv_out]:
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    nn.init.trunc_normal_(self.coord_enc[0].weight, std=0.02)
    if self.coord_enc[0].bias is not None:
        nn.init.zeros_(self.coord_enc[0].bias)
    nn.init.zeros_(self.coord_enc[2].weight)
    if self.coord_enc[2].bias is not None:
        nn.init.zeros_(self.coord_enc[2].bias)
```

Note: with independent pass-2 weights, the pass-2 decoder layer's attention/FFN weights are randomly initialized (not copied from pass 1). At init, `decoded_2` is therefore NOT equal to `decoded_1` — the baseline-matching property (Design 001) holds only on the `coord_enc` path. However, `joints_residual = joints_out(decoded_2)` will start near random-small because `joints_out` has std=0.02 init, so the residual joints correction is still small at init (~O(0.02·||decoded_2||)). Combined with the direct supervision on `joints_initial` (Design 003's intermediate loss), this prevents pass 1 from drifting during early training.

#### 4. `forward()` — use `self.decoder_layer_2` on pass 2

Replace the existing `forward()` body (same structure as Designs 001/002, with the branch on `shared_decoder`):

```python
def forward(
    self, feats: Tuple[torch.Tensor, ...]
) -> Dict[str, torch.Tensor]:
    feat = feats[-1]
    B, C, H, W = feat.shape

    spatial = feat.flatten(2).transpose(1, 2)
    spatial = self.input_proj(spatial)
    pos_enc = self._get_pos_enc(H, W, feat.device)
    spatial = spatial + pos_enc

    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

    # Pass 1
    decoded_1 = self.decoder_layer(queries, spatial)
    joints_1 = self.joints_out(decoded_1)

    if self.num_refine_passes <= 1:
        pelvis_token = decoded_1[:, 0, :]
        pelvis_depth = self.depth_out(pelvis_token)
        pelvis_uv = self.uv_out(pelvis_token)
        return {
            'joints': joints_1,
            'joints_initial': joints_1,
            'pelvis_depth': pelvis_depth,
            'pelvis_uv': pelvis_uv,
        }

    # Refinement pass(es)
    joints_cur = joints_1
    decoded_cur = decoded_1
    for _ in range(self.num_refine_passes - 1):
        queries_next = decoded_cur + self.coord_enc(joints_cur)
        if self.shared_decoder:
            decoded_next = self.decoder_layer(queries_next, spatial)
        else:
            decoded_next = self.decoder_layer_2(queries_next, spatial)
        joints_residual = self.joints_out(decoded_next)
        joints_cur = joints_cur + joints_residual
        decoded_cur = decoded_next

    joints_final = joints_cur
    pelvis_token = decoded_cur[:, 0, :]
    pelvis_depth = self.depth_out(pelvis_token)
    pelvis_uv = self.uv_out(pelvis_token)

    return {
        'joints': joints_final,
        'joints_initial': joints_1,
        'pelvis_depth': pelvis_depth,
        'pelvis_uv': pelvis_uv,
    }
```

Constraints:
- With `shared_decoder=False`, the loop's `else` branch is taken on pass 2, so `decoded_next = self.decoder_layer_2(queries_next, spatial)`.
- With `num_refine_passes=2`, the loop iterates exactly once (`num_refine_passes - 1 == 1`), so `self.decoder_layer_2` is called exactly once per forward.
- The `joints_out` head is shared for both pass 1 and pass 2 residual readouts (single set of weights). This is consistent with Deformable-DETR's recipe: iterative refinement shares regression-head weights across layers even when the transformer layers themselves are independent.
- For future designs that may use `num_refine_passes > 2` with independent weights, only ONE extra layer (`self.decoder_layer_2`) is built here; such extensions are out of scope for Design 003.

#### 5. `loss()` — intermediate supervision enabled

Identical to Design 002. The config sets `intermediate_supervision_weight=0.5`, so:

```python
_BODY = list(range(0, 22))
losses = dict()
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])

if self.intermediate_supervision_weight > 0.0 and 'joints_initial' in pred:
    losses['loss/joints_init/train'] = (
        self.intermediate_supervision_weight *
        self.loss_joints_module(
            pred['joints_initial'][:, _BODY],
            gt_joints[:, _BODY]))

losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
    pred['pelvis_depth'], gt_depth)
losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
    pred['pelvis_uv'], gt_uv)
```

`torch.no_grad()` MPJPE recording block unchanged.

#### 6. `predict()` — no change

Body unchanged. Reads `pred['joints']` = refined final joints.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block, add the three new kwargs (note `shared_decoder=False`):

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    num_refine_passes=2,
    shared_decoder=False,
    intermediate_supervision_weight=0.5,
),
```

All other config values identical to baseline.

---

## Exact Config Values (unchanged from baseline except three head kwargs)

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.03 |
| backbone lr_mult | 0.1 |
| clip_grad max_norm | 1.0 |
| accumulative_counts | 8 (effective batch 32) |
| LR schedule | LinearLR (epoch 0-3, start_factor=0.333) + CosineAnnealingLR (epoch 3-20, eta_min=0), both convert_to_iter_based=True |
| seed | 2026 |
| batch_size | 4 |
| hidden_dim | 256 |
| num_heads | 8 |
| dropout | 0.1 |
| loss_joints loss_weight | 1.0 |
| loss_depth loss_weight | 1.0 (× loss_weight_depth=1.0) |
| loss_uv loss_weight | 1.0 (× loss_weight_uv=1.0) |
| **num_refine_passes** | **2 (new)** |
| **shared_decoder** | **False (new)** |
| **intermediate_supervision_weight** | **0.5 (new)** |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only (both `loss/joints/train` and `loss/joints_init/train`).
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present.
4. No Python `import` statements in `config.py` — `num_refine_passes=2`, `shared_decoder=False`, `intermediate_supervision_weight=0.5` are int/bool/float literals.
5. Head file uses ABSOLUTE imports.
6. `coord_enc` last Linear weight AND bias MUST both be zero-initialized.
7. `self.decoder_layer_2` is ONLY built when `shared_decoder=False` AND `num_refine_passes >= 2`. Conversely, when `shared_decoder=True`, `self.decoder_layer_2` MUST NOT exist (to preserve Design-001/002 behaviour and avoid loading unused params into the optimizer).
8. `self.decoder_layer_2` uses the SAME `_DecoderLayer(hidden_dim, num_heads, dropout)` constructor as `self.decoder_layer`, with identical args (256, 8, 0.1) — do NOT alter its internal structure.
9. `joints_out` is SHARED between pass 1 and pass 2 residual readout — do NOT create a second linear.
10. `self.num_refine_passes=1` must produce baseline-equivalent forward behaviour (short-circuit to single-pass path).
11. Residual formulation: `joints_final = joints_1 + joints_residual`, not the absolute pass-2 `joints_out(decoded_2)`.
12. Do NOT `.detach()` any tensor in the forward path, including `joints_1` fed into `coord_enc`.
13. Default values for the three new `__init__` kwargs MUST be `num_refine_passes=1, shared_decoder=True, intermediate_supervision_weight=0.0`, so omitting them reproduces baseline behaviour.
14. `forward()` output dict keys: `joints`, `joints_initial`, `pelvis_depth`, `pelvis_uv`.
15. `predict()` body unchanged.
16. Intermediate loss key is EXACTLY `'loss/joints_init/train'`.
17. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, `BedlamMPJPEMetric` untouched — they see `pred['joints']` = refined final joints with shape `(B, 70, 3)`.
18. No changes to `bedlam_metric.py`, dataset, transforms, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`.
19. No changes to `pelvis_utils.py`.
20. Optimizer `paramwise_cfg` is unchanged — the only `custom_keys` entry is `'backbone': dict(lr_mult=0.1)`. The new `coord_enc` and `decoder_layer_2` parameters fall under the default head LR (1e-4), which is intentional and consistent with how `self.decoder_layer`, `self.joint_queries`, and the output projections are treated.

---

## Expected Behaviour After Change

- `forward()` runs pass 1 through `self.decoder_layer` and pass 2 through the NEW `self.decoder_layer_2`, each of which is a full decoder layer (self-attn → cross-attn → FFN). Per-batch wall-time increases by ~35-40% over baseline (one extra full decoder layer worth of attention/FFN compute, plus the `coord_enc` projection). Still well within the 20-epoch budget on 1080 Ti.
- Training emits FOUR loss scalars: `loss/joints/train`, `loss/joints_init/train`, `loss/depth/train`, `loss/uv/train`.
- At init, `decoded_2 ≠ decoded_1` (because `self.decoder_layer_2` has randomly-initialized weights distinct from `self.decoder_layer`), but `coord_enc(joints_1) == 0` so `queries_2 = decoded_1 + 0 = decoded_1`. The second-pass attention/FFN then transforms `decoded_1` with its own weights, producing a different `decoded_2`. However, `joints_residual = joints_out(decoded_2)` still uses the shared `joints_out` head (std=0.02 init), so `joints_residual` stays small near init even though `decoded_2` differs from `decoded_1`. Combined with intermediate supervision on `joints_1`, pass 1 continues to receive direct training signal, preventing collapse.
- Validation metrics computed on `pred['joints']` (refined final joints) — unchanged pipeline.
- `MetricsCSVHook` CSV schema unchanged.
- Parameter count increase over baseline: ~67K (coord_enc) + ~1.1-1.3M (extra decoder layer) ≈ **~1.2M additional params**.
- Expected result vs. baseline (`composite_val ~168.7`): given the extra capacity, this design has the strongest body MPJPE improvement potential of the three (target `< 145`) and may also help `mpjpe_pelvis_val` (unlike Design 002, the pass-2 self-attention is free to re-specialize token 0 for absolute regression — target `< 172`). `mpjpe_abs_val < 410`. `composite_val < 155`.
- Risk: extra capacity could cause overfitting within the 20-epoch budget. Intermediate supervision on pass 1 mitigates this by providing a strong regularizing signal that prevents pass 2 from dominating pass 1's gradients.
- At inference, `self.decoder_layer_2` is called once per forward; no other change.
