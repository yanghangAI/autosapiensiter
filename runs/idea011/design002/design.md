# Design 002 — Two-pass Coordinate-Conditioned Decoder with Intermediate Supervision (Shared Weights)

**Design Description:** Identical architecture to Design 001 (two-pass shared-weight decoder with a zero-init `coord_enc: Linear(3,256)→GELU→Linear(256,256)` feeding pass-1 coordinates back into pass-2 queries, residual joint output, pelvis depth/UV read from pass-2 token 0), but add a direct supervised body-joint loss on the pass-1 output `joints_initial` with weight `intermediate_supervision_weight=0.5`; pelvis supervision remains only on the pass-2 outputs.

**Starting Point:** `baseline/`

---

## Overview

Design 002 builds on Design 001's two-pass architecture and adds **intermediate supervision** on the pass-1 joint output. Rationale: without direct supervision, pass 1 could learn to be a "noise generator" that pass 2 cleans up, hampering early-training convergence (the refinement head would have to simultaneously drive both passes through a single loss signal at the pass-2 output). Intermediate supervision is standard practice in Deformable-DETR / DAB-DETR and is known to accelerate convergence and stabilize training in iterative refinement setups.

Concretely:

- `forward()` is IDENTICAL to Design 001 — returns `{'joints', 'joints_initial', 'pelvis_depth', 'pelvis_uv'}`. Two-pass shared-weight decoder with residual joint output.
- `loss()` additionally computes `loss/joints_init/train = 0.5 * SoftWeightSmoothL1(joints_initial[:, 0-21], gt_joints[:, 0-21])` on top of the standard `loss/joints/train`, `loss/depth/train`, `loss/uv/train` terms.
- Pelvis depth and UV losses are applied ONLY on the pass-2 outputs (i.e., only `loss/depth/train` and `loss/uv/train` on the refined pelvis). There is no intermediate pelvis supervision because:
  - The coord_enc only maps the 70×3 joint coordinates back, not pelvis depth/UV. Pelvis information is not cleanly split between the passes.
  - The pelvis head reads pass-2 token 0 only; there is no "initial pelvis" prediction to supervise.
- `predict()` returns `pred['joints']` (the refined final joints) as before.

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are unchanged from the baseline. Parameter count overhead same as Design 001 (~66.8K for `coord_enc`).

---

## Files to Change

1. `pose3d_transformer_head.py` — add `coord_enc`, modify `forward()` (same as Design 001), modify `loss()` to enable the `loss/joints_init/train` term.
2. `config.py` — add `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.5` head kwargs.
3. `pelvis_utils.py` — **no change**.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. Imports

No new imports required.

#### 2. `Pose3dTransformerHead.__init__` — new parameters

**Identical to Design 001.** Add three kwargs to `__init__` signature (after `loss_weight_uv`, before `init_cfg`):

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

Build `coord_enc` unconditionally:

```python
self.coord_enc = nn.Sequential(
    nn.Linear(3, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
)
```

Placed AFTER `self.decoder_layer = _DecoderLayer(...)` and BEFORE `self.joints_out = nn.Linear(...)`. Do NOT build a second decoder layer in Design 002 (`shared_decoder=True`).

#### 3. `_init_head_weights` — zero-init last layer of `coord_enc`

**Identical to Design 001.** Extend `_init_head_weights`:

```python
def _init_head_weights(self) -> None:
    nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
    for m in [self.joints_out, self.depth_out, self.uv_out]:
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # Coordinate encoder
    nn.init.trunc_normal_(self.coord_enc[0].weight, std=0.02)
    if self.coord_enc[0].bias is not None:
        nn.init.zeros_(self.coord_enc[0].bias)
    nn.init.zeros_(self.coord_enc[2].weight)
    if self.coord_enc[2].bias is not None:
        nn.init.zeros_(self.coord_enc[2].bias)
```

#### 4. `forward()` — two-pass decoder with residual output

**Identical to Design 001.** Replace the existing `forward()` body with the two-pass logic. Reproduced here for self-containment:

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

#### 5. `loss()` — enable intermediate supervision

Inside `Pose3dTransformerHead.loss`, REPLACE the body-joint loss block and keep pelvis losses unchanged. Same logic as Design 001, but in Design 002 the config sets `intermediate_supervision_weight=0.5` so the `loss/joints_init/train` branch IS taken:

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

Constraints:
- The intermediate supervision term uses the SAME `self.loss_joints_module` (shared SoftWeightSmoothL1Loss), since that module is stateless at inference and shareable between calls.
- The intermediate term uses the SAME joint index set `_BODY = list(range(0, 22))` as the main joint loss (body joints only).
- The multiplier `self.intermediate_supervision_weight` is applied BEFORE the loss module (identical convention to how `self.loss_weight_depth` is used for the depth loss).
- `torch.no_grad()` MPJPE recording block UNCHANGED (reads `pred['joints']` = final joints).

#### 6. `predict()` — no change

Body unchanged from baseline. Reads `pred['joints']` (refined) and builds InstanceData per sample. `joints_initial` is ignored.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block, add the three new kwargs (note the difference from Design 001 — `intermediate_supervision_weight=0.5`):

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
    shared_decoder=True,
    intermediate_supervision_weight=0.5,
),
```

All other config values identical to baseline (optimizer, LR, schedule, data pipeline, hooks, batch size, seed, pretrained, `custom_imports`).

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
| **shared_decoder** | **True (new)** |
| **intermediate_supervision_weight** | **0.5 (new)** |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only (both `loss/joints/train` and `loss/joints_init/train`).
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present.
4. No Python `import` statements in `config.py` — `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.5` are int/bool/float literals.
5. Head file uses ABSOLUTE imports.
6. `coord_enc` last Linear weight AND bias MUST both be zero-initialized.
7. `joints_out` is shared between pass 1 and pass 2 residual readout — do NOT create a second linear.
8. `self.num_refine_passes=1` must produce baseline-equivalent forward behaviour.
9. Residual formulation: final `joints` output MUST be `joints_1 + joints_residual`.
10. Do NOT `.detach()` any tensor in the forward path, including `joints_1` fed into `coord_enc`.
11. Default values for the three new `__init__` kwargs MUST be `num_refine_passes=1, shared_decoder=True, intermediate_supervision_weight=0.0`. Default `intermediate_supervision_weight=0.0` ensures omitting the kwarg disables the intermediate loss and reproduces baseline/Design-001 behaviour (depending on `num_refine_passes`).
12. `forward()` output dict keys: `joints`, `joints_initial`, `pelvis_depth`, `pelvis_uv`.
13. `predict()` body unchanged.
14. The intermediate loss key is EXACTLY `'loss/joints_init/train'` (using `_init` suffix to distinguish from `loss/joints/train`).
15. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, `BedlamMPJPEMetric` untouched. They see only `pred['joints']` = final joints.
16. No changes to `bedlam_metric.py`, dataset, transforms, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`.
17. No changes to `pelvis_utils.py`.
18. The intermediate loss uses `self.loss_joints_module` (NOT a fresh module instance). Building a second module would change initial state / RNG order and increase parameter count unnecessarily (all MMEngine-built "losses" in this head are stateless modules).

---

## Expected Behaviour After Change

- `forward()` behaviour identical to Design 001 (two-pass decoder, shared weights).
- Training emits FOUR loss scalars: `loss/joints/train`, `loss/joints_init/train` (new), `loss/depth/train`, `loss/uv/train`. The new term converges similarly in magnitude to the main `loss/joints/train` (both body-joint SoftWeight-SmoothL1 with beta=0.05) but scaled by 0.5.
- At init, `coord_enc(joints_1) == 0` so `joints_residual ≈ 0` and `joints_final ≈ joints_1`. The intermediate loss (0.5·L(joints_1)) and the main loss (L(joints_final)) are initially highly correlated because the two tensors are nearly identical. As training progresses and the residual signal grows, they decouple.
- Total effective body-joint loss weight at init ≈ 1.5× baseline (1.0·L(joints_final) + 0.5·L(joints_initial), with joints_final ≈ joints_initial). This is a mild effective LR boost on the body-joint term at early training, which is consistent with the intended "accelerate pass-1 convergence" effect.
- Validation metrics computed on `pred['joints']` (refined final joints) by `BedlamMPJPEMetric` — unchanged pipeline.
- `MetricsCSVHook` CSV schema unchanged.
- Per-iteration overhead: same as Design 001 (two decoder passes + coord_enc; ~25-30% wall-time increase over baseline). The extra loss term is a single SoftWeightSmoothL1 call on a `(B, 22, 3)` tensor — negligible (~<1 ms).
- Expected result vs. baseline (`composite_val ~168.7`, `mpjpe_body ~160`): `mpjpe_body_val` improves (target `< 148`, potentially stronger than Design 001 because pass 1 is directly supervised), `mpjpe_pelvis_val` neutral (pelvis supervision unchanged), `mpjpe_abs_val < 425`, `composite_val < 158`.
- At inference, only `pred['joints']` matters; `joints_initial` is ignored by `predict()`. No inference-time overhead relative to Design 001.
