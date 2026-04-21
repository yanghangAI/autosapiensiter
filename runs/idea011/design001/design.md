# Design 001 — Two-pass Coordinate-Conditioned Decoder (Shared Weights, No Intermediate Supervision)

**Design Description:** Run the single `_DecoderLayer` twice with the *same* weights; after pass 1, project the first-pass joint predictions `(B, 70, 3)` through a zero-init two-layer MLP `coord_enc: Linear(3,256)→GELU→Linear(256,256)` and add the result to the first-pass hidden states; the second pass outputs *residuals* that are summed with the first-pass joints (`joints_final = joints_1 + joints_residual`), and pelvis depth/UV are read from the refined (pass-2) token 0. No intermediate supervision; shared decoder weights only; everything else identical to baseline.

**Starting Point:** `baseline/`

---

## Overview

The baseline head runs a single decoder pass producing `(B, 70, 3)` joints and a pelvis head on token 0. This design adds an **iterative refinement** mechanism modeled after Deformable-DETR / DAB-DETR:

1. Pass 1: existing decoder consumes learnable queries and spatial tokens, emits hidden states `decoded_1`.
2. Read *initial* joint coordinates `joints_1 = joints_out(decoded_1)` → `(B, 70, 3)`.
3. Coordinate encoder `coord_enc` maps those coordinates to a feature of shape `(B, 70, hidden_dim)`; its last `Linear` is zero-initialized (weight and bias) so `coord_enc(joints_1)` starts as 0 and the refined queries match `decoded_1` at init.
4. Refined query input `queries_2 = decoded_1 + coord_enc(joints_1)`.
5. Pass 2: **same** `self.decoder_layer` invoked again on `(queries_2, spatial)` → `decoded_2`.
6. Read residual coordinates `joints_residual = joints_out(decoded_2)` and final coordinates `joints_final = joints_1 + joints_residual`.
7. Read pelvis depth/UV from `decoded_2[:, 0, :]` (pass-2 token 0).
8. `forward()` returns `{'joints': joints_final, 'joints_initial': joints_1, 'pelvis_depth', 'pelvis_uv'}`.
9. `loss()` computes the standard body-joint loss on `pred['joints']` (= final), plus the unchanged pelvis depth/UV losses. `joints_initial` is returned for API symmetry with later designs but is NOT supervised in Design 001.
10. `predict()` uses `pred['joints']` exactly as before.

All other components (optimizer, LR schedule, data pipeline, seed, batch size, accumulation, backbone, data preprocessor, evaluation, hooks) are identical to the baseline. Total extra parameter count: `coord_enc` has `3*256 + 256 + 256*256 + 256 ≈ 66,816` params; no extra decoder layer (shared weights).

---

## Files to Change

1. `pose3d_transformer_head.py` — add `coord_enc` module, modify `forward()` to run two passes and compute residual output, extend `loss()`/`predict()` to match.
2. `config.py` — add `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0` head kwargs.
3. `pelvis_utils.py` — **no change**.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. Imports

No new imports required. Existing imports (`torch`, `torch.nn`, etc.) are sufficient.

#### 2. `Pose3dTransformerHead.__init__` — new parameters

Add three kwargs to the `__init__` signature, placed immediately after `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`:

```python
num_refine_passes: int = 1,
shared_decoder: bool = True,
intermediate_supervision_weight: float = 0.0,
```

Store them as attributes:

```python
self.num_refine_passes = num_refine_passes
self.shared_decoder = shared_decoder
self.intermediate_supervision_weight = intermediate_supervision_weight
```

Constraints:
- Default `num_refine_passes=1` preserves baseline behaviour exactly (only the first pass runs; no coord_enc use).
- Design 001 sets `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0` via config.

Build the `coord_enc` module **unconditionally** (it is cheap and its zero-init guarantees it is a no-op if unused):

```python
self.coord_enc = nn.Sequential(
    nn.Linear(3, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, hidden_dim),
)
```

This construction lives AFTER `self.decoder_layer = _DecoderLayer(...)` and BEFORE `self.joints_out = nn.Linear(...)`.

**Do NOT** build a second decoder layer in Design 001 (`shared_decoder=True`). The second pass reuses `self.decoder_layer`.

#### 3. `_init_head_weights` — zero-init the last layer of `coord_enc`

Extend `_init_head_weights` so that after the existing initializations, the FIRST linear in `coord_enc` uses truncated normal (std=0.02) and the SECOND (last) linear's weight AND bias are zero:

```python
def _init_head_weights(self) -> None:
    # Query embeddings
    nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
    # Output projections
    for m in [self.joints_out, self.depth_out, self.uv_out]:
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # Coordinate encoder: trunc_normal on first Linear, zero-init on last Linear
    nn.init.trunc_normal_(self.coord_enc[0].weight, std=0.02)
    if self.coord_enc[0].bias is not None:
        nn.init.zeros_(self.coord_enc[0].bias)
    nn.init.zeros_(self.coord_enc[2].weight)
    if self.coord_enc[2].bias is not None:
        nn.init.zeros_(self.coord_enc[2].bias)
```

The indices `[0]` and `[2]` correspond to the two `nn.Linear` modules inside the `nn.Sequential` (index 1 is `nn.GELU()`).

Constraints:
- Zero-init MUST apply to BOTH weight and bias of `self.coord_enc[2]`. This guarantees `coord_enc(joints_1) == 0` at initialization, so the refined queries `queries_2 = decoded_1 + 0 = decoded_1` and the second decoder pass reproduces the first pass's hidden states exactly at init — i.e., the network begins training in an approximately baseline-equivalent state.

#### 4. `forward()` — two-pass decoder with residual output

Replace the existing `forward()` body. The new logic (keeping the preamble unchanged up through `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)`):

```python
def forward(
    self, feats: Tuple[torch.Tensor, ...]
) -> Dict[str, torch.Tensor]:
    feat = feats[-1]  # (B, C, H, W)
    B, C, H, W = feat.shape

    spatial = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
    spatial = self.input_proj(spatial)          # (B, H*W, hidden_dim)
    pos_enc = self._get_pos_enc(H, W, feat.device)
    spatial = spatial + pos_enc

    queries = self.joint_queries.weight.unsqueeze(0).expand(
        B, -1, -1)  # (B, num_joints, hidden_dim)

    # ── Pass 1 ───────────────────────────────────────────────────────
    decoded_1 = self.decoder_layer(queries, spatial)  # (B, J, D)
    joints_1 = self.joints_out(decoded_1)             # (B, J, 3)

    if self.num_refine_passes <= 1:
        # Baseline-equivalent single-pass path
        pelvis_token = decoded_1[:, 0, :]
        pelvis_depth = self.depth_out(pelvis_token)
        pelvis_uv = self.uv_out(pelvis_token)
        return {
            'joints': joints_1,
            'joints_initial': joints_1,
            'pelvis_depth': pelvis_depth,
            'pelvis_uv': pelvis_uv,
        }

    # ── Refinement pass(es) ──────────────────────────────────────────
    # Currently only num_refine_passes=2 is exercised; the loop below
    # generalises to >2 passes if a future design wants it.
    joints_cur = joints_1
    decoded_cur = decoded_1
    for _ in range(self.num_refine_passes - 1):
        queries_next = decoded_cur + self.coord_enc(joints_cur)  # (B, J, D)
        if self.shared_decoder:
            decoded_next = self.decoder_layer(queries_next, spatial)
        else:
            decoded_next = self.decoder_layer_2(queries_next, spatial)
        joints_residual = self.joints_out(decoded_next)          # (B, J, 3)
        joints_cur = joints_cur + joints_residual
        decoded_cur = decoded_next

    joints_final = joints_cur
    pelvis_token = decoded_cur[:, 0, :]  # pass-2 token 0
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
- `self.decoder_layer_2` is only referenced when `shared_decoder=False`; in Design 001 `shared_decoder=True`, so the `else` branch is never hit and need not exist (but for shared code across the three designs it is written here; Builder must still create `self.decoder_layer_2` in `__init__` when `shared_decoder=False` — see Design 003).
- The `if self.num_refine_passes <= 1` short-circuit MUST be kept so that setting `num_refine_passes=1` reproduces baseline behaviour bit-identically (aside from the unused `coord_enc` parameters, which do not participate in the forward compute path).
- `joints_out` is SHARED between pass 1 and the residual readout of pass 2. This is intentional and matches Deformable-DETR's standard iterative refinement recipe (single set of regression head weights, shared across passes).
- The residual sum `joints_cur + joints_residual` is a differentiable torch op; gradient flows back through both `joints_1` (via `coord_enc → queries_2 → decoded_2 → joints_residual`, and directly via the `+` connection) and `joints_residual` itself.

#### 5. `loss()` — compute body-joint loss on final joints, keep pelvis unchanged

Inside `Pose3dTransformerHead.loss`, REPLACE the single body-joint loss line with the logic below. The gt extraction block (starting with `gt_joints = torch.cat(...)`) is unchanged; only the `_BODY = list(range(0, 22))` section onward changes:

```python
_BODY = list(range(0, 22))
losses = dict()
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])

# Optional intermediate supervision (Design 001: weight=0.0, disabled).
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
- In Design 001, `self.intermediate_supervision_weight == 0.0`, so the `loss/joints_init/train` branch is NOT taken. The losses dict contains exactly the same keys as the baseline: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`.
- The guard `'joints_initial' in pred` is defensive — Design 001's forward always emits it, but keeping the guard lets the `num_refine_passes=1` path skip the intermediate loss cleanly.

Keep the `with torch.no_grad(): ...` block (train-time MPJPE recording) UNCHANGED — it reads `pred['joints']`, which is now the refined final joints.

#### 6. `predict()` — no change

`predict()` already reads `pred['joints']`. Because `forward()` now writes `joints_final` to `pred['joints']`, `predict()` automatically returns the refined predictions. The extra `pred['joints_initial']` key is ignored by `predict()` — it is only used by `loss()` for intermediate supervision (disabled in Design 001).

No changes to the body of `predict()`.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model`, add the three new kwargs at the end (after `loss_weight_uv=1.0,`):

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
    intermediate_supervision_weight=0.0,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights, `custom_imports` list) are identical to the baseline.

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
| **intermediate_supervision_weight** | **0.0 (new)** |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only (`_BODY = list(range(0, 22))`). This applies to both the main `loss/joints/train` term and (in future designs) the `loss/joints_init/train` term.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0` are int/bool/float literals.
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package).
6. `coord_enc` last Linear weight AND bias MUST both be zero-initialized. If either is nonzero at init, the refined queries will diverge from `decoded_1`, the second pass will operate on unfamiliar inputs, and training may be unstable or regress for the first several epochs.
7. `joints_out` is shared between pass 1 and pass 2 residual readout — do NOT create a second linear for the second pass.
8. `self.num_refine_passes=1` must produce baseline-equivalent forward behaviour: pass 1 is identical, pelvis is read from `decoded_1[:, 0, :]`, and `joints_residual` is never computed.
9. Residual formulation: the final `joints` output MUST be `joints_1 + joints_residual`, not the absolute pass-2 `joints_out(decoded_2)`. Using the absolute prediction instead of the residual loses the zero-init baseline-match property.
10. Do NOT `.detach()` any tensor in the forward path. In particular, `joints_1` must NOT be detached when fed into `coord_enc` — the first pass needs gradient from the final loss.
11. Default values for the three new `__init__` kwargs MUST be `num_refine_passes=1, shared_decoder=True, intermediate_supervision_weight=0.0`, so omitting them reproduces baseline behaviour exactly.
12. `forward()` output dict MUST contain keys `joints`, `pelvis_depth`, `pelvis_uv` (existing); ADDING `joints_initial` is required and backward-compatible because `predict()` and downstream code read only the three existing keys.
13. `predict()` body is unchanged — it reads `pred['joints']` (now the refined final joints) and builds `InstanceData`. Do not add `joints_initial` to the predictions.
14. `MetricsCSVHook`, `TrainMPJPEAveragingHook`, and `BedlamMPJPEMetric` are untouched — they see the refined joints as `pred['joints']` with shape `(B, 70, 3)`, matching baseline.
15. No changes to `bedlam_metric.py`, dataset, transforms, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
16. No changes to `pelvis_utils.py`.

---

## Expected Behaviour After Change

- `forward()` runs two sequential calls to `self.decoder_layer` instead of one. Per-batch wall-time increases by ≈25-30% (a single decoder layer with `hidden_dim=256, num_heads=8` is small relative to the ViT backbone). Still well within the 20-epoch budget on 1080 Ti.
- Training emits the same three loss scalars as baseline (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`). No new loss keys are added in Design 001.
- At init, `coord_enc(joints_1) == 0`, so `decoded_2 ≈ decoded_1` (up to attention dropout stochasticity) and `joints_residual ≈ 0`. Training starts in an approximately baseline-equivalent state and gradually learns to use coordinate feedback.
- Validation metrics (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`) are computed by the unchanged `BedlamMPJPEMetric` on `pred['joints'] = joints_final`. No change to evaluation.
- `MetricsCSVHook` writes the same columns as before.
- Extra parameter count: `coord_enc` = 3·256 + 256 + 256·256 + 256 ≈ **66.8K params** (negligible vs. 0.3B Sapiens backbone).
- Expected result vs. baseline (`composite_val ~168.7`, `mpjpe_body ~160`, `mpjpe_pelvis ~176`, `mpjpe_abs ~455`): `mpjpe_body_val` improves meaningfully (target `< 150`), `mpjpe_abs_val` improves (target `< 430`), `mpjpe_pelvis_val` expected neutral to mildly positive. Target `composite_val < 160`.
- At inference, the two-pass logic runs identically; no special handling needed. The output shape `(B, 70, 3)` is unchanged.
