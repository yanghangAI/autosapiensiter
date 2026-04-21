# Design 001 — Body-only decoder, hand outputs zero-padded (diagnostic)

**Design Description:** Replace 70-query decoder with 22-query body-only decoder; zero-pad output to (B, 70, 3) with no-gradient padding.

**Starting Point:** `baseline/`

---

## Overview

This is the diagnostic variant. The core algorithm change: reduce the joint query set from 70 to 22 (body joints only), so that decoder self-attention and cross-attention operate exclusively over evaluated body joints. The decoder self-attention shrinks from 70×70 = 4,900 to 22×22 = 484 elements (90% reduction). Cross-attention shrinks from (70, 960) to (22, 960) rows (69% reduction). Hand joints (indices 22–69) are zero-padded after decoding and never receive gradients. Any metric change is attributable purely to removing hand query contamination from the decoder algorithm.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Constructor (`__init__`):**

Add `num_body_queries: int = 22` as a new kwarg after `dropout`. Store as `self.num_body_queries = num_body_queries`.

Change the joint query embedding from:
```python
self.joint_queries = nn.Embedding(num_joints, hidden_dim)
```
to:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

All other constructor arguments and module definitions remain identical to baseline (`input_proj`, `decoder_layer`, `joints_out`, `depth_out`, `uv_out`).

Updated `_init_head_weights` is unchanged — it still initialises `self.joint_queries.weight` with `nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)`.

**`forward()` method:**

Change the query broadcast line from:
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_joints, hidden_dim)
```
to:
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_body_queries, hidden_dim)
```

The decoder produces:
```python
decoded = self.decoder_layer(queries, spatial)  # (B, 22, hidden_dim)
```

`joints_out` then gives:
```python
body_joints = self.joints_out(decoded)  # (B, 22, 3)
```

After `body_joints`, zero-pad to full 70-joint tensor:
```python
pad = torch.zeros(B, self.num_joints - self.num_body_queries, 3,
                  device=body_joints.device, dtype=body_joints.dtype)
joints = torch.cat([body_joints, pad], dim=1)  # (B, 70, 3)
```

`pelvis_token` is `decoded[:, 0, :]` — unchanged.

The returned dict has the same keys as baseline: `{'joints': joints, 'pelvis_depth': pelvis_depth, 'pelvis_uv': pelvis_uv}` with `joints.shape == (B, 70, 3)`.

**`loss()` method:**

No change to loss computation. The body joint loss already restricts to `_BODY = list(range(0, 22))`:
```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```
The zero-padded region (indices 22–69) is never referenced in loss or metrics.

`self._train_mpjpe` and `self._train_mpjpe_abs` computations are unchanged.

**`predict()` method:**

The `predict()` method references `self.num_joints` in the keypoint_scores line:
```python
inst.keypoint_scores = torch.ones(1, self.num_joints, dtype=torch.float32).numpy()
```
This remains correct because `self.num_joints = 70` and `joints.shape[1] == 70` after zero-padding.

**Full constructor signature after change:**
```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_body_queries: int = 22,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
    loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                  beta=0.05, loss_weight=1.0),
    loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                               beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    init_cfg: OptConfigType = None,
):
```

### 2. `config.py`

In the `model.head` dict, add `num_body_queries=22` as a literal integer after `num_joints`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_body_queries=22,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                     loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                    loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                 loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, backbone) are **identical to baseline**.

### 3. `pelvis_utils.py`

No changes.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. `self.num_joints = 70` must remain set in `__init__` so `predict()` produces correct-shape keypoint_scores.
3. The zero-pad tensor must have `requires_grad=False` (default for `torch.zeros` — no special action needed).
4. Output `joints` tensor shape must be `(B, 70, 3)` before returning from `forward()`.
5. MMEngine config: `num_body_queries=22` is an integer literal — no import required. Compliant.
6. Body joint loss indices `_BODY = list(range(0, 22))` — do not change.
7. Pelvis token is `decoded[:, 0, :]` (first of the 22 body queries) — unchanged.
8. `_DecoderLayer` is unchanged. No modification to self-attention or cross-attention internals.
9. Backbone, data preprocessor, metric, transforms are invariant — do not touch.
10. Seed `2026`, batch size `4`, accumulation `8` — do not change.

---

## Expected Behavior After Change

- Self-attention operates on 22×22 query pairs (vs. 70×70 baseline). All 22 queries correspond to evaluated body joints; gradients from `loss/joints/train` flow to every active query.
- Cross-attention matrix has 22 rows × 960 spatial tokens (vs. 70 × 960 baseline).
- Hand indices 22–69 in the output tensor are identically zero and detached; they contribute zero gradient.
- Body MPJPE (`mpjpe_body_val`) expected to decrease 8–15 mm vs. baseline.
- Pelvis MPJPE (`mpjpe_pelvis_val`) expected to maintain or improve (pelvis token only negotiates with 21 body joint queries, not 48 hand queries).
- `composite_val` target: < 158 (vs. baseline 169.75).
- All downstream metric code expecting shape `(B, 70, 3)` receives correct shape.
