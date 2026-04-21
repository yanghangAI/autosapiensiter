# Design 003 — Structured Spatial Token Dropout with Linear Annealing (p=0.30 → 0.10)

**Design Description:** Apply spatial token dropout starting at p=0.30 for epochs 1–10 and annealing linearly to p=0.10 for epochs 11–20, via a custom MMEngine hook (`SpatialDropAnnealHook`) registered in `pose3d_transformer_head.py`.

**Starting Point:** `baseline/`

---

## Overview

This design combines high early-training regularisation (p=0.30 forces broad spatial exploration) with lower late-training regularisation (p=0.10 allows precise spatial localisation). A custom hook `SpatialDropAnnealHook` updates `model.head.spatial_drop_prob` at the start of each epoch via linear interpolation. The hook is defined in `pose3d_transformer_head.py` (already in `custom_imports`) and registered with MMEngine's `HOOKS` registry.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` — **no changes**.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. Additional import at top of file

Add to the existing imports (after `from mmpose.registry import MODELS`):

```python
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
```

#### 2. `_DecoderLayer.forward` signature change

Add a `spatial_drop_prob: float = 0.0` keyword argument (identical to designs 001/002):

```
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            spatial_drop_prob: float = 0.0) -> torch.Tensor:
```

#### 3. Cross-attention key_padding_mask logic inside `_DecoderLayer.forward`

Replace the existing cross-attention call:
```python
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
```

With:
```python
key_padding_mask = None
if self.training and spatial_drop_prob > 0.0:
    B, N_spatial, _ = spatial_tokens.shape
    key_padding_mask = torch.rand(B, N_spatial, device=spatial_tokens.device) < spatial_drop_prob
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                     key_padding_mask=key_padding_mask)[0]
```

Constraints identical to designs 001/002: boolean `(B, N_spatial)` mask, fresh per call, `key_padding_mask=None` at inference.

#### 4. `Pose3dTransformerHead.__init__` — new parameters for annealing

Add `spatial_drop_prob_start: float = 0.30` and `spatial_drop_prob_end: float = 0.10` to the `__init__` signature (after `dropout`):

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    spatial_drop_prob_start: float = 0.30,
    spatial_drop_prob_end: float = 0.10,
    loss_joints: ConfigType = ...,
    ...
):
    ...
    self.spatial_drop_prob_start = spatial_drop_prob_start
    self.spatial_drop_prob_end = spatial_drop_prob_end
    # Runtime probability; hook updates this each epoch.
    self.spatial_drop_prob = spatial_drop_prob_start
```

Initialize `self.spatial_drop_prob = spatial_drop_prob_start` so training starts at p=0.30 from epoch 1 without waiting for the first hook call.

#### 5. `set_drop_prob` method on `Pose3dTransformerHead`

Add a public method (called by the hook):

```python
def set_drop_prob(self, p: float) -> None:
    """Update spatial dropout probability (called by SpatialDropAnnealHook)."""
    self.spatial_drop_prob = float(p)
```

#### 6. `Pose3dTransformerHead.forward` — pass drop prob to decoder

Change the decoder call from:
```python
decoded = self.decoder_layer(queries, spatial)
```
To:
```python
decoded = self.decoder_layer(queries, spatial, spatial_drop_prob=self.spatial_drop_prob)
```

#### 7. `SpatialDropAnnealHook` class — add after `Pose3dTransformerHead`

Define the hook at module level (not nested) after the `Pose3dTransformerHead` class definition:

```python
@HOOKS.register_module()
class SpatialDropAnnealHook(Hook):
    """Linearly anneals spatial dropout probability each epoch.

    At the start of epoch `e` (1-indexed), sets:
        p = start_prob + (end_prob - start_prob) * (e - 1) / (num_epochs - 1)

    For epoch 1: p = start_prob.
    For epoch num_epochs: p = end_prob.

    Args:
        num_epochs (int): Total training epochs (matches train_cfg.max_epochs).
        start_prob (float): Drop probability at epoch 1.
        end_prob (float): Drop probability at epoch num_epochs.
    """

    def __init__(self, num_epochs: int, start_prob: float, end_prob: float):
        self.num_epochs = num_epochs
        self.start_prob = start_prob
        self.end_prob = end_prob

    def before_train_epoch(self, runner) -> None:
        # runner.epoch is 0-indexed; convert to 1-indexed.
        epoch = runner.epoch + 1
        if self.num_epochs <= 1:
            p = self.end_prob
        else:
            t = (epoch - 1) / (self.num_epochs - 1)
            p = self.start_prob + (self.end_prob - self.start_prob) * t
        runner.model.head.set_drop_prob(p)
```

Important implementation notes:
- The hook uses `before_train_epoch`, which fires before each epoch's training loop begins.
- `runner.epoch` is 0-indexed in MMEngine. Epoch 0 → p=0.30 (start). Epoch 19 → p=0.10 (end).
- `runner.model.head` must be the `Pose3dTransformerHead` instance. In MMEngine with `RGBDPose3dEstimator`, the head is accessed via `runner.model.head`. If the model wraps the head differently (e.g., `runner.model.module.head` under DDP), access it safely:

  ```python
  model = runner.model
  if hasattr(model, 'module'):
      model = model.module
  model.head.set_drop_prob(p)
  ```

  Use this defensive access pattern in the actual implementation.

- The hook is defined in `pose3d_transformer_head.py` and registered via `@HOOKS.register_module()`. Since `pose3d_transformer_head` is already in `custom_imports`, it will be imported (and the hook registered) before MMEngine processes `custom_hooks`.

---

## Config Changes

### `config.py`

#### 1. Head kwargs — use `spatial_drop_prob_start` and `spatial_drop_prob_end`

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    spatial_drop_prob_start=0.30,
    spatial_drop_prob_end=0.10,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

#### 2. `custom_hooks` — add `SpatialDropAnnealHook`

Replace the baseline `custom_hooks` list with:

```python
custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(type='TrainMPJPEAveragingHook'),
    dict(type='MetricsCSVHook'),
    dict(type='SpatialDropAnnealHook', num_epochs=20, start_prob=0.30, end_prob=0.10),
]
```

The `SpatialDropAnnealHook` entry must use `type='SpatialDropAnnealHook'` exactly as registered by `@HOOKS.register_module()`.

---

## Annealing Schedule (explicit per-epoch values)

| Epoch (1-indexed) | `spatial_drop_prob` |
|---|---|
| 1 | 0.300 |
| 2 | 0.2895 |
| 3 | 0.2789 |
| 4 | 0.2684 |
| 5 | 0.2579 |
| 6 | 0.2474 |
| 7 | 0.2368 |
| 8 | 0.2263 |
| 9 | 0.2158 |
| 10 | 0.2053 |
| 11 | 0.1947 |
| 12 | 0.1842 |
| 13 | 0.1737 |
| 14 | 0.1632 |
| 15 | 0.1526 |
| 16 | 0.1421 |
| 17 | 0.1316 |
| 18 | 0.1211 |
| 19 | 0.1105 |
| 20 | 0.100 |

Formula: `p = 0.30 + (0.10 - 0.30) * (epoch - 1) / 19`

---

## Exact Config Values (unchanged from baseline)

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW, lr=1e-4, betas=(0.9,0.999), weight_decay=0.03 |
| backbone lr_mult | 0.1 |
| clip_grad max_norm | 1.0 |
| accumulative_counts | 8 (effective batch 32) |
| LR schedule | LinearLR (epoch 0–3, start_factor=0.333) + CosineAnnealingLR (epoch 3–20, eta_min=0) |
| seed | 2026 |
| batch_size | 4 |
| hidden_dim | 256 |
| num_heads | 8 |
| dropout | 0.1 |
| spatial_drop_prob_start | **0.30** (new) |
| spatial_drop_prob_end | **0.10** (new) |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0–21 only (`_BODY = list(range(0, 22))`).
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present in baseline; keep unchanged. The hook class is registered when this module is imported, so no additional import entry is needed.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals.
5. `key_padding_mask` must be on the same device as `spatial_tokens` — use `device=spatial_tokens.device`.
6. The mask is shared across all queries (one mask per batch element). Shape is `(B, N_spatial)`.
7. At inference (`self.training == False`), pass `key_padding_mask=None` unconditionally.
8. Do NOT register the mask as a buffer; create it fresh each call.
9. `SpatialDropAnnealHook` must use `@HOOKS.register_module()` decorator; the import `from mmengine.registry import HOOKS` and `from mmengine.hooks import Hook` must be added at the top of `pose3d_transformer_head.py`.
10. Use the defensive DDP unwrap pattern (`if hasattr(model, 'module'): model = model.module`) when accessing `runner.model.head` in the hook.
11. The head file uses absolute imports — do not add relative imports.
12. No changes to `pelvis_utils.py`, dataset, transforms, backbone, or metric files.
13. `self.spatial_drop_prob` is initialised to `spatial_drop_prob_start` in `__init__`, so the first epoch uses p=0.30 even before the hook's `before_train_epoch` fires (the hook also sets it to 0.30 for epoch 1, so behaviour is consistent).

---

## Expected Behaviour After Change

- Training epoch 1: p=0.30 dropout (high regularisation, broad spatial exploration).
- Training epoch 20: p=0.10 dropout (low regularisation, precise localisation).
- Validation/inference: all 960 spatial tokens used, no dropout.
- Expected composite_val improvement of −10 to −18 mm versus baseline 171.12.
- Best-of-both-worlds: early epochs benefit from aggressive regularisation (better query route diversity), late epochs can refine precise spatial attention patterns with reduced noise.
