# Design 001 — Uniform Spatial Token Dropout (p=0.15)

**Design Description:** Add 15% uniform spatial token dropout to cross-attention during training by masking random spatial tokens via `key_padding_mask`, leaving inference unchanged.

**Starting Point:** `baseline/`

---

## Overview

During training, randomly zero out 15% of the H'×W' = 960 spatial tokens before they participate as keys/values in cross-attention. This is implemented as a boolean `key_padding_mask` passed to `nn.MultiheadAttention`. At inference, no masking is applied (standard DropKey behaviour). All other architecture, loss, and training settings remain identical to the baseline.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` — **no changes**.

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `_DecoderLayer.forward` signature change

Add a `spatial_drop_prob: float = 0.0` keyword argument:

```
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            spatial_drop_prob: float = 0.0) -> torch.Tensor:
```

#### 2. Cross-attention key_padding_mask logic inside `_DecoderLayer.forward`

Replace the existing cross-attention call:
```python
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
```

With:
```python
key_padding_mask = None
if self.training and spatial_drop_prob > 0.0:
    B, N_spatial, _ = spatial_tokens.shape
    # Boolean mask: True = ignore this token.
    # Shape: (B, N_spatial). Generated fresh every forward call (not a buffer).
    key_padding_mask = torch.rand(B, N_spatial, device=spatial_tokens.device) < spatial_drop_prob
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                     key_padding_mask=key_padding_mask)[0]
```

Key constraints:
- `key_padding_mask` shape must be `(B, N_spatial)` — matches `nn.MultiheadAttention(batch_first=True)`.
- `True` entries are ignored (masked out). PyTorch convention for `key_padding_mask` in `nn.MultiheadAttention`.
- The mask is regenerated fresh on every forward call (not a registered buffer, not stored as an attribute).
- The condition `self.training` ensures inference uses all tokens (`key_padding_mask=None`).

#### 3. `Pose3dTransformerHead.__init__` — new parameter

Add `spatial_drop_prob: float = 0.0` to the `__init__` signature (after `dropout`):

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    spatial_drop_prob: float = 0.0,
    loss_joints: ConfigType = ...,
    ...
):
    ...
    self.spatial_drop_prob = spatial_drop_prob
```

Store as `self.spatial_drop_prob = spatial_drop_prob`.

#### 4. `Pose3dTransformerHead.forward` — pass drop prob to decoder

Change the decoder call from:
```python
decoded = self.decoder_layer(queries, spatial)
```
To:
```python
decoded = self.decoder_layer(queries, spatial, spatial_drop_prob=self.spatial_drop_prob)
```

No other changes to `forward`, `loss`, or `predict`.

---

## Config Changes

### `config.py`

In the `head` dict inside `model`, add `spatial_drop_prob=0.15`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    spatial_drop_prob=0.15,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed) are identical to the baseline.

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
| spatial_drop_prob | **0.15** (new) |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0–21 only (`_BODY = list(range(0, 22))`).
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present in baseline; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals.
5. `key_padding_mask` must be on the same device as `spatial_tokens` — use `device=spatial_tokens.device`.
6. The mask is shared across all queries (one mask per batch element, not per query per batch element). Shape is `(B, N_spatial)`, not `(B * num_heads, num_queries, N_spatial)`.
7. At inference (`self.training == False`), pass `key_padding_mask=None` unconditionally.
8. Do NOT register the mask as a buffer; create it fresh each call.
9. The head file uses absolute imports — do not add relative imports.
10. No changes to `pelvis_utils.py`, dataset, transforms, backbone, or metric files.

---

## Expected Behaviour After Change

- During training: each forward pass randomly masks ~15% of the 960 spatial tokens (expected ~144 masked, ~816 visible). All joint queries see the same dropped set for a given batch element.
- During validation/inference: all 960 spatial tokens are used; behaviour is identical to baseline.
- Training loss may be slightly higher early on due to reduced spatial coverage, but should converge to better generalisation.
- Expected composite_val improvement of −5 to −10 mm versus baseline 171.12.
