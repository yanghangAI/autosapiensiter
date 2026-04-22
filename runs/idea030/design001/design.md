**Design Description:** Single-layer spatial encoder (8 heads, zero-init) inserted before decoder cross-attention.

**Starting Point:** `baseline/`

---

## Algorithm

Pre-decoder spatial self-attention encoder: run one transformer encoder layer (self-attention over all 960 spatial tokens + FFN) between the `input_proj + pos_enc` step and the decoder cross-attention step. Each spatial token attends to all other spatial tokens, producing globally-contextualized key/value representations that the decoder joint queries then cross-attend to. Zero-initialize encoder output projections so the algorithm starts from exact baseline behaviour.

## Overview

Add one `_EncoderLayer` that runs self-attention over the 960 spatial tokens (after `input_proj` + positional encoding) before they are used as keys/values in the decoder cross-attention. Zero-initialize the encoder output projections so training starts from exact baseline behaviour (encoder residual deltas are zero at step 0). This is Design A from idea030: the minimal diagnostic variant with 8 encoder heads.

---

## Files to Change

1. `pose3d_transformer_head.py` — add `_EncoderLayer` class; add new kwargs to `Pose3dTransformerHead.__init__`; insert encoder call in `forward()`.
2. `config.py` — add new kwargs to the `head` dict in `model`.

No changes to `pelvis_utils.py`.

---

## `pose3d_transformer_head.py` Changes

### 1. Add `_EncoderLayer` class

Insert the following class **before** `_DecoderLayer` (i.e., before line 77 in the baseline):

```python
class _EncoderLayer(nn.Module):
    """Single transformer encoder layer: self-attention over spatial tokens + FFN."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 zero_init: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if zero_init:
            nn.init.zeros_(self.self_attn.out_proj.weight)
            nn.init.zeros_(self.self_attn.out_proj.bias)
            nn.init.zeros_(self.ffn[-2].weight)
            nn.init.zeros_(self.ffn[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) spatial tokens

        Returns:
            (B, N, embed_dim) enriched spatial tokens
        """
        # Self-attention (pre-norm)
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2)[0]
        x = x + self.dropout1(x2)

        # FFN (pre-norm)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x
```

**Notes on `_EncoderLayer`:**
- `ffn[-2]` refers to the second `nn.Linear` in the `nn.Sequential` (index -2 is the `nn.Linear(embed_dim*4, embed_dim)` layer; index -1 is `nn.Dropout`). This is the output projection of the FFN.
- `self.self_attn.out_proj` is the output linear of `nn.MultiheadAttention`.
- Zero-init on both projections guarantees the encoder adds zero residual delta at initialization, preserving baseline behaviour.
- Pre-norm architecture (LayerNorm before attention/FFN), matching the decoder's style.
- No attention mask or key padding mask — all 960 spatial tokens are valid.

### 2. Add new kwargs to `Pose3dTransformerHead.__init__`

Extend the signature with these parameters (all defaulting to baseline-equivalent values):

```python
use_spatial_encoder: bool = False,
num_encoder_layers: int = 1,
encoder_num_heads: int = 8,
encoder_dropout: float = 0.1,
encoder_zero_init: bool = True,
```

Insert them after the existing `dropout: float = 0.1` parameter.

Inside `__init__`, after the line `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
# Spatial encoder (optional)
self.use_spatial_encoder = use_spatial_encoder
if use_spatial_encoder:
    self.spatial_encoder = nn.ModuleList([
        _EncoderLayer(hidden_dim, encoder_num_heads, encoder_dropout, encoder_zero_init)
        for _ in range(num_encoder_layers)
    ])
```

When `use_spatial_encoder=False` (baseline default), no `spatial_encoder` attribute is created and the `nn.ModuleList` is not registered, keeping the parameter count identical to baseline.

### 3. Modify `forward()` to call the encoder

In `forward()`, after the line:
```python
spatial = spatial + pos_enc
```
and before the line:
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(...)
```

Insert:

```python
# Optional spatial encoder: self-attention over all spatial tokens
if self.use_spatial_encoder:
    for enc_layer in self.spatial_encoder:
        spatial = enc_layer(spatial)  # (B, H*W, hidden_dim)
```

The encoder receives positional-encoded spatial tokens and returns enriched tokens of the same shape. The decoder cross-attention then uses these enriched tokens as keys and values — semantically equivalent to the DETR encoder+decoder pattern.

### 4. No changes to `loss()` or `predict()`

`loss()` and `predict()` both call `self.forward(feats)` — no modifications needed. The encoder change is transparent to the rest of the head.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add the following kwargs after `loss_weight_uv=1.0`:

```python
use_spatial_encoder=True,
num_encoder_layers=1,
encoder_num_heads=8,
encoder_dropout=0.1,
encoder_zero_init=True,
```

All values are bool/int/float literals. No Python `import` statements. MMEngine config constraint fully satisfied.

The full updated `head` dict in `config.py`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
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
    use_spatial_encoder=True,
    num_encoder_layers=1,
    encoder_num_heads=8,
    encoder_dropout=0.1,
    encoder_zero_init=True,
),
```

Everything else in `config.py` is unchanged (optimizer, LR schedule, data pipeline, hooks, etc.).

---

## Invariants to Preserve

- `persistent_workers=False` — do not change.
- Loss restricted to body joints (indices 0–21) — unchanged, in `loss()`.
- `resume=True` and `CheckpointHook` with `max_keep_ckpts=1` — unchanged.
- `accumulative_counts=8`, `batch_size=4` — unchanged.
- `seed=2026` — unchanged.
- No Python `import` statements in `config.py` — satisfied (all new values are literals).
- Absolute imports in `pose3d_transformer_head.py` — unchanged.
- `_BODY = list(range(0, 22))` loss restriction — unchanged.

---

## Expected Behaviour After Change

- At initialization (step 0): encoder output projections are zero-initialized → encoder residual delta is exactly 0 → `spatial` tensor after encoder is identical to baseline `spatial + pos_enc` → first forward pass is numerically identical to baseline.
- During training: encoder learns to enrich spatial tokens with global context via self-attention. Joint queries in the decoder cross-attend to richer, globally-contextualized keys/values.
- Parameter count added: ~1.05M params (1 encoder layer: 262,144 self-attn + 786,432 FFN + norms).
- Memory overhead: encoder self-attn matrix `4 × 8 × 960 × 960 × 2B ≈ 58 MB` in float16 — within 2080 Ti budget.
- No change to output shapes: `joints (B,70,3)`, `pelvis_depth (B,1)`, `pelvis_uv (B,2)`.

---

## Target Metrics (Stage 1)

- `composite_val < 335`
- `mpjpe_body_val < 188 mm`
- `mpjpe_rel_val < 420 mm`
- `mpjpe_pelvis_val < 620 mm`
