**Design Description:** Single-layer spatial encoder (4 heads, zero-init) — memory-efficient variant of design001.

**Starting Point:** `baseline/`

---

## Algorithm

Same algorithm as design001: pre-decoder spatial self-attention encoder with one transformer encoder layer (self-attention over all 960 spatial tokens + FFN, with residual and zero-init). The only algorithmic difference from design001 is `encoder_num_heads=4` instead of 8, halving the encoder self-attention memory at equivalent spatial context propagation capacity.

## Overview

Identical to design001 except the encoder self-attention uses `encoder_num_heads=4` instead of 8. Rationale: the 960×960 encoder self-attention matrix dominates memory; halving from 8 to 4 heads cuts encoder attention memory from ~58 MB to ~29 MB in float16, improving the GPU memory budget on the 2080 Ti (10.57 GB VRAM). Spatial context propagation is largely insensitive to head count — 4 heads are sufficient for long-range token communication.

This design shares all implementation logic with design001 (same `_EncoderLayer` class, same `__init__` kwargs, same `forward()` insertion point). Only the config value differs.

---

## Files to Change

1. `pose3d_transformer_head.py` — same changes as design001 (add `_EncoderLayer` class; add new kwargs to `__init__`; insert encoder call in `forward()`).
2. `config.py` — add new kwargs to the `head` dict with `encoder_num_heads=4`.

No changes to `pelvis_utils.py`.

---

## `pose3d_transformer_head.py` Changes

**Identical to design001.** Implement the same:

1. Add `_EncoderLayer` class before `_DecoderLayer`:

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

2. Extend `Pose3dTransformerHead.__init__` signature with:

```python
use_spatial_encoder: bool = False,
num_encoder_layers: int = 1,
encoder_num_heads: int = 8,
encoder_dropout: float = 0.1,
encoder_zero_init: bool = True,
```

After `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`, add:

```python
self.use_spatial_encoder = use_spatial_encoder
if use_spatial_encoder:
    self.spatial_encoder = nn.ModuleList([
        _EncoderLayer(hidden_dim, encoder_num_heads, encoder_dropout, encoder_zero_init)
        for _ in range(num_encoder_layers)
    ])
```

3. In `forward()`, after `spatial = spatial + pos_enc` and before `queries = ...`, add:

```python
if self.use_spatial_encoder:
    for enc_layer in self.spatial_encoder:
        spatial = enc_layer(spatial)  # (B, H*W, hidden_dim)
```

**Notes (same as design001):**
- `ffn[-2]` is the `nn.Linear(embed_dim*4, embed_dim)` layer (index -2 in `nn.Sequential`; index -1 is `nn.Dropout`).
- Zero-init on `self_attn.out_proj` and `ffn[-2]` ensures encoder adds zero residual delta at init.
- Pre-norm architecture, no attention masking.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add after `loss_weight_uv=1.0`:

```python
use_spatial_encoder=True,
num_encoder_layers=1,
encoder_num_heads=4,
encoder_dropout=0.1,
encoder_zero_init=True,
```

The full updated `head` dict:

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
    encoder_num_heads=4,
    encoder_dropout=0.1,
    encoder_zero_init=True,
),
```

All values are bool/int/float literals. No Python `import` statements. MMEngine config constraint fully satisfied.

Everything else in `config.py` is unchanged.

---

## Difference from design001

| Parameter | design001 | design002 |
|---|---|---|
| `encoder_num_heads` | 8 | **4** |
| All other params | same | same |

The `_EncoderLayer` is instantiated with `num_heads=4`. The `hidden_dim=256` is divisible by 4, so `nn.MultiheadAttention(256, 4, ...)` is valid (head_dim = 64).

---

## Invariants to Preserve

- `persistent_workers=False` — do not change.
- Loss restricted to body joints (indices 0–21) — unchanged.
- `resume=True` and `CheckpointHook` with `max_keep_ckpts=1` — unchanged.
- `accumulative_counts=8`, `batch_size=4` — unchanged.
- `seed=2026` — unchanged.
- No Python `import` statements in `config.py` — satisfied.
- Absolute imports in `pose3d_transformer_head.py` — unchanged.

---

## Expected Behaviour After Change

- At initialization: encoder residual delta is exactly 0 (zero-init) → numerically identical to baseline.
- During training: encoder self-attends over 960 spatial tokens with 4 heads (head_dim=64).
- Memory overhead: encoder self-attn matrix `4 × 4 × 960 × 960 × 2B ≈ 29 MB` in float16 — more comfortable than design001's 58 MB.
- Parameter count: same as design001 (~1.05M params; head count doesn't change param count since `embed_dim` is fixed).
- No change to output shapes: `joints (B,70,3)`, `pelvis_depth (B,1)`, `pelvis_uv (B,2)`.

---

## Target Metrics (Stage 1)

- `composite_val < 335`
- `mpjpe_body_val < 188 mm`
- `mpjpe_rel_val < 420 mm`
- `mpjpe_pelvis_val < 620 mm`
