**Design Description:** Two-layer spatial encoder (4 heads, zero-init) — deeper context enrichment variant.

**Starting Point:** `baseline/`

---

## Algorithm

Two-layer pre-decoder spatial self-attention encoder: the algorithm runs two sequential `_EncoderLayer` modules over the 960 spatial tokens before decoder cross-attention. Each encoder layer performs self-attention over all tokens + FFN with residual connections (pre-norm). After two passes, each token's representation integrates higher-order global context (first pass: pairwise token interactions; second pass: context-of-context). Zero-init on all output projections ensures baseline-equivalent behaviour at step 0.

## Overview

Same as design002 but with `num_encoder_layers=2` instead of 1. Two stacked `_EncoderLayer` modules run self-attention over the 960 spatial tokens before decoder cross-attention. After two self-attention passes, each spatial token's representation integrates higher-order dependencies (first pass: pairwise token context; second pass: context-of-context, e.g. wrist token "knows" about both elbow and shoulder). Memory estimate: 2 × 29 MB ≈ 58 MB encoder attention — within the 2080 Ti budget.

This design shares all implementation logic with design002. The only config difference is `num_encoder_layers=2`.

---

## Files to Change

1. `pose3d_transformer_head.py` — same changes as design001/design002 (add `_EncoderLayer` class; add new kwargs to `__init__`; insert encoder loop in `forward()`).
2. `config.py` — add new kwargs to the `head` dict with `num_encoder_layers=2`, `encoder_num_heads=4`.

No changes to `pelvis_utils.py`.

---

## `pose3d_transformer_head.py` Changes

**Identical to design001/design002.** Implement the same:

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

With `num_encoder_layers=2`, this loop runs twice — the output of the first `_EncoderLayer` is the input to the second, chaining two self-attention passes over the spatial tokens.

**Notes (same as design001/002):**
- `ffn[-2]` is the `nn.Linear(embed_dim*4, embed_dim)` layer (second-to-last element in the `nn.Sequential`).
- Zero-init applies independently to each layer's `self_attn.out_proj` and `ffn[-2]`. Each layer starts with zero residual delta.
- With `zero_init=True` and 2 layers: both layers add zero at initialization → encoder output is identical to baseline `spatial + pos_enc` at step 0.
- Pre-norm, no attention masking, all 960 spatial tokens are valid.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add after `loss_weight_uv=1.0`:

```python
use_spatial_encoder=True,
num_encoder_layers=2,
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
    num_encoder_layers=2,
    encoder_num_heads=4,
    encoder_dropout=0.1,
    encoder_zero_init=True,
),
```

All values are bool/int/float literals. No Python `import` statements. MMEngine config constraint fully satisfied.

Everything else in `config.py` is unchanged.

---

## Difference from design001 and design002

| Parameter | design001 | design002 | design003 |
|---|---|---|---|
| `num_encoder_layers` | 1 | 1 | **2** |
| `encoder_num_heads` | 8 | 4 | **4** |
| All other params | same | same | same |

The `nn.ModuleList` will contain 2 `_EncoderLayer` instances (each with 4 heads, `hidden_dim=256`). The `forward()` loop iterates twice.

---

## Memory Analysis

- Per encoder layer (4 heads, batch=4, float16): `4 × 4 × 960 × 960 × 2B ≈ 29 MB`
- 2 encoder layers: ~58 MB total encoder attention
- Decoder cross-attention (70 queries): `4 × 8 × 70 × 960 × 2B ≈ 8.5 MB`
- Backbone activations: ~6–7 GB (dominant)
- Total estimate: ~7 GB — within the 10.57 GB 2080 Ti VRAM budget.

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

- At initialization: both encoder layers have zero-init output projections → both layers add zero residual delta → `spatial` after encoder is identical to baseline `spatial + pos_enc` at step 0.
- During training: two sequential self-attention encoder layers enrich spatial tokens before decoder cross-attention. Second layer integrates context-of-context (higher-order global dependencies).
- Parameter count added: ~2.1M params (2 × 1.05M per encoder layer).
- Memory overhead: ~58 MB for encoder attention matrices — same as design001 (8 heads × 1 layer ≈ 4 heads × 2 layers).
- No change to output shapes: `joints (B,70,3)`, `pelvis_depth (B,1)`, `pelvis_uv (B,2)`.

---

## Target Metrics (Stage 1)

- `composite_val < 328` (higher bar than design001/002, approaching best prior 323.75)
- `mpjpe_body_val < 186 mm`
- `mpjpe_rel_val < 410 mm`
- `mpjpe_pelvis_val < 615 mm`
