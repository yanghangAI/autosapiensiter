# Design 001 — K_s=8 Deformable Spatial Sampling, 70 Queries, Single Decoder Layer

**Design Description:** Replace standard dense cross-attention (Q×K: 70×960) with per-query deformable sampling (K_s=8 bilinear-interpolated spatial features) via a shared offset MLP; single decoder layer, all 70 joint queries.

**Starting Point:** `baseline/`

---

## Overview

The baseline cross-attention algorithm attends over all 960 spatial tokens (24×40 feature grid) identically for every joint query. This design replaces that dense cross-attention algorithm with per-query deformable sparse sampling: each of the 70 joint queries predicts K_s=8 2D reference-point offsets, bilinearly samples K_s features from the 2D spatial grid, and computes a lightweight weighted sum over those K_s=8 features instead of full softmax attention over 960 tokens.

All other components are unchanged: self-attention, FFN, positional encoding, pelvis head, loss, backbone, data pipeline.

This is the minimal diagnostic design — does learned sparse sampling alone improve over baseline?

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### New class: `_DeformableDecoderLayer`

Add this class after `_DecoderLayer` and before `Pose3dTransformerHead`. Full implementation spec:

```python
class _DeformableDecoderLayer(nn.Module):
    """Transformer decoder layer with per-query deformable sparse cross-attention.

    Self-attention is unchanged (all queries attend to each other via MHA).
    Cross-attention is replaced: each query independently predicts K_s=num_points
    2D offset vectors from a learnable reference point, bilinearly samples
    num_points spatial features from the 2D feature grid, then performs a
    learned weighted sum over the num_points sampled features.

    Args:
        embed_dim: Embedding/hidden dimension (256).
        num_heads: Number of attention heads for self-attention (8).
        dropout: Dropout probability (0.1).
        num_points: Number of sampling points per query K_s (8).
        deform_hidden_dim: Bottleneck width in offset MLP (64).
        num_queries: Number of joint queries (must match ref_points shape) (70).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_points: int = 8,
        deform_hidden_dim: int = 64,
        num_queries: int = 70,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_points = num_points
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Self-attention (unchanged from baseline _DecoderLayer)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Offset network: shared across all queries
        # Input: (B, num_queries, embed_dim)
        # Output: (B, num_queries, num_points * 2) — 2D offsets per sampling point
        self.offset_net = nn.Sequential(
            nn.Linear(embed_dim, deform_hidden_dim),
            nn.GELU(),
            nn.Linear(deform_hidden_dim, num_points * 2),
        )

        # Learnable reference points: one (u, v) centre per query, in [0,1]^2
        # Initialised to (0.5, 0.5) = grid centre for all queries
        self.ref_points = nn.Parameter(torch.full((num_queries, 2), 0.5))

        # Lightweight cross-attention over K_s sampled features:
        # per-query scalar attention weights (not full MHA — single-head linear attention)
        self.attn_weight_net = nn.Linear(embed_dim, num_points)

        # Value and output projections for the sparse cross-attention
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN (identical to baseline)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms and dropouts
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sample_spatial_features(
        self,
        queries: torch.Tensor,       # (B, num_queries, embed_dim)
        spatial_grid: torch.Tensor,  # (B, embed_dim, H', W')
    ) -> torch.Tensor:
        """Sample K_s=num_points features per query via bilinear interpolation.

        Returns:
            Tensor of shape (B, num_queries, num_points, embed_dim).
        """
        B, Nq, D = queries.shape
        _, _, H, W = spatial_grid.shape

        # Predict per-query offset vectors from query features
        offsets = self.offset_net(queries)                       # (B, Nq, num_points*2)
        offsets = offsets.view(B, Nq, self.num_points, 2)        # (B, Nq, K_s, 2)

        # Reference points broadcast: (1, Nq, 1, 2)
        ref = self.ref_points.unsqueeze(0).unsqueeze(2)          # (1, Nq, 1, 2)

        # Sampling locations in [0,1]^2; offsets scaled by 0.1 (±10% grid)
        sample_locs = ref + offsets * 0.1                        # (B, Nq, K_s, 2)
        sample_locs = sample_locs.clamp(0.0, 1.0)

        # Convert to [-1,1] for F.grid_sample (expects x=W-axis, y=H-axis)
        # sample_locs[:,:,:,0] = u (width direction → x for grid_sample)
        # sample_locs[:,:,:,1] = v (height direction → y for grid_sample)
        grid = sample_locs * 2.0 - 1.0                          # (B, Nq, K_s, 2)

        # Reshape for grid_sample: needs (B, H_out, W_out, 2)
        # Treat Nq*K_s points as a 1-row grid of width Nq*K_s
        grid = grid.view(B, Nq * self.num_points, 1, 2)          # (B, Nq*K_s, 1, 2)

        # Cast grid to match spatial_grid dtype for AMP compatibility
        grid = grid.to(spatial_grid.dtype)

        sampled = torch.nn.functional.grid_sample(
            spatial_grid, grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )  # (B, embed_dim, Nq*K_s, 1)

        sampled = sampled.squeeze(-1).transpose(1, 2)            # (B, Nq*K_s, embed_dim)
        sampled = sampled.view(B, Nq, self.num_points, D)        # (B, Nq, K_s, embed_dim)
        return sampled

    def forward(
        self,
        queries: torch.Tensor,       # (B, num_queries, embed_dim)
        spatial_grid: torch.Tensor,  # (B, embed_dim, H', W') — NOT flattened
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, embed_dim)
            spatial_grid: (B, embed_dim, H', W') — 2D feature map after input_proj + pos_enc

        Returns:
            (B, num_queries, embed_dim)
        """
        # 1. Self-attention (pre-norm, unchanged from baseline)
        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        # 2. Deformable cross-attention
        q = self.norm2(queries)
        sampled = self._sample_spatial_features(q, spatial_grid)
        # sampled: (B, Nq, K_s, embed_dim)

        # Per-query attention weights over K_s points (scalar, linear attention)
        attn_w = self.attn_weight_net(q)                         # (B, Nq, K_s)
        attn_w = attn_w.softmax(dim=-1).unsqueeze(-1)            # (B, Nq, K_s, 1)

        # Project sampled values and aggregate
        values = self.value_proj(sampled)                        # (B, Nq, K_s, embed_dim)
        attended = (attn_w * values).sum(dim=2)                  # (B, Nq, embed_dim)
        attended = self.out_proj(attended)                       # (B, Nq, embed_dim)
        queries = queries + self.dropout2(attended)

        # 3. FFN (pre-norm)
        queries = queries + self.ffn(self.norm3(queries))
        return queries
```

#### Changes to `Pose3dTransformerHead.__init__`

New constructor kwargs (insert after `dropout: float = 0.1`):
- `deform_num_points: int = 0` — 0 means disabled (use baseline `_DecoderLayer`)
- `deform_hidden_dim: int = 64` — offset MLP bottleneck width
- `num_body_queries: int = 70` — number of joint queries (kept at 70 for Design 001)
- `num_decoder_layers: int = 1`
- `hand_aux_loss_weight: float = 0.0`
- `aux_body_loss_weight: float = 0.0`

Store: `self.use_deform = deform_num_points > 0`, `self.num_body_queries = num_body_queries`, `self.hand_aux_loss_weight = hand_aux_loss_weight`, `self.aux_body_loss_weight = aux_body_loss_weight`, `self.num_decoder_layers = num_decoder_layers`.

Replace the single `self.decoder_layer = _DecoderLayer(...)` with:

```python
self.use_deform = deform_num_points > 0
if self.use_deform:
    self.decoder_layers = nn.ModuleList([
        _DeformableDecoderLayer(
            hidden_dim, num_heads, dropout,
            num_points=deform_num_points,
            deform_hidden_dim=deform_hidden_dim,
            num_queries=num_body_queries,
        )
        for _ in range(num_decoder_layers)
    ])
else:
    self.decoder_layers = nn.ModuleList([
        _DecoderLayer(hidden_dim, num_heads, dropout)
        for _ in range(num_decoder_layers)
    ])
# Alias for any code referencing self.decoder_layer (backward compat)
self.decoder_layer = self.decoder_layers[0]
```

Change `self.joint_queries = nn.Embedding(num_joints, hidden_dim)` to:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

For Design 001, `num_body_queries=70` so `joint_queries` shape is identical to baseline (70, 256).

No `hand_proj` needed for Design 001 (`hand_aux_loss_weight=0.0`, `num_body_queries=70`).

#### Changes to `_init_head_weights`

After existing init code, add:

```python
if self.use_deform:
    for layer_mod in self.decoder_layers:
        # Near-zero init for offset net output: all offsets ≈ 0 at start
        nn.init.zeros_(layer_mod.offset_net[-1].weight)
        nn.init.zeros_(layer_mod.offset_net[-1].bias)
        # Near-zero init for attention weight net: uniform weights at start
        nn.init.zeros_(layer_mod.attn_weight_net.weight)
        nn.init.zeros_(layer_mod.attn_weight_net.bias)
        # Standard small-std init for value/output projections
        nn.init.trunc_normal_(layer_mod.value_proj.weight, std=0.02)
        if layer_mod.value_proj.bias is not None:
            nn.init.zeros_(layer_mod.value_proj.bias)
        nn.init.trunc_normal_(layer_mod.out_proj.weight, std=0.02)
        if layer_mod.out_proj.bias is not None:
            nn.init.zeros_(layer_mod.out_proj.bias)
```

Rationale: with zero offset_net output and zero attn_weight_net output, at initialisation all K_s=8 sampled features are at the same grid-centre location, attn_w is uniform (softmax over zeros = 1/K_s), so attended = value_proj(centre_feature) — a stable linear transform of the single centre token. Gradients from the body joint loss push ref_points and offset_net toward the relevant body regions over the first few epochs.

#### Changes to `forward()`

Replace the existing forward method body:

```python
def forward(self, feats):
    feat = feats[-1]  # (B, C, H, W)
    B, C, H, W = feat.shape

    # Project backbone features to hidden_dim
    spatial_flat = feat.flatten(2).transpose(1, 2)   # (B, H*W, C)
    spatial_proj = self.input_proj(spatial_flat)      # (B, H*W, hidden_dim)
    pos_enc = self._get_pos_enc(H, W, feat.device)
    spatial_proj = spatial_proj + pos_enc             # (B, H*W, hidden_dim)

    if self.use_deform:
        # Reshape back to 2D grid for grid_sample
        spatial_grid = spatial_proj.transpose(1, 2).view(
            B, self.hidden_dim, H, W)                 # (B, hidden_dim, H', W')
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        # (B, num_body_queries, hidden_dim)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_grid)
        decoded = queries                              # (B, num_body_queries, hidden_dim)
    else:
        # Standard baseline path (non-deformable)
        queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.decoder_layers:
            queries = layer(queries, spatial_proj)
        decoded = queries                              # (B, num_joints, hidden_dim)

    # For Design 001: num_body_queries=70, so decoded is (B, 70, hidden_dim)
    joints = self.joints_out(decoded)                 # (B, 70, 3)

    pelvis_token = decoded[:, 0, :]                   # (B, hidden_dim)
    pelvis_depth = self.depth_out(pelvis_token)       # (B, 1)
    pelvis_uv = self.uv_out(pelvis_token)             # (B, 2)

    return {
        'joints': joints,
        'pelvis_depth': pelvis_depth,
        'pelvis_uv': pelvis_uv,
    }
```

Note: For Design 001, `joints.shape == (B, 70, 3)` — identical to baseline output shape.

#### Changes to `loss()`

For Design 001, no structural changes to `loss()`. The existing body joint loss `_BODY = list(range(0, 22))`, depth loss, and UV loss remain. `hand_aux_loss_weight=0.0` so no auxiliary hand loss is computed.

The Builder must still add a guard: `if self.hand_aux_loss_weight > 0: ...` for forward compatibility, but for Design 001 this block is never entered.

---

### 2. `config.py`

Replace the `head=dict(...)` block with:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=1024,
    hidden_dim=256,
    num_joints=70,
    num_heads=8,
    dropout=0.1,
    deform_num_points=8,
    deform_hidden_dim=64,
    num_body_queries=70,
    num_decoder_layers=1,
    hand_aux_loss_weight=0.0,
    aux_body_loss_weight=0.0,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All values are int/float/str literals. No Python import statements. The `in_channels=1024` replaces the variable `embed_dim` (which equals 1024 — hardcoded literal here per MMEngine config constraint that literals must be used in nested dicts that cannot resolve outer variables).

All other config (optimizer, LR schedule, warmup, data pipeline, hooks, backbone, dataloaders, evaluators) are **identical to baseline**.

---

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Count Delta vs Baseline

New parameters in `_DeformableDecoderLayer` (replacing `_DecoderLayer`):
- `offset_net`: Linear(256, 64) + Linear(64, 16) = 256×64+64 + 64×16+16 = 16,448 + 1,040 = **17,488**
- `ref_points`: 70×2 = **140**
- `attn_weight_net`: Linear(256, 8) = 256×8+8 = **2,056**
- `value_proj`: Linear(256, 256) = 256×256+256 = **65,792**
- `out_proj`: Linear(256, 256) = 256×256+256 = **65,792**

Removed (from `_DecoderLayer`):
- `cross_attn` (`nn.MultiheadAttention(256, 8)`): in_proj (3×256×256=196,608) + out_proj (256×256=65,536) + biases = **~263,168**

Net change: +151,268 − 263,168 ≈ **−112,000 parameters** (net reduction vs. baseline cross-attention MHA).

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Output `joints` shape must be `(B, 70, 3)` for all downstream metric code.
3. `pelvis_token = decoded[:, 0, :]` — query 0 remains the pelvis depth/UV token.
4. Loss restricted to body joints `_BODY = list(range(0, 22))` — unchanged.
5. `self.num_joints = 70` must remain to keep `predict()` output shape correct.
6. `in_channels=1024` is hardcoded as literal in config (not `embed_dim` variable).
7. AMP compatibility: `grid = grid.to(spatial_grid.dtype)` cast is **required** before `F.grid_sample` to prevent dtype mismatch under float16 AMP.
8. `padding_mode='border'` (not 'zeros') to avoid gradient discontinuities at boundaries.
9. `align_corners=True` in `F.grid_sample` — consistent with the coordinate convention in the design.
10. `offset_net[-1]` refers to the last `nn.Linear` in the `nn.Sequential` (index 2). Builder must verify this indexing matches the actual offset_net definition.
11. `decoder_layers` (plural, `nn.ModuleList`) must be used in `forward()` — never index `decoder_layer` (singular) in the deformable path.
12. MMEngine config: all values are int or float literals. No `import` statements inside the config dict.
13. `num_body_queries` passed to `_DeformableDecoderLayer.__init__` must equal `self.joint_queries.weight.shape[0]` — an `assert` is recommended in the layer `__init__`.
14. Backbone, data preprocessor, metric, transforms are invariant.
15. Seed `2026`, batch size `4`, accumulation `8` — do not change.

---

## Expected Behavior After Change

- Cross-attention key-value set shrinks from 960 to 8 per query; attention matrix per batch is `(B, 70, 8)` vs. `(B, 70, 960)` baseline.
- At initialisation: all 70 queries sample the same centre feature (stable cold-start).
- Over 20 epochs: `ref_points` migrate toward anatomically relevant grid positions per joint; offset network learns to expand the sampling footprint around each reference.
- VRAM for cross-attention activations: 70×8/(70×960) = 0.83% of baseline cross-attention. Overall head VRAM reduces ~30–40%, likely eliminating any VRAM pressure.
- `composite_val` target: < 340 at stage-1. Primary diagnostic: if deformable sampling provides any benefit at all over baseline (baseline stage-1 composite ~360–380 range from prior runs).
- `mpjpe_body_val` target: < 190 mm at stage-1.
- Output dict shape `{'joints': (B,70,3), 'pelvis_depth': (B,1), 'pelvis_uv': (B,2)}` — unchanged from baseline. All downstream hooks and metrics receive identical-shape tensors.
