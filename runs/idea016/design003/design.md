**Design Description:** Hierarchical spatial-block FiLM conditioning (4×4=16 blocks, shared MLP) on spatial tokens before cross-attention, identity-initialised.

**Starting Point:** `baseline/`

---

## Overview

Insert a FiLM layer after `input_proj + pos_enc` that divides the 24×40 feature grid into 16 spatial blocks (4 row-blocks × 4 col-blocks). The algorithm: each block's average-pooled context vector `(B, hidden_dim)` is independently fed through a **shared** 2-layer MLP to produce block-specific `(γ_block, β_block)` that modulate only the tokens within that block. This allows body-region-specific scale conditioning: upper-body blocks (torso/head region) may have different depth gradients from lower-body blocks (legs/feet), and the model can learn distinct conditioning per region. The shared MLP keeps parameter count identical to Design 001 (~100K) while adding spatial specificity.

**Files changed:**
- `pose3d_transformer_head.py` — add `film_pool_type`, `film_hidden_dim`, `film_num_blocks` constructor args; add `self.film_net`; apply hierarchical FiLM in `forward()`
- `config.py` — add `film_pool_type='spatial_block'`, `film_hidden_dim=128`, `film_num_blocks=16` to the `head` dict

**Files NOT changed:** `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, `train.py`

---

## Spatial Block Layout

Feature map spatial dimensions: `H'=40` (height, from img_h=640 ÷ patch=16), `W'=24` (width, from img_w=384 ÷ patch=16). Total tokens: 40×24 = 960.

Block decomposition: 4 row-blocks × 4 col-blocks = 16 blocks.
- Row-block size: 40 ÷ 4 = 10 rows per block
- Col-block size: 24 ÷ 4 = 6 columns per block
- Tokens per block: 10 × 6 = 60 tokens
- Total: 16 × 60 = 960 tokens ✓ (exact partition, no remainder)

**Critical note on token layout:** In `forward()`, the spatial tokens are flattened as `feat.flatten(2).transpose(1, 2)` where `feat` is `(B, C, H', W')` = `(B, 1024, 40, 24)`. This means the flattened tokens are in row-major order: token index `i = row * 24 + col`. The reshape must respect this layout.

---

## `pose3d_transformer_head.py` Changes

### Constructor signature addition

Add three new keyword arguments to `Pose3dTransformerHead.__init__`, after `loss_weight_uv`:

```python
film_pool_type: str = 'none',
film_hidden_dim: int = 128,
film_num_blocks: int = 16,
```

### Constructor body additions

After `self.loss_weight_uv = loss_weight_uv` and before `self.loss_joints_module = MODELS.build(loss_joints)`:

```python
self.film_pool_type = film_pool_type
self.film_num_blocks = film_num_blocks  # only used when film_pool_type='spatial_block'

if film_pool_type == 'spatial_block':
    film_in_dim = hidden_dim  # per-block avg pool: (B, 16, hidden_dim)
else:
    film_in_dim = 0  # disabled; no film_net created

if film_in_dim > 0:
    self.film_net = nn.Sequential(
        nn.Linear(film_in_dim, film_hidden_dim),
        nn.GELU(),
        nn.Linear(film_hidden_dim, 2 * hidden_dim),
    )
    # Zero-init output layer → identity transform at init (gamma≈1, beta≈0)
    nn.init.zeros_(self.film_net[-1].weight)
    nn.init.zeros_(self.film_net[-1].bias)
```

For `film_pool_type='spatial_block'`, `film_hidden_dim=128`, `hidden_dim=256`:
- First linear: `256 → 128` (32768 params)
- Second linear: `128 → 512` (65536 params)
- Total added: ~98K parameters (same as Design 001, shared across 16 blocks)

Also store the block shape parameters for use in `forward()`:

```python
# These are set lazily in forward() based on actual H', W'
self._film_block_h: int | None = None
self._film_block_w: int | None = None
```

### `forward()` changes

Insert the hierarchical FiLM block **immediately after** `spatial = spatial + pos_enc` and **immediately before** `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)`.

The full FiLM block for `film_pool_type='spatial_block'`:

```python
if self.film_pool_type == 'spatial_block':
    # spatial: (B, H'*W', hidden_dim) = (B, 960, 256) for H'=40, W'=24
    # Block decomposition: 4 row-blocks × 4 col-blocks = 16 blocks
    # Each block: 10 rows × 6 cols = 60 tokens
    # Reshape layout (row-major flattening from feat.flatten(2)):
    #   spatial[i] corresponds to (row=i//W', col=i%W')
    #   Reshape to (B, H', W', D) = (B, 40, 24, 256)
    #   Then view as (B, 4, 10, 4, 6, D) — 4 row-blocks of 10 rows, 4 col-blocks of 6 cols
    #   Pool over dims (2, 4) → (B, 4, 4, D) = 16 block contexts

    D = spatial.size(-1)  # hidden_dim = 256
    # Reshape to spatial grid: (B, H', W', D)
    spatial_grid = spatial.view(B, H, W, D)
    # Partition into 16 blocks: (B, 4, 10, 4, 6, D)
    spatial_blocks = spatial_grid.view(B, 4, H // 4, 4, W // 4, D)
    # Average pool within each block over dims 2 (10 rows) and 4 (6 cols):
    ctx_blocks = spatial_blocks.mean(dim=2).mean(dim=3)  # (B, 4, 4, D)
    # Apply shared film_net independently to each block context:
    film_params = self.film_net(ctx_blocks)              # (B, 4, 4, 2*D)
    gamma_b, beta_b = film_params.chunk(2, dim=-1)       # each (B, 4, 4, D)
    gamma_b = gamma_b + 1.0                              # residual: identity at init

    # Scatter block FiLM params back to all tokens in each block:
    # Expand from (B, 4, 4, D) to (B, 4, H//4, 4, W//4, D)
    gamma_expanded = gamma_b.unsqueeze(2).expand(B, 4, H // 4, 4, W // 4, D)
    beta_expanded  = beta_b.unsqueeze(2).expand(B, 4, H // 4, 4, W // 4, D)
    # Reshape back to (B, H'*W', D) = (B, 960, D)
    gamma_spatial = gamma_expanded.reshape(B, H * W, D)
    beta_spatial  = beta_expanded.reshape(B, H * W, D)

    # Apply FiLM: element-wise per-token (different per block)
    spatial = spatial * gamma_spatial + beta_spatial     # (B, 960, 256)
```

**Important:** The `H` and `W` variables in this block refer to the feature map spatial dims extracted earlier in `forward()`: `B, C, H, W = feat.shape` → `H=40, W=24`. These are already available from the existing `forward()` code. The FiLM block uses `H // 4 = 10` (row-block size) and `W // 4 = 6` (col-block size). No new variables need to be introduced beyond what's already in scope.

**Step-by-step reshape verification:**
- `spatial.view(B, 40, 24, 256)` → `(B, 40, 24, 256)` ✓ (40×24=960)
- `.view(B, 4, 10, 4, 6, 256)` → `(B, 4, 10, 4, 6, 256)` ✓ (4×10=40, 4×6=24)
- `.mean(dim=2)` → `(B, 4, 4, 6, 256)` (avg over 10 rows within each row-block)
- `.mean(dim=3)` → `(B, 4, 4, 256)` (avg over 6 cols within each col-block)
- `self.film_net(ctx_blocks)` → `(B, 4, 4, 512)` (shared MLP applied to each of 16 block contexts via broadcasting over dims 1,2)
- `.chunk(2, dim=-1)` → `gamma_b, beta_b` each `(B, 4, 4, 256)` ✓
- `gamma_b.unsqueeze(2).expand(B, 4, 10, 4, 6, 256)` → `(B, 4, 10, 4, 6, 256)` ✓
- `.reshape(B, 960, 256)` ✓

**Note on `mean(dim=2).mean(dim=3)`:** After first `mean(dim=2)`, tensor is `(B, 4, 4, 6, 256)` (dim 2 was the 10-row dim; after mean it collapses). After second `mean(dim=3)`, tensor is `(B, 4, 4, 256)` (dim 3 was the 6-col dim; after mean it collapses). This is equivalent to `spatial_blocks.mean(dim=(2, 4))` but written as two sequential mean calls to avoid potential dim-index confusion after the first reduction.

**Alternative one-liner (equivalent, use if preferred):**
```python
ctx_blocks = spatial_blocks.mean(dim=(2, 4))  # (B, 4, 4, D)
```
This is mathematically identical; Builder may use either form.

### `loss()` and `predict()` — NO changes

Both call `self.forward(feats)`. Output dict shape unchanged.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add:

```python
film_pool_type='spatial_block',
film_hidden_dim=128,
film_num_blocks=16,
```

Full updated `head` dict:

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
    film_pool_type='spatial_block',
    film_hidden_dim=128,
    film_num_blocks=16,
),
```

All other config values identical to baseline.

---

## Exact Behaviour Specification

1. **At initialisation (step 0):** `film_net[-1].weight == 0`, `film_net[-1].bias == 0`. Therefore all block-wise `(gamma_b, beta_b) = (0,0)` → `gamma_b + 1.0 = 1.0`, `beta_b = 0.0`. Spatial tokens pass through unchanged. Identical to baseline at step 0.

2. **Spatial specificity:** Each of the 16 blocks gets its own `(γ_block, β_block)` derived from that block's average context. Upper-body crop regions (typically in top blocks for frontal captures) receive different conditioning from lower-body regions. The MLP is shared: all 16 blocks share the same linear weights, applied independently to 16 different input vectors `(B, 4, 4, 256)`. This is equivalent to a 1D convolution with kernel size 1 applied to 16 spatial locations.

3. **Token ordering:** The spatial tokens are created by `feat.flatten(2).transpose(1, 2)` where `feat=(B,C,H',W')`. PyTorch's `flatten(2)` flattens dims 2 and 3 (H' and W') in C-contiguous (row-major) order: token `i = row * W' + col`. The reshape `spatial.view(B, H, W, D)` followed by `.view(B, 4, H//4, 4, W//4, D)` preserves this ordering correctly, partitioning the grid into contiguous 10×6 blocks.

4. **`film_num_blocks=16` in config:** The constructor stores `self.film_num_blocks = film_num_blocks`. This kwarg is accepted for interface consistency but the implementation hardcodes the 4×4 block layout (derived from the known `H'=40, W'=24` and `film_num_blocks=16`). If `film_num_blocks` is not 16 or the feature map dims change, the hardcoded `4` in the view calls must be adjusted accordingly. For this design, `film_num_blocks=16` is always used and the hardcoded `4` is correct.

5. **AMP compatibility:** All ops (`view`, `mean`, `unsqueeze`, `expand`, `reshape`, `chunk`, element-wise multiply/add) are float16-safe.

6. **Memory footprint:** Extra tensors: `(B, 40, 24, 256)` (spatial_grid, a view), `(B, 4, 10, 4, 6, 256)` (spatial_blocks, a view), `(B, 4, 4, 256)` (ctx_blocks), `(B, 4, 4, 512)` (film_params), `(B, 4, 4, 256)` × 2 (gamma_b, beta_b), `(B, 4, 10, 4, 6, 256)` × 2 (expanded — these are expanded views, not new allocations), `(B, 960, 256)` × 2 (gamma_spatial, beta_spatial). The `expand` call does not allocate memory (lazy view). `reshape` after `expand` will allocate: `(B, 960, 256)` × 2 = same size as spatial. Total additional memory: roughly 2× the spatial tensor size. Acceptable for batch size 4.

---

## Constraints and Invariants the Builder Must Preserve

- `film_pool_type='spatial_block'`, `film_hidden_dim=128`, `film_num_blocks=16` are all str/int literals in config. No Python imports in config.
- The block sizes (row-block = 10, col-block = 6) are derived from `H' ÷ 4 = 40 ÷ 4 = 10` and `W' ÷ 4 = 24 ÷ 4 = 6`. These are exact integer divisions (no remainder). The Builder must not pad or truncate tokens.
- The FiLM block is inserted between `spatial + pos_enc` and `decoder_layer(queries, spatial)`. Order: `input_proj → pos_enc → FiLM → decoder_layer`.
- The `B`, `H`, `W` variables used in the FiLM block are from `B, C, H, W = feat.shape` earlier in `forward()` — they are already in scope. No new variable extraction is needed.
- When `film_pool_type='none'` (default), no `film_net` is created, `film_num_blocks` is stored but unused. Backward-compatible.
- Loss restricted to body joints 0-21 unchanged.
- `persistent_workers=False`, `resume=True`, `max_keep_ckpts=1`, seed 2026 all unchanged.
- The shared MLP (`film_net`) is applied to `(B, 4, 4, hidden_dim)` tensors. PyTorch's `nn.Linear` operates on the last dimension, broadcasting over all leading dims — this is the correct behaviour for a shared MLP across 16 block positions.
