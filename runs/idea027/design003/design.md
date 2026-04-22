**Design Description:** Add two-layer stacked depthwise-separable spatial context conv with GroupNorm(32), zero-init on final pointwise only, giving 5×5 effective receptive field over the spatial token grid.

**Starting Point:** `baseline/`

---

## Overview

Same architecture as design002 but stacks **two** successive depthwise-separable blocks (each: depthwise 3×3 → GroupNorm(32) → GELU → pointwise 1×1). Stacking two 3×3 depthwise convolutions produces a 5×5 effective receptive field (each grid cell sees its neighbors up to 2 cells away), covering approximately 80×80 pixels at the 40×24 grid scale (each cell is 640/40=16 px). This receptive field is large enough to encompass an entire forearm or lower leg within a single enriched token's context.

**Zero-init strategy for two layers:** only the **second (final)** pointwise conv is zero-initialized. The first pointwise conv uses trunc_normal(std=0.02) init. This means: the first layer starts randomly and can immediately learn spatial context from step 1; but the second layer's residual output is zero at init, guaranteeing the overall stack output equals zero (baseline-equivalent start). As the second layer's pointwise gradient grows, the two layers progressively learn a wider-receptive-field context representation.

---

## Algorithm

The core algorithmic change is inserting a two-layer stacked depthwise-separable 2D convolution pass over the spatial token grid between the input projection + positional encoding step and the transformer decoder cross-attention. Layer 1: 3×3 depthwise → GroupNorm(32) → GELU → 1×1 pointwise (trunc_normal init). Layer 2: 3×3 depthwise → GroupNorm(32) → GELU → 1×1 pointwise (zero init). Effective receptive field: 5×5 grid cells (~80×80 px). The residual connection (`spatial + delta`) adds the full two-layer stack output back to the original spatial tokens. Zero-init on layer 2's pointwise weight and bias guarantees delta=0 at training start, making epoch-0 behavior identical to baseline.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` — **no changes**.

---

## `pose3d_transformer_head.py` Changes

### 1. Add `_SpatialContextNet` class

**Exactly the same class definition as design001/design002** — the class already handles `num_layers=2` via its loop, and `zero_init_last=True` ensures only the final (second) pointwise layer is zero-initialized. Insert before `_DecoderLayer`. Full class:

```python
class _SpatialContextNet(nn.Module):
    """Lightweight depthwise-separable 2D conv for spatial token context enrichment.

    Reshapes the flattened spatial token sequence to a 2D grid, applies
    depthwise-separable convolution with optional normalization, and returns
    the enriched tokens via a residual connection.

    The last pointwise conv layer is zero-initialized so that at training start
    the module outputs exactly zero (residual = 0) — identical to the baseline.

    Args:
        hidden_dim: Channel dimension (equals head hidden_dim).
        kernel_size: Depthwise conv kernel size (use 3 for 3x3).
        num_layers: Number of depthwise-separable blocks to stack.
        norm: Normalization type: 'none' or 'groupnorm'.
        num_groups: GroupNorm groups (only used if norm='groupnorm').
        act: Activation function: 'gelu' or 'relu'.
        zero_init_last: If True, zero-initialize only the final pointwise layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        norm: str = 'none',
        num_groups: int = 32,
        act: str = 'gelu',
        zero_init_last: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            # Depthwise conv: per-channel spatial filtering
            dw = nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            )
            nn.init.kaiming_normal_(dw.weight, mode='fan_out', nonlinearity='relu')
            layers.append(dw)

            # Optional normalization
            if norm == 'groupnorm':
                layers.append(nn.GroupNorm(num_groups, hidden_dim))
            else:
                layers.append(nn.Identity())

            # Activation
            if act == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU(inplace=True))

            # Pointwise conv: channel mixing
            pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
            if zero_init_last and is_last:
                nn.init.zeros_(pw.weight)
                nn.init.zeros_(pw.bias)
            else:
                nn.init.trunc_normal_(pw.weight, std=0.02)
                nn.init.zeros_(pw.bias)
            layers.append(pw)

        self.net = nn.Sequential(*layers)

    def forward(self, spatial: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            spatial: (B, H*W, hidden_dim) — flattened spatial tokens.
            h: Height of the spatial grid.
            w: Width of the spatial grid.

        Returns:
            (B, H*W, hidden_dim) — enriched spatial tokens (residual added).
        """
        B, _, D = spatial.shape
        # Reshape to 2D grid: (B, D, H, W)
        x = spatial.transpose(1, 2).reshape(B, D, h, w)
        # Apply spatial context network
        delta = self.net(x)                               # (B, D, H, W)
        delta = delta.reshape(B, D, -1).transpose(1, 2)  # (B, H*W, D)
        return spatial + delta
```

### 2. Modify `Pose3dTransformerHead.__init__`

**Add new kwargs** after existing kwargs, before `init_cfg`:

```python
use_spatial_ctx: bool = False,
spatial_ctx_kernel: int = 3,
spatial_ctx_layers: int = 1,
spatial_ctx_norm: str = 'none',
spatial_ctx_groups: int = 32,
spatial_ctx_act: str = 'gelu',
```

**Store instance flag:**

After `self.loss_weight_uv = loss_weight_uv`, add:

```python
self.use_spatial_ctx = use_spatial_ctx
```

**Conditionally instantiate module** after `self.decoder_layer = _DecoderLayer(...)`:

```python
if use_spatial_ctx:
    self.spatial_ctx_net = _SpatialContextNet(
        hidden_dim=hidden_dim,
        kernel_size=spatial_ctx_kernel,
        num_layers=spatial_ctx_layers,
        norm=spatial_ctx_norm,
        num_groups=spatial_ctx_groups,
        act=spatial_ctx_act,
        zero_init_last=True,
    )
```

### 3. Modify `Pose3dTransformerHead.forward`

After `spatial = spatial + pos_enc` and before `queries = self.joint_queries...`, insert:

```python
# Optional spatial context enrichment (depthwise-separable conv on token grid)
if self.use_spatial_ctx:
    spatial = self.spatial_ctx_net(spatial, H, W)
```

### 4. No changes to `loss()` or `predict()`

---

## `config.py` Changes

In `model.head` dict, add the following keys after `loss_weight_uv=1.0,`:

```python
use_spatial_ctx=True,
spatial_ctx_kernel=3,
spatial_ctx_layers=2,
spatial_ctx_norm='groupnorm',
spatial_ctx_groups=32,
spatial_ctx_act='gelu',
```

All values are bool/int/str literals. No Python import statements. MMEngine config constraint satisfied.

---

## Exact Parameter Values

| Parameter | Value |
|---|---|
| `use_spatial_ctx` | `True` |
| `spatial_ctx_kernel` | `3` |
| `spatial_ctx_layers` | `2` |
| `spatial_ctx_norm` | `'groupnorm'` |
| `spatial_ctx_groups` | `32` |
| `spatial_ctx_act` | `'gelu'` |
| depthwise kernel (each layer) | 3×3, `padding=1`, `groups=hidden_dim`, `bias=False` |
| GroupNorm (each layer) | `nn.GroupNorm(32, 256)` |
| pointwise kernel (each layer) | 1×1, `bias=True` |
| layer 1 pointwise init | `trunc_normal_(std=0.02)` weight, `zeros_` bias |
| layer 2 pointwise init | `zeros_` weight and bias |
| activation (each layer) | GELU |
| residual | `spatial + delta` (applied once to the full two-layer stack output) |
| `zero_init_last` | `True` |
| effective receptive field | 5×5 grid cells = ~80×80 pixels |

---

## Two-Layer Stack Structure

The `nn.Sequential` built by the constructor for `num_layers=2` contains 8 modules in order:

1. `dw_0`: Conv2d(256, 256, 3, padding=1, groups=256, bias=False) — layer 0 depthwise
2. `GroupNorm(32, 256)` — layer 0 norm
3. `GELU()` — layer 0 activation
4. `pw_0`: Conv2d(256, 256, 1, bias=True) — layer 0 pointwise, **trunc_normal(0.02)** init
5. `dw_1`: Conv2d(256, 256, 3, padding=1, groups=256, bias=False) — layer 1 depthwise
6. `GroupNorm(32, 256)` — layer 1 norm
7. `GELU()` — layer 1 activation
8. `pw_1`: Conv2d(256, 256, 1, bias=True) — layer 1 pointwise, **zeros_** init (weight and bias)

At init: `pw_1.weight = 0`, `pw_1.bias = 0` → `delta = net(x) = 0` for any `x` → `spatial + delta = spatial`. The residual equals spatial at init regardless of the first layer's output.

Note: the residual is computed at the `_SpatialContextNet` level — it adds `spatial` (the full input) to the output of `self.net(x)` (the output of the entire two-layer sequential). There is **no per-layer residual** inside the sequential. The two-layer stack is applied in sequence before the single residual addition.

---

## Invariants the Builder Must Preserve

1. **Zero-init guarantee:** `spatial_ctx_net(spatial, H, W) == spatial` at init because the final (second) pointwise has weight=0, bias=0 → the entire sequential output is 0.
2. **No per-layer residual inside the sequential:** the two conv layers are stacked without intermediate residuals. The single outer residual `spatial + delta` is the only residual connection.
3. **Spatial token shape invariant:** enrichment returns `(B, H*W, hidden_dim)` — same shape as input.
4. **H, W correctness:** pass `H, W` from `B, C, H, W = feat.shape`. For 640×384 input with stride 16: H=40, W=24.
5. **Config constraint:** no Python `import` statements in `config.py`. All new kwargs are literals.
6. **GroupNorm divisibility:** `hidden_dim=256` divisible by `num_groups=32`. 256/32=8 channels per group. Satisfied.
7. **Loss and output interfaces unchanged.**
8. **`persistent_workers=False`** must remain in both dataloaders (not modified).
9. **AMP safety:** all modules used are AMP-safe at float16.
10. When `use_spatial_ctx=False` (default), the module is not instantiated and forward is unchanged.

---

## Expected Behavior After Change

- At epoch 0 step 0: outputs identical to baseline (zero-init final pointwise ensures delta=0).
- After training starts: the first layer (randomly initialized pointwise) immediately begins learning a 3×3 spatial context transform; the second layer's output is near-zero initially and grows as gradients flow through it.
- Final trained behavior: each spatial token's representation encodes a 5×5 neighborhood (two chained 3×3 convolutions), sufficient to capture an entire forearm (approximately 48 px = 3 grid cells) within the receptive field.
- Parameter overhead: 2 × (2304 + 65536 + 512) = 2 × 68,352 = 136,704 extra parameters ≈ 0.52% of backbone.
- Expected stage-1 `composite_val < 335`. If the wider 5×5 receptive field helps (full-limb structures), this will outperform design002. If the wider receptive field blurs precise single-joint signals, it may underperform design002.
