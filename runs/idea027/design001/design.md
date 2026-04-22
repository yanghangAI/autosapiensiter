**Design Description:** Add single-layer depthwise-separable spatial context conv (no norm) to spatial tokens before cross-attention, with zero-init residual for baseline-equivalent start.

**Starting Point:** `baseline/`

---

## Overview

Add a `_SpatialContextNet` module to `pose3d_transformer_head.py` that reshapes the projected spatial tokens `(B, 960, 256)` to a `(B, 256, 40, 24)` grid, applies a depthwise 3×3 convolution followed by a pointwise 1×1 convolution with GELU activation (no normalization), and adds the result back via a residual connection. The pointwise conv weight and bias are zero-initialized so the module outputs exactly zero at init → training starts from the same state as the baseline.

---

## Algorithm

The core algorithmic change is inserting a depthwise-separable 2D convolution pass over the spatial token grid between the input projection + positional encoding step and the transformer decoder cross-attention. Each spatial token at position (h, w) in the 40×24 grid accumulates information from its 8 immediate neighbors via the 3×3 depthwise filter, then the pointwise 1×1 conv mixes channels. The residual connection (`spatial + delta`) with zero-init on the pointwise weight and bias guarantees the module outputs zero at training start, making epoch-0 behavior identical to baseline.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` — **no changes**.

---

## `pose3d_transformer_head.py` Changes

### 1. Add `_SpatialContextNet` class

Insert before the `_DecoderLayer` class definition. Full class:

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

The variable `H, W` are already available from `B, C, H, W = feat.shape` on the line above. No other changes to `forward()`.

### 4. No changes to `loss()` or `predict()`

---

## `config.py` Changes

In `model.head` dict, add the following keys after `loss_weight_uv=1.0,`:

```python
use_spatial_ctx=True,
spatial_ctx_kernel=3,
spatial_ctx_layers=1,
spatial_ctx_norm='none',
spatial_ctx_act='gelu',
```

All values are bool/int/str literals. No Python import statements. MMEngine config constraint satisfied.

---

## Exact Parameter Values

| Parameter | Value |
|---|---|
| `use_spatial_ctx` | `True` |
| `spatial_ctx_kernel` | `3` |
| `spatial_ctx_layers` | `1` |
| `spatial_ctx_norm` | `'none'` |
| `spatial_ctx_act` | `'gelu'` |
| `spatial_ctx_groups` | not passed (unused when norm='none') |
| depthwise kernel | 3×3, `padding=1`, `groups=hidden_dim`, `bias=False` |
| pointwise kernel | 1×1, `bias=True` |
| pointwise init | `zeros_` (weight and bias) |
| depthwise init | `kaiming_normal_` (fan_out, relu nonlinearity) |
| activation | GELU |
| normalization | none (nn.Identity) |
| residual | `spatial + delta` |
| `zero_init_last` | `True` |

---

## Invariants the Builder Must Preserve

1. **Zero-init guarantee:** at training start, `self.spatial_ctx_net(spatial, H, W)` returns exactly `spatial` (delta=0). This requires zero-init on the **only** (final) pointwise conv weight and bias.
2. **Spatial token shape invariant:** the enrichment module returns `(B, H*W, hidden_dim)` — same shape as input. The decoder receives the same tensor shape as baseline.
3. **H, W correctness:** the `h, w` passed to `spatial_ctx_net.forward()` must match the actual feature map spatial dims. Use `H, W` from `B, C, H, W = feat.shape`. For a 640×384 input with stride 16: H=40, W=24, H*W=960.
4. **Config constraint:** no Python `import` statements in `config.py`. All new kwargs are literals.
5. **Loss and output interfaces unchanged:** `loss()`, `predict()`, and `forward()` output dict shapes are identical to baseline.
6. **`persistent_workers=False`** must remain in both dataloaders (not modified).
7. **AMP safety:** `nn.Conv2d` and `nn.GELU` are AMP-safe at float16.
8. When `use_spatial_ctx=False` (default), the module is not instantiated and forward is unchanged — baseline behavior exactly preserved.

---

## Expected Behavior After Change

- At epoch 0 step 0: outputs identical to baseline (zero-init ensures this).
- After training: each spatial token's key/value in cross-attention encodes a weighted summary of its 8 immediate 2D grid neighbors (3×3 minus self = 8 neighbors, plus self through the residual).
- No change to query initialization, loss function, output projections, or any other component.
- Minimal parameter overhead: 2304 (depthwise) + 65536 (pointwise) = 67,840 extra parameters ≈ 0.26% of backbone.
- Expected stage-1 `composite_val < 345` (baseline is ~355-360 range; best prior is 323.75).
