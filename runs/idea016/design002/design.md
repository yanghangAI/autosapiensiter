**Design Description:** Dual-pool FiLM conditioning (avg + max concatenated) on spatial tokens before cross-attention, identity-initialised.

**Starting Point:** `baseline/`

---

## Overview

Insert a FiLM layer after `input_proj + pos_enc`. The FiLM algorithm applied here uses a dual-pool conditioning signal: the concatenation of global average pool and global max pool over the 960 spatial tokens → `(B, 2*hidden_dim)`. A 2-layer MLP maps this to `(γ, β) ∈ R^{hidden_dim}` for per-channel modulation of all spatial tokens before cross-attention. Max pool captures peak activations (salient foreground/body tokens); avg pool captures global scene context. Together they form a richer depth-scale descriptor than avg-only (Design 001). The output layer of the MLP is zero-initialised for identity-at-init.

**Files changed:**
- `pose3d_transformer_head.py` — add `film_pool_type`, `film_hidden_dim` constructor args; add `self.film_net`; apply FiLM in `forward()`
- `config.py` — add `film_pool_type='avg_max'`, `film_hidden_dim=128` to the `head` dict

**Files NOT changed:** `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, `train.py`

---

## `pose3d_transformer_head.py` Changes

### Constructor signature addition

Add two new keyword arguments to `Pose3dTransformerHead.__init__`, after `loss_weight_uv`:

```python
film_pool_type: str = 'none',
film_hidden_dim: int = 128,
```

### Constructor body additions

After `self.loss_weight_uv = loss_weight_uv` and before `self.loss_joints_module = MODELS.build(loss_joints)`:

```python
self.film_pool_type = film_pool_type

if film_pool_type == 'avg_max':
    film_in_dim = 2 * hidden_dim   # avg_pool concat max_pool: (B, 512)
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

For `film_pool_type='avg_max'` and `hidden_dim=256`, `film_hidden_dim=128`:
- First linear: `512 → 128` (weight: 512×128 = 65536 params)
- Second linear: `128 → 512` (weight: 128×512 = 65536 params)
- Total added: ~131K parameters

### `forward()` changes

Insert the FiLM modulation block **immediately after** `spatial = spatial + pos_enc` and **immediately before** `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)`:

```python
# FiLM conditioning: dual-pool (avg+max) scale embedding
if self.film_pool_type == 'avg_max':
    ctx_avg = spatial.mean(dim=1)                    # (B, hidden_dim)
    ctx_max = spatial.max(dim=1).values              # (B, hidden_dim)
    ctx = torch.cat([ctx_avg, ctx_max], dim=-1)      # (B, 2*hidden_dim)
    film = self.film_net(ctx)                        # (B, 2*hidden_dim)
    gamma, beta = film.chunk(2, dim=-1)              # each (B, hidden_dim)
    gamma = gamma + 1.0                              # residual: identity at init
    spatial = spatial * gamma.unsqueeze(1) + beta.unsqueeze(1)  # (B, 960, hidden_dim)
```

The rest of `forward()` is unchanged.

### `loss()` and `predict()` — NO changes

Both call `self.forward(feats)`. Output dict shape unchanged.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add:

```python
film_pool_type='avg_max',
film_hidden_dim=128,
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
    film_pool_type='avg_max',
    film_hidden_dim=128,
),
```

All other config values identical to baseline.

---

## Exact Behaviour Specification

1. **At initialisation (step 0):** `film_net[-1].weight == 0`, `film_net[-1].bias == 0`. Therefore `film(ctx) == 0` → `gamma = 1.0`, `beta = 0.0`. Spatial tokens pass through unchanged. Training starts at exact baseline.

2. **Dual pooling:** `ctx_avg = spatial.mean(dim=1)` computes the mean over all 960 tokens, capturing the global scene context. `ctx_max = spatial.max(dim=1).values` takes the element-wise maximum over tokens, capturing the highest-activated (most salient/foreground) token per channel. Concatenation gives the MLP access to both global and peak information simultaneously.

3. **Gradient flow:** Gradients flow through both pool operations. `max(dim=1)` is differentiable (subgradient at the argmax); `mean(dim=1)` is fully differentiable.

4. **Spatial token count:** 960 tokens (H'=40, W'=24 → 40×24 = 960). Note: the backbone produces H'=40 for height dim (img_h=640, patch=16 → 640/16=40) and W'=24 (img_w=384, patch=16 → 384/16=24). FiLM applies uniformly to all 960 tokens.

5. **AMP compatibility:** `mean`, `max`, `cat`, `chunk`, element-wise multiply/add are all float16-safe.

6. **Memory footprint:** Two `(B, 256)` pool tensors + one `(B, 512)` concat + one `(B, 512)` film output + the modulated `(B, 960, 256)` spatial tensor. No additional large allocations.

7. **MLP dims for `film_pool_type='avg_max'`, `film_hidden_dim=128`, `hidden_dim=256`:**
   - `film_net[0]`: `Linear(512, 128)` — 65536 + 128 params
   - `film_net[2]`: `Linear(128, 512)` — 65536 + 512 params
   - Total: ~131K parameters

---

## Constraints and Invariants the Builder Must Preserve

- `film_pool_type='avg_max'` is a `str` literal in config. No Python imports in config.
- `film_hidden_dim=128` is an `int` literal in config.
- The FiLM block is inserted between `spatial + pos_enc` and `decoder_layer(queries, spatial)`. Order: `input_proj → pos_enc → FiLM → decoder_layer`.
- When `film_pool_type='none'` (default), no `film_net` is created and no FiLM is applied. Backward-compatible default.
- The MLP input dim for `avg_max` is `2 * hidden_dim = 512` (not `2 * film_hidden_dim`). The bottleneck is `film_hidden_dim=128`. The output is always `2 * hidden_dim = 512` (for `gamma` and `beta` each of dim `hidden_dim=256`).
- `spatial.max(dim=1).values` — use `.values` attribute (PyTorch `max` over a dim returns a named tuple; must access `.values` to get the tensor).
- Loss restricted to body joints 0-21 unchanged.
- `persistent_workers=False`, `resume=True`, `max_keep_ckpts=1`, seed 2026 all unchanged.
