**Design Description:** Global average-pool FiLM conditioning on spatial tokens (γ·h + β) before cross-attention, identity-initialised.

**Starting Point:** `baseline/`

---

## Overview

Insert a FiLM (Feature-wise Linear Modulation) layer after `input_proj + pos_enc` in the decoder forward pass. The FiLM algorithm: a 2-layer MLP takes the global average-pooled spatial tokens `(B, hidden_dim)` and produces per-channel affine parameters `(γ, β) ∈ R^{hidden_dim}` that modulate all 960 spatial tokens before cross-attention. The MLP output layer is zero-initialised so that training starts at the identity transform (γ=1, β=0) — identical to baseline at step 0.

**Files changed:**
- `pose3d_transformer_head.py` — add `film_pool_type`, `film_hidden_dim` constructor args; add `self.film_net`; apply FiLM in `forward()`
- `config.py` — add `film_pool_type='avg'`, `film_hidden_dim=128` to the `head` dict

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

After the existing `self.loss_weight_uv = loss_weight_uv` line and before `self.loss_joints_module = MODELS.build(loss_joints)`, store the new args and build the FiLM network:

```python
self.film_pool_type = film_pool_type

if film_pool_type == 'avg':
    film_in_dim = hidden_dim
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

Note: `film_hidden_dim=128` is stored only implicitly via the `film_net` layer sizes; no need to store it as `self.film_hidden_dim`.

### `_init_head_weights()` — NO changes required

The zero-init of `film_net[-1]` is already done in `__init__` above. The existing `_init_head_weights()` method remains unchanged.

### `forward()` changes

Insert the FiLM modulation block **immediately after** `spatial = spatial + pos_enc` and **immediately before** the `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)` line.

Exact insertion:

```python
# FiLM conditioning: modulate spatial tokens with a global scale embedding
if self.film_pool_type == 'avg':
    ctx = spatial.mean(dim=1)                        # (B, hidden_dim)
    film = self.film_net(ctx)                        # (B, 2*hidden_dim)
    gamma, beta = film.chunk(2, dim=-1)              # each (B, hidden_dim)
    gamma = gamma + 1.0                              # residual: identity at init
    spatial = spatial * gamma.unsqueeze(1) + beta.unsqueeze(1)  # (B, 960, hidden_dim)
```

The existing line `queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)` follows immediately after, unchanged. The rest of `forward()` is unchanged.

### `loss()` and `predict()` — NO changes

Both call `self.forward(feats)`, which now includes FiLM internally. Output dict shape is unchanged: `{'joints': (B, 70, 3), 'pelvis_depth': (B, 1), 'pelvis_uv': (B, 2)}`.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add two new keys:

```python
film_pool_type='avg',
film_hidden_dim=128,
```

Full updated `head` dict for reference (only the two new lines are additions; all other values remain baseline):

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
    film_pool_type='avg',
    film_hidden_dim=128,
),
```

All other config values (optimizer, scheduler, data pipeline, hooks, seeds) are identical to baseline.

---

## Exact Behaviour Specification

1. **At initialisation (step 0):** `film_net[-1].weight == 0`, `film_net[-1].bias == 0`. Therefore `film(ctx) == 0` → `gamma = 0 + 1.0 = 1.0`, `beta = 0.0`. FiLM becomes `spatial * 1.0 + 0.0 = spatial`. Training starts at exact baseline configuration.

2. **During training:** `film_net` learns to produce non-trivial `(gamma, beta)` that rescale and shift the spatial tokens per-channel, conditioned on the global average of all 960 tokens. The gradient flows through both the pooling path (`∂loss/∂ctx → ∂loss/∂spatial`) and the modulated path (`∂loss/∂(spatial * gamma + beta) → ∂loss/∂spatial`).

3. **Spatial token count:** The feature map is `(B, 1024, H', W')`. For `img_h=640, img_w=384`, the backbone outputs `H'=40, W'=24` → 960 tokens. FiLM applies uniformly to all 960 tokens.

4. **AMP compatibility:** `mean(dim=1)`, `chunk`, element-wise multiply/add are all float16-safe. The near-zero init prevents overflow at step 0.

5. **Parameter count added:** `256×128 + 128×512 = 32768 + 65536 = 98304 ≈ 98K` parameters. Negligible on a 300M parameter model.

6. **Memory footprint:** One `(B, 256)` context tensor (pool output) + one `(B, 512)` film output tensor + the modulated `(B, 960, 256)` spatial tensor (same shape as before, overwritten in place). No additional large tensor allocations.

---

## Constraints and Invariants the Builder Must Preserve

- `film_pool_type` is a `str` literal in config; `film_hidden_dim` is an `int` literal. No Python imports in config.
- The FiLM block operates between `spatial + pos_enc` and `decoder_layer(queries, spatial)`. Order is strictly: `input_proj → pos_enc → FiLM → decoder_layer`.
- When `film_pool_type='none'` (default), no `film_net` is created and no FiLM is applied. The default constructor kwargs keep backward compatibility.
- Loss is still restricted to body joints indices 0-21. Loss computation unchanged.
- `persistent_workers=False` unchanged.
- `resume=True`, `max_keep_ckpts=1` unchanged.
- Seed 2026 unchanged.
- The `in_channels` passed to `Pose3dTransformerHead` remains 1024 (`embed_dim`); `hidden_dim` remains 256.
- `film_hidden_dim=128` is the bottleneck dimension of the FiLM MLP (first linear: `256 → 128`, second linear: `128 → 512 = 2*256`).
