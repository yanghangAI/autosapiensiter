# Design 001 — Fixed-sigma Gaussian depth gate on cross-attention logits

**Design Description:** Add per-token depth gate to cross-attention logits via two zero-init linear probes (global depth + per-token depth); fixed sigma=1.0; gate is additive in log-space.

**Starting Point:** `baseline/`

---

## Overview

This design injects a soft Gaussian depth-plane gate into the single decoder layer's cross-attention. The algorithm adds per-token depth-consistency weighting to cross-attention logits: two lightweight linear probes are applied to the projected spatial tokens: one pools over all spatial tokens to produce a preliminary body-depth estimate `z_hat (B, 1)`, and one projects each token independently to a per-token depth estimate `z_tok (B, 960)`. The gate logit is `-0.5 * ((z_tok - z_hat) / sigma)^2` (a Gaussian log-probability), added to the cross-attention logits before softmax. This suppresses spatial tokens far from the estimated body depth plane and lets queries focus on body-consistent tokens.

Both probes are zero-initialized so that at step 0 `z_hat = 0` and `z_tok = 0`, giving `gate_logit = 0` everywhere — the gate is flat and the model exactly reproduces baseline cross-attention. The gate emerges as the probes learn during training.

`sigma = 1.0` is a fixed float (not learned). The gate applies uniformly over all joint queries (broadcast over the query dimension), since all body joints are assumed to lie in roughly the same depth plane.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### A. `_DecoderLayer.forward()` — add optional `attn_logit_bias` argument

Change the method signature and cross-attention block:

```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            attn_logit_bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        queries:         (B, num_queries, embed_dim)
        spatial_tokens:  (B, num_spatial, embed_dim)
        attn_logit_bias: (B, num_spatial) additive bias to cross-attn logits,
                         or None for standard cross-attention.
    Returns:
        (B, num_queries, embed_dim)
    """
    # Self-attention (unchanged)
    q = self.norm1(queries)
    q2 = self.self_attn(q, q, q)[0]
    queries = queries + self.dropout1(q2)

    # Cross-attention with optional logit bias
    q = self.norm2(queries)
    if attn_logit_bias is not None:
        B = attn_logit_bias.shape[0]
        N_spatial = attn_logit_bias.shape[1]
        Nq = q.shape[1]
        num_heads = self.cross_attn.num_heads
        # Expand gate to (B*num_heads, Nq, N_spatial) as required by
        # nn.MultiheadAttention with batch_first=True and float attn_mask.
        # (B, N_spatial) -> (B, 1, 1, N_spatial) -> (B, num_heads, Nq, N_spatial)
        # -> (B*num_heads, Nq, N_spatial)
        mask = attn_logit_bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N_spatial)
        mask = mask.expand(B, num_heads, Nq, N_spatial)    # (B, num_heads, Nq, N_spatial)
        mask = mask.reshape(B * num_heads, Nq, N_spatial)  # (B*num_heads, Nq, N_spatial)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens, attn_mask=mask)[0]
    else:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
    queries = queries + self.dropout2(q2)

    # FFN (unchanged)
    queries = queries + self.ffn(self.norm3(queries))
    return queries
```

Key detail: PyTorch's `nn.MultiheadAttention` interprets a float-dtype `attn_mask` as an **additive** bias (not boolean). `mask.dtype` will match the query tensor's dtype (float32 or float16 under AMP). Shape `(B*num_heads, Nq, N_spatial)` is accepted when `batch_first=True`. This is the correct call convention.

#### B. `Pose3dTransformerHead.__init__()` — new kwargs and depth gate modules

Add the following new keyword arguments after `loss_weight_uv`:

```python
depth_gate_type: str = 'none',   # 'none' | 'gaussian'
depth_gate_sigma: float = 1.0,   # fixed bandwidth (Design 001: 1.0)
```

Full updated constructor signature (everything else unchanged):
```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
    loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                  beta=0.05, loss_weight=1.0),
    loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                               beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    depth_gate_type: str = 'none',
    depth_gate_sigma: float = 1.0,
    init_cfg: OptConfigType = None,
):
```

Inside `__init__`, after building `self.uv_out`, add:

```python
self.depth_gate_type = depth_gate_type
if depth_gate_type == 'gaussian':
    # Global depth probe: pool all spatial tokens → scalar body depth estimate
    self.depth_probe_global = nn.Linear(hidden_dim, 1)
    # Per-token depth probe: each spatial token → scalar depth estimate
    self.depth_probe_token = nn.Linear(hidden_dim, 1)
    # Fixed sigma as a non-learned buffer
    self.register_buffer(
        'depth_gate_sigma_buf',
        torch.tensor(depth_gate_sigma, dtype=torch.float32))
```

#### C. `_init_head_weights()` — zero-init depth probes

Append to the existing `_init_head_weights` method (after existing init code):

```python
if self.depth_gate_type == 'gaussian':
    # Zero init: gate is flat at step 0 → identical to baseline
    nn.init.zeros_(self.depth_probe_global.weight)
    nn.init.zeros_(self.depth_probe_global.bias)
    nn.init.zeros_(self.depth_probe_token.weight)
    nn.init.zeros_(self.depth_probe_token.bias)
```

#### D. `forward()` — compute gate and pass to decoder

After `spatial = spatial + pos_enc` and before `decoded = self.decoder_layer(...)`:

```python
attn_logit_bias = None
if self.depth_gate_type == 'gaussian':
    # z_hat: preliminary body depth estimate from global spatial pool
    z_hat = self.depth_probe_global(spatial.mean(dim=1))  # (B, 1)
    # z_tok: per-token depth estimate
    z_tok = self.depth_probe_token(spatial).squeeze(-1)    # (B, H*W)
    sigma = self.depth_gate_sigma_buf                      # scalar
    depth_err = (z_tok - z_hat) / sigma                    # (B, H*W)
    attn_logit_bias = -0.5 * depth_err ** 2               # (B, H*W), ≤ 0

decoded = self.decoder_layer(queries, spatial, attn_logit_bias=attn_logit_bias)
```

The existing code `decoded = self.decoder_layer(queries, spatial)` becomes the line above. All code after `decoded` (output projections, return) is unchanged.

#### E. `loss()` — no changes needed

No auxiliary depth probe loss in Design 001. The `loss()` method is unchanged.

---

### 2. `config.py`

In `model.head`, add two new keyword arguments as float/str literals:

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
    depth_gate_type='gaussian',
    depth_gate_sigma=1.0,
),
```

All other config sections (optimizer, LR schedule, data pipeline, hooks, backbone) are **identical to baseline**.

---

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Budget

- `depth_probe_global`: `Linear(256, 1)` = 256 + 1 = **257 parameters**
- `depth_probe_token`: `Linear(256, 1)` = 256 + 1 = **257 parameters**
- Total new parameters: **514**

No additional memory for the gate computation: `(B, H*W)` = `(4, 960)` scalars — negligible.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Output tensor shapes from `forward()` are **unchanged**: `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)`.
3. `depth_gate_type='none'` (the default) must produce **exactly** the baseline behavior: `attn_logit_bias=None` is passed, the `else` branch in `_DecoderLayer.forward()` is taken, and the standard `cross_attn(q, k, v)` call is made.
4. The `attn_mask` passed to `nn.MultiheadAttention` **must** be a float tensor (not bool). A float mask is interpreted as additive. Do not cast to bool.
5. `mask.shape` must be `(B*num_heads, Nq, N_spatial)` = `(4*8, 70, 960)` = `(32, 70, 960)` for batch size 4. If AMP casts queries to float16, the mask must also be in float16. PyTorch's MHA handles this automatically (it casts `attn_mask` to the query dtype internally); no manual cast needed.
6. `attn_logit_bias` values are `≤ 0` (Gaussian log-gate). This is numerically safe under AMP: large negative values → softmax weight near 0 (desired), not overflow.
7. `depth_gate_sigma_buf` is a buffer (non-parameter). It appears in `state_dict` but is not optimised. This is correct for a fixed hyperparameter.
8. Zero-init of both probes: at step 0, `z_hat = 0`, `z_tok = 0`, `depth_err = 0`, `attn_logit_bias = 0` everywhere. The gate is flat and the model is identical to baseline at initialisation.
9. The `_depth_probe_z_hat` attribute caching pattern from idea018/idea.md is **not needed** for Design 001 (no auxiliary loss in `loss()`). Do not add it.
10. `self._train_mpjpe` and `self._train_mpjpe_abs` computations in `loss()` are unchanged.
11. AMP is ON via `FixedAmpOptimWrapper`. The depth gate computation (`z_hat`, `z_tok`, `depth_err`, `attn_logit_bias`) runs in the AMP context under float16. The operations are: `mean(dim=1)` → `Linear` → `squeeze` → `Linear` → subtraction → division → squaring → negation. All are AMP-safe.
12. MMEngine config constraint: `depth_gate_type='gaussian'` (str literal), `depth_gate_sigma=1.0` (float literal). No Python import statements. Compliant.
13. `backbone`, `data_preprocessor`, `bedlam_metric`, `bedlam2_dataset`, `bedlam2_transforms`, `train.py`, `pelvis_utils.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` are invariant — do not touch.

---

## Expected Behavior After Change

- At step 0: gate is flat (all zeros), model identical to baseline. Loss values at epoch 0 match baseline.
- As training progresses: `depth_probe_global` learns to output a rough body depth (meters scale); `depth_probe_token` learns to distinguish body-region tokens (depth close to pelvis) from background tokens (depth far from pelvis). Gate logits become negative for background tokens.
- Cross-attention softmax weights become concentrated on body-plane tokens; background tokens at floor/wall depths are suppressed.
- `loss/joints/train` and `loss/depth/train` are the only losses (same keys as baseline).
- Target composite_val stage-1: < 330. Target composite_val stage-2: < 215.
- `mpjpe_abs` at stage-2: target < 480 mm.
