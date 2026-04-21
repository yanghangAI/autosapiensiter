# Design 002 — Learnable-sigma Gaussian depth gate with auxiliary depth probe loss

**Design Description:** Depth-gated cross-attention with learnable log-sigma (init=0 → sigma=1) and auxiliary smooth-L1 loss (weight 0.1) on the global depth probe output.

**Starting Point:** `baseline/`

---

## Overview

Same algorithm as Design 001 (two linear depth probes producing per-token gate logits applied to cross-attention) with two changes:

1. **Learnable sigma**: `sigma = exp(log_sigma)` where `log_sigma` is an `nn.Parameter` initialised to `0.0` (so `sigma_init = 1.0`). Sigma is clamped to `>= 0.01` during forward to prevent numerical collapse. The model learns the optimal depth-gate bandwidth: small sigma → tight depth-plane selection; large sigma → relaxed gate (approaches baseline).

2. **Auxiliary depth probe loss**: a small smooth-L1 loss `L_probe = 0.1 * smooth_l1(z_hat, gt_depth)` on the global probe's output `z_hat (B, 1)` vs. the ground-truth pelvis depth. This loss provides a direct training signal to `depth_probe_global` from epoch 1, ensuring the probe converges to a useful depth estimate rather than relying solely on indirect gradients through the gate.

Both probes remain zero-initialized (flat gate at step 0 = baseline). `log_sigma` is initialised to 0.0 separately.

The auxiliary loss is **training-only** — no effect on inference or evaluation.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### A. `_DecoderLayer.forward()` — add optional `attn_logit_bias` argument

Identical to Design 001. Change the method signature and cross-attention block:

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

The `attn_mask` is a float tensor; PyTorch MHA treats it as additive. Shape `(B*num_heads, Nq, N_spatial)` is required for per-sample batched masks with `batch_first=True`.

#### B. `Pose3dTransformerHead.__init__()` — new kwargs and depth gate modules

Add the following new keyword arguments after `loss_weight_uv`:

```python
depth_gate_type: str = 'none',           # 'none' | 'gaussian_learnable_sigma'
depth_probe_loss_weight: float = 0.0,    # auxiliary probe loss weight (Design 002: 0.1)
```

Full updated constructor signature:
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
    depth_probe_loss_weight: float = 0.0,
    init_cfg: OptConfigType = None,
):
```

Inside `__init__`, after building `self.uv_out`, add:

```python
self.depth_gate_type = depth_gate_type
self.depth_probe_loss_weight = depth_probe_loss_weight
if depth_gate_type == 'gaussian_learnable_sigma':
    # Global depth probe: pool all spatial tokens → scalar body depth estimate
    self.depth_probe_global = nn.Linear(hidden_dim, 1)
    # Per-token depth probe: each spatial token → scalar depth estimate
    self.depth_probe_token = nn.Linear(hidden_dim, 1)
    # Learnable log-sigma: initialised to 0.0 → sigma = exp(0) = 1.0
    self.log_sigma = nn.Parameter(torch.zeros(1))
```

#### C. `_init_head_weights()` — zero-init depth probes

Append to the existing `_init_head_weights` method:

```python
if self.depth_gate_type == 'gaussian_learnable_sigma':
    # Zero init: gate is flat at step 0 → identical to baseline
    nn.init.zeros_(self.depth_probe_global.weight)
    nn.init.zeros_(self.depth_probe_global.bias)
    nn.init.zeros_(self.depth_probe_token.weight)
    nn.init.zeros_(self.depth_probe_token.bias)
    # log_sigma is already initialised to 0.0 in __init__ via torch.zeros(1)
```

#### D. `forward()` — compute gate and pass to decoder; cache z_hat

After `spatial = spatial + pos_enc` and before `decoded = self.decoder_layer(...)`:

```python
attn_logit_bias = None
if self.depth_gate_type == 'gaussian_learnable_sigma':
    z_hat = self.depth_probe_global(spatial.mean(dim=1))  # (B, 1)
    z_tok = self.depth_probe_token(spatial).squeeze(-1)    # (B, H*W)
    sigma = torch.exp(self.log_sigma).clamp(min=0.01)     # scalar, always > 0
    depth_err = (z_tok - z_hat) / sigma                    # (B, H*W)
    attn_logit_bias = -0.5 * depth_err ** 2               # (B, H*W), ≤ 0
    # Cache z_hat for auxiliary loss in loss() — same pattern as _train_mpjpe
    self._depth_probe_z_hat = z_hat

decoded = self.decoder_layer(queries, spatial, attn_logit_bias=attn_logit_bias)
```

All code after `decoded` (output projections, return dict) is unchanged.

#### E. `loss()` — add auxiliary depth probe loss

After the existing `losses['loss/uv/train'] = ...` line, add:

```python
# Auxiliary depth probe loss — trains depth_probe_global to predict pelvis depth
if self.depth_probe_loss_weight > 0.0 and hasattr(self, '_depth_probe_z_hat'):
    losses['loss/depth_probe/train'] = self.depth_probe_loss_weight * self.loss_depth_module(
        self._depth_probe_z_hat, gt_depth)
```

`gt_depth` is already computed earlier in `loss()` (the ground-truth pelvis depth tensor, shape `(B, 1)`). `self.loss_depth_module` is the existing `SoftWeightSmoothL1Loss` instance — reused, no new module.

The `self._train_mpjpe` and `self._train_mpjpe_abs` computations after this block are unchanged.

---

### 2. `config.py`

In `model.head`, add two new keyword arguments as str/float literals:

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
    depth_gate_type='gaussian_learnable_sigma',
    depth_probe_loss_weight=0.1,
),
```

All other config sections (optimizer, LR schedule, data pipeline, hooks, backbone) are **identical to baseline**.

---

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Budget

- `depth_probe_global`: `Linear(256, 1)` = **257 parameters**
- `depth_probe_token`: `Linear(256, 1)` = **257 parameters**
- `log_sigma`: `nn.Parameter` scalar = **1 parameter**
- Total new parameters: **515**

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Output tensor shapes from `forward()` unchanged: `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)`.
3. `depth_gate_type='none'` (default) must produce exactly the baseline behavior: `attn_logit_bias=None`, standard cross-attention.
4. `attn_mask` passed to `nn.MultiheadAttention` must be a **float** tensor (additive interpretation). Do not cast to bool.
5. `mask.shape` must be `(B*num_heads, Nq, N_spatial)`. For training batch size 4, num_heads=8, num_joints=70, N_spatial=960: shape is `(32, 70, 960)`.
6. `sigma = exp(log_sigma).clamp(min=0.01)` — the clamp ensures sigma is always strictly positive, preventing division by zero in `depth_err`. The clamp is applied **inside forward()** every step.
7. `log_sigma` is an `nn.Parameter` and will be optimized by AdamW with the same learning rate as the head parameters (no custom `lr_mult` in `paramwise_cfg` for it — that is correct).
8. `self._depth_probe_z_hat` is set in `forward()` and read in `loss()`. `loss()` always calls `forward()` first (via `pred = self.forward(feats)`), so the attribute is always set before it is read. The `hasattr` guard is a safety check only.
9. The auxiliary loss key `'loss/depth_probe/train'` will appear in the MMEngine training log. It does NOT affect `composite_val` (which depends only on mpjpe metrics from the evaluator). It does contribute to the total training loss and affects parameter gradients.
10. `self.loss_depth_module` is reused for the probe loss. It is a `SoftWeightSmoothL1Loss` with `beta=0.05, loss_weight=1.0`. The final probe loss is `0.1 * loss_depth_module(z_hat, gt_depth)` — the `loss_weight=1.0` inside the module is multiplied by the external `depth_probe_loss_weight=0.1`, giving an effective weight of 0.1 relative to the main depth loss.
11. Zero-init of both probes: at step 0, gate is flat (all zeros), model is identical to baseline.
12. `log_sigma` initialised to `0.0` in `__init__` via `nn.Parameter(torch.zeros(1))`. After the first gradient step, `log_sigma` will adapt.
13. MMEngine config constraint: `depth_gate_type='gaussian_learnable_sigma'` (str literal), `depth_probe_loss_weight=0.1` (float literal). No Python import statements. Compliant.
14. `backbone`, `data_preprocessor`, `bedlam_metric`, `bedlam2_dataset`, `bedlam2_transforms`, `train.py`, `pelvis_utils.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` are invariant — do not touch.

---

## Expected Behavior After Change

- At step 0: gate is flat, model identical to baseline. `log_sigma` = 0.0, `sigma` = 1.0.
- `loss/depth_probe/train` appears from step 1. Initial value will be large (probe outputs 0, gt_depth is 2–8m); it decreases as `depth_probe_global` learns.
- `log_sigma` adapts: if depth gating helps, sigma decreases (tighter gate); if the depth spread in BEDLAM2 is large relative to body thickness, sigma may stabilize at a moderate value.
- Cross-attention learns to concentrate on body-plane tokens, with adaptive bandwidth.
- Target composite_val stage-1: < 328. Target composite_val stage-2: < 212.
- `mpjpe_abs` at stage-2: target < 460 mm (better than Design 001 due to supervised probe convergence).
