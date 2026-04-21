# Design 003 — Depth-gated cross-attention combined with 22-query body-only decoder

**Design Description:** 22 body-only joint queries + linear hand recovery (aux weight 0.1) + fixed-sigma Gaussian depth gate (sigma=1.0) on cross-attention logits; both mechanisms combined.

**Starting Point:** `baseline/`

---

## Overview

This design compositionally combines two independently validated algorithm mechanisms:

1. **22-query body-only decoder** (from idea008/design002): replaces 70 joint queries with 22 body-only queries; hand joints (indices 22–69) are predicted via a linear projection `hand_proj: Linear(22*256, 48*3)` from the flattened body query features; a small auxiliary hand loss (weight 0.1) provides regularising gradient.

2. **Fixed-sigma Gaussian depth gate** (from idea018/design001): two zero-init linear probes produce a per-token depth gate logit added to cross-attention logits before softmax, suppressing spatially inconsistent background tokens.

The combination is architecturally clean: with 22 body queries, the cross-attention is `(B, 22, 256) × (B, 960, 256)`. The depth gate `(B, 960)` broadcasts over the 22 query dimension uniformly (all 22 body queries share the same depth gate — appropriate since all body joints lie in roughly the same depth plane). The gate adds no query-specific logic.

Both mechanisms are zero-initialized relative to the baseline: the depth probes are zero-init (flat gate at step 0), and the body-only decoder's `hand_proj` starts from a near-zero trunc-normal init. At step 0 the model is equivalent to the baseline.

**Why this combination**: idea008/design002 achieved the best prior stage-2 composite (241.14 mm) by restricting the query side. Design 003 additionally restricts the **token side** by depth plane — the two restrictions are orthogonal.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### A. `_DecoderLayer.forward()` — add optional `attn_logit_bias` argument

Identical to Designs 001 and 002:

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

The `attn_mask` is float; PyTorch MHA treats it as additive. Shape `(B*num_heads, Nq, N_spatial)` is required.

#### B. `Pose3dTransformerHead.__init__()` — combined new kwargs and modules

Add the following new keyword arguments after `loss_weight_uv`:

```python
num_body_queries: int = 70,          # set to 22 in config for this design
hand_aux_loss_weight: float = 0.0,   # set to 0.1 in config for this design
depth_gate_type: str = 'none',       # set to 'gaussian' in config for this design
depth_gate_sigma: float = 1.0,       # fixed bandwidth
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
    num_body_queries: int = 70,
    hand_aux_loss_weight: float = 0.0,
    depth_gate_type: str = 'none',
    depth_gate_sigma: float = 1.0,
    init_cfg: OptConfigType = None,
):
```

Inside `__init__`, store all new attributes and build modules. Replace the existing `self.joint_queries = nn.Embedding(num_joints, hidden_dim)` with:

```python
self.num_body_queries = num_body_queries
self.hand_aux_loss_weight = hand_aux_loss_weight
# Use num_body_queries for the embedding; when == num_joints, identical to baseline
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

Add hand projection (only created when `num_body_queries < num_joints`). After `self.uv_out`:

```python
if num_body_queries < num_joints:
    # Linear projection: flattened body query features → hand joint coords
    # Input: num_body_queries * hidden_dim = 22 * 256 = 5632
    # Output: (num_joints - num_body_queries) * 3 = 48 * 3 = 144
    self.hand_proj = nn.Linear(
        num_body_queries * hidden_dim,
        (num_joints - num_body_queries) * 3)
else:
    self.hand_proj = None
```

Add depth gate modules (after `self.hand_proj` assignment):

```python
self.depth_gate_type = depth_gate_type
if depth_gate_type == 'gaussian':
    self.depth_probe_global = nn.Linear(hidden_dim, 1)
    self.depth_probe_token = nn.Linear(hidden_dim, 1)
    self.register_buffer(
        'depth_gate_sigma_buf',
        torch.tensor(depth_gate_sigma, dtype=torch.float32))
```

#### C. `_init_head_weights()` — zero-init depth probes; trunc-normal init hand_proj

Append to `_init_head_weights`:

```python
# Hand projection init
if self.hand_proj is not None:
    nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
    nn.init.zeros_(self.hand_proj.bias)

# Depth probe zero-init (flat gate at step 0)
if self.depth_gate_type == 'gaussian':
    nn.init.zeros_(self.depth_probe_global.weight)
    nn.init.zeros_(self.depth_probe_global.bias)
    nn.init.zeros_(self.depth_probe_token.weight)
    nn.init.zeros_(self.depth_probe_token.bias)
```

#### D. `forward()` — compute gate and produce hand joints

After `spatial = spatial + pos_enc` and before the decoder call:

```python
attn_logit_bias = None
if self.depth_gate_type == 'gaussian':
    z_hat = self.depth_probe_global(spatial.mean(dim=1))  # (B, 1)
    z_tok = self.depth_probe_token(spatial).squeeze(-1)    # (B, H*W)
    sigma = self.depth_gate_sigma_buf                      # scalar
    depth_err = (z_tok - z_hat) / sigma                    # (B, H*W)
    attn_logit_bias = -0.5 * depth_err ** 2               # (B, H*W), ≤ 0

decoded = self.decoder_layer(queries, spatial, attn_logit_bias=attn_logit_bias)
# decoded: (B, num_body_queries, hidden_dim) = (B, 22, 256)
```

Replace the existing joints output block with the body+hand combination:

```python
# Body joints from decoder output
body_joints = self.joints_out(decoded)  # (B, num_body_queries, 3) = (B, 22, 3)

# Hand joint recovery via linear projection of flattened body features
if self.hand_proj is not None:
    body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)  # (B, 5632)
    num_hand = self.num_joints - self.num_body_queries  # 48
    hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)           # (B, 48, 3)
    joints = torch.cat([body_joints, hand_joints], dim=1)                     # (B, 70, 3)
else:
    joints = body_joints  # (B, 70, 3) when num_body_queries == num_joints
```

Pelvis token and output projections (unchanged logic, but now referencing `decoded` which has `num_body_queries` rows):

```python
pelvis_token = decoded[:, 0, :]          # (B, hidden_dim) — still token 0
pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
pelvis_uv = self.uv_out(pelvis_token)        # (B, 2)

return {
    'joints': joints,            # (B, 70, 3)
    'pelvis_depth': pelvis_depth,
    'pelvis_uv': pelvis_uv,
}
```

#### E. `loss()` — add auxiliary hand loss

After existing `losses['loss/uv/train'] = ...`, add:

```python
# Auxiliary hand loss — anchors hand_proj in pose space; does not affect composite metric
if self.hand_aux_loss_weight > 0.0 and self.hand_proj is not None:
    _HAND = list(range(self.num_body_queries, self.num_joints))  # [22, ..., 69]
    losses['loss/hand_aux/train'] = self.hand_aux_loss_weight * self.loss_joints_module(
        pred['joints'][:, _HAND], gt_joints[:, _HAND])
```

`self._train_mpjpe` and `self._train_mpjpe_abs` computations are unchanged.

---

### 2. `config.py`

In `model.head`, add four new keyword arguments as int/float/str literals:

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
    num_body_queries=22,
    hand_aux_loss_weight=0.1,
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

- `depth_probe_global`: `Linear(256, 1)` = **257 parameters**
- `depth_probe_token`: `Linear(256, 1)` = **257 parameters**
- `hand_proj`: `Linear(5632, 144)` = 5632 × 144 + 144 = **810,576 parameters**
- Removed from baseline: 48 query embeddings (22 vs 70) = -48 × 256 = **-12,288 parameters**
- Net new parameters: ~**798,802 parameters**

Cross-attention compute is lower than baseline: 22 × 960 = 21,120 query-token pairs vs. 70 × 960 = 67,200 in baseline. The depth gate adds 2 × 960 scalars per sample — negligible.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Output `joints` shape from `forward()` must be `(B, 70, 3)` — always. The concatenation `cat([body_joints, hand_joints], dim=1)` must produce exactly 70 joint rows.
3. `num_body_queries=22`, `num_joints=70` → `num_hand = 70 - 22 = 48`. All three values are dynamic: compute as `(num_joints - num_body_queries)` in `__init__` and `forward()`. Do not hardcode 48.
4. `self.num_joints = 70` must remain unchanged (set in `BaseHead.__init__` or explicitly) so that `predict()` works.
5. `self.joint_queries` embedding has size `num_body_queries = 22` — not 70. The `queries` tensor in `forward()` has shape `(B, 22, hidden_dim)`.
6. Pelvis token is `decoded[:, 0, :]` (the first of the 22 body query outputs). This is appropriate because the first body query (hip/pelvis region) is expected to carry the most depth information. Unchanged from idea008/design002.
7. `hand_proj` input is `decoded.reshape(B, num_body_queries * hidden_dim)` — the entire 22-query decoded output, flattened. Do not use only a subset of query features.
8. `_BODY = list(range(0, 22))` for body joint loss — unchanged.
9. `_HAND = list(range(22, 70))` for auxiliary hand loss — uses `self.num_body_queries` dynamically: `list(range(self.num_body_queries, self.num_joints))`.
10. The auxiliary hand loss key `'loss/hand_aux/train'` appears in training log but does not affect `composite_val`.
11. The depth gate: `attn_logit_bias` shape is `(B, H*W) = (B, 960)`. The expansion in `_DecoderLayer.forward()` is `(B, num_heads, Nq, N_spatial) = (B, 8, 22, 960)`, then reshaped to `(B*8, 22, 960) = (32, 22, 960)` for batch size 4.
12. Both depth probes are zero-initialized; `hand_proj` is trunc-normal initialized (std=0.02). At step 0: gate is flat (all zeros), body decoder runs like a 22-query baseline, hand joints come from a near-zero `hand_proj`.
13. `attn_mask` must be a float tensor (not bool). PyTorch MHA interprets it as additive bias.
14. MMEngine config constraint: `num_body_queries=22` (int literal), `hand_aux_loss_weight=0.1` (float literal), `depth_gate_type='gaussian'` (str literal), `depth_gate_sigma=1.0` (float literal). No Python import statements. Compliant.
15. `backbone`, `data_preprocessor`, `bedlam_metric`, `bedlam2_dataset`, `bedlam2_transforms`, `train.py`, `pelvis_utils.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` are invariant — do not touch.
16. When `num_body_queries == num_joints` (default 70), `hand_proj is None` and the model produces `joints` directly from `joints_out(decoded)` — exactly the baseline. No conditional logic branches are exercised in the default case.
17. AMP ON: all depth gate operations are float16-safe. `hand_proj` runs in AMP context correctly (linear layers are auto-cast).

---

## Expected Behavior After Change

- Training log: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train` (four loss keys).
- The decoder operates on 22 body queries with depth-gated cross-attention. Both the query-side restriction (22 instead of 70) and the token-side restriction (depth gate) are active simultaneously.
- If both mechanisms are independently beneficial, their combination should outperform either alone.
- idea008/design002 achieved stage-1 composite 333.63, stage-2 composite 241.14.
- idea018/design001 targets stage-1 composite < 330.
- Design 003 target: stage-1 composite < 320, stage-2 composite < 230.
- `mpjpe_abs` at stage-2: target < 440 mm (best prior: 533 mm — idea008/design002).
- Body MPJPE target stage-2: < 170 mm (best prior: 156 mm — idea002/design003).
