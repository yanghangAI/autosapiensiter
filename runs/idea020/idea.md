**Idea Name:** Focal Cross-Attention via Learnable Per-Query Temperature Scaling

**Approach:** Add one learnable scalar temperature τ_i per joint query (70 parameters total, initialised to 1.0) that scales the cross-attention logits before softmax — `attn_logits / τ_i` — allowing each joint to independently control whether it attends sharply to a small spatial region or broadly over many tokens, so that distal joints (wrists, ankles) can learn sharp focal attention while proximal joints (pelvis, spine) retain diffuse global context.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Homogeneous Temperature Problem

In the baseline decoder's cross-attention, every joint query attends over all 960 spatial tokens using the same effective attention temperature (implicitly set by `1/sqrt(head_dim) = 1/sqrt(32)` for 8 heads and hidden_dim=256). This uniform temperature is a structural mismatch with the anatomical heterogeneity of body joints:

| Joint type | Optimal attention behaviour | Reason |
|---|---|---|
| Wrist, ankle, toe | Sharp, focal (small region) | Small joints with precise spatial footprint; background tokens dilute signal |
| Shoulder, hip, knee | Moderate | Medium-sized joints with some limb context |
| Pelvis (query 0) | Very broad / diffuse | Pelvis depth/UV depends on global body scale and scene context; requires global aggregation |
| Spine, neck | Broad | Torso occupies large crop region; context from multiple body parts is informative |

With a single shared temperature, the softmax distribution is a compromise that is neither sharp enough for distal joints nor broad enough for proximal/pelvis joints. The model must learn to "work around" this temperature mismatch via its weight matrices — an indirect, capacity-consuming adaptation.

A learnable per-query temperature directly exposes this degree of freedom, allowing the model to:
- Set τ_i < 1.0 for distal joints → logits/τ_i are amplified → sharper, more peaked softmax → more focal spatial attention
- Set τ_i > 1.0 for proximal joints → logits/τ_i are damped → flatter softmax → more diffuse, global aggregation

### Grounding in Observed Results

Across all 19 prior ideas, `mpjpe_pelvis_val` stubbornly stays in the 608–720 mm range at stage-1 and 322–422 mm at stage-2 (baseline 653/366 mm). The best pelvis improvement comes from idea001/design001 (stage-2: 322 mm) and idea003/design002 (stage-2: 322 mm). Meanwhile, `mpjpe_body_val` has improved from baseline 195.7→156.6 mm (stage-2), a 20% gain. The gap between body and pelvis improvement suggests that:

1. The pelvis token (query 0) is being squeezed by the same cross-attention dynamics as body joint queries, when it actually needs a fundamentally different attention pattern (broad, global, depth-informative).
2. Distal joint queries (hands/feet — queries 22-69 and the later body queries like wrists/ankles) likely need sharper attention, but the shared temperature forces them to use the same distribution as the torso.

Per-query temperature directly addresses point (1): the pelvis token can learn τ_0 >> 1 (highly diffuse attention, aggregating global scene depth), while wrist/ankle tokens learn τ_i < 1 (sharp focal attention on limb endpoints). This is the first mechanism that allows the pelvis attention to differ in *sharpness* from body joint attention (all prior ideas modifying attention — ideas 006, 007, 009 — applied the same structural change uniformly across queries or applied group-level, not per-query).

### Differentiation from All 19 Prior Ideas

| Prior Idea | What it changes in attention | Key Difference |
|---|---|---|
| idea006 | Learnable **additive bias** to self-attention logits (query×query) | Different: (1) self-attention not cross-attention, (2) additive bias shifts which joints attend to each other, not temperature |
| idea007 | Learnable **multiplicative gating weights** on cross-attention **values** (post-softmax) | Different: idea007 gates the output values after softmax; this idea scales logits **before** softmax — the two operations have fundamentally different effects on the attention distribution shape |
| idea009 | **Random dropout** of spatial tokens (key_padding_mask) | Different: (1) unstructured and non-learnable, (2) removes tokens entirely, (3) not temperature-based |
| idea019 | **Deformable sparse sampling** — replaces cross-attention entirely | Different: changes which tokens are sampled, not the sharpness of attention over a fixed set |
| idea015 | **Super-token pooling** — compresses spatial tokens before cross-attention | Different: changes the number of tokens, not their attention sharpness |

This is the **first idea to modify the sharpness (temperature) of the cross-attention distribution**, and the first per-query mechanism applied to cross-attention logit scaling. All prior ideas that touched cross-attention (007, 009, 015, 018, 019) changed either the spatial token set, the attention values/outputs, or applied gate signals — none changed the softmax temperature.

### Why Per-Query and Not Per-Head or Per-Layer

- **Per-head**: would add 8 temperatures per query (560 total). Increasing parameter count without clear motivation; the attention pattern is query-driven, not head-specific at the granularity we target.
- **Per-layer**: only 70 temperatures for a 1-layer decoder. Would add little expressivity with a 2-layer decoder. Per-query temperature is already joint-specific — the core target.
- **Per-query (this idea)**: 70 scalars, one per joint. Anatomically meaningful: each joint has its own spatial attention focus requirement. Minimal parameter overhead.

### Mathematical Formulation

Standard cross-attention logits for joint query i, spatial token j:
```
a_{i,j} = (q_i @ k_j^T) / sqrt(d_head)           # baseline
```

With per-query temperature:
```
a_{i,j} = (q_i @ k_j^T) / (sqrt(d_head) * tau_i)  # this idea
         = (q_i @ k_j^T) / sqrt(d_head) / tau_i
```

The softmax is then applied over j: `softmax_j(a_{i,j})`.

When `tau_i > 1`: the logits are scaled down → softer/broader attention distribution.
When `tau_i < 1`: the logits are scaled up → sharper/more peaked attention distribution.
When `tau_i = 1`: identical to baseline — guaranteed at initialisation.

Implementation: one learnable `nn.Parameter(torch.ones(num_joints))`, applied in `_DecoderLayer.forward()` by dividing the cross-attention logits by the temperature vector before computing softmax. Since `nn.MultiheadAttention` does not expose raw logits before softmax, the temperature is applied via a **custom cross-attention forward** or via the `attn_mask` mechanism.

**Implementation detail — using `attn_mask`:** The simplest implementation uses `nn.MultiheadAttention`'s `attn_mask` argument. The attention computation is `softmax(QK^T/sqrt(d) + attn_mask)`. By setting `attn_mask[i, :] = log(1/tau_i)` for all spatial positions j, the effective temperature is implemented as:
```
softmax(a_{i,j} + log(1/tau_i)) = softmax(a_{i,j} - log(tau_i))
                                 ∝ softmax(a_{i,j}) * (1/tau_i)
```
Wait — this is not equivalent to temperature scaling. The correct approach for temperature scaling is to divide the entire logit vector for row i by tau_i. The additive form above would subtract a constant from all logits in row i, which has no effect on softmax (constant offset cancels). So we cannot use `attn_mask` directly for temperature scaling.

**Correct implementation — custom cross-attention with temperature:** Override the cross-attention forward to expose logits before softmax:

```python
# In _DecoderLayer.forward():
# Instead of: q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
# Use temperature-scaled cross-attention:
q2 = self._temp_cross_attn(q, spatial_tokens, spatial_tokens, self.cross_temp)
```

```python
def _temp_cross_attn(self, query, key, value, temperature):
    """Cross-attention with per-query temperature scaling.

    Args:
        query: (B, num_queries, D)
        key, value: (B, num_spatial, D)
        temperature: (num_queries,) — one scalar per query, shape matches query dim 1
    """
    # Use F.scaled_dot_product_attention (available in PyTorch >= 2.0)
    # or implement manually via projection + scaled_dot_product
    B, Nq, D = query.shape
    _, Ns, _ = key.shape
    Nh = self.cross_attn.num_heads
    dh = D // Nh

    # Project Q, K, V
    in_proj_weight = self.cross_attn.in_proj_weight   # (3D, D)
    in_proj_bias = self.cross_attn.in_proj_bias       # (3D,)
    out_proj = self.cross_attn.out_proj

    Q = torch.nn.functional.linear(query, in_proj_weight[:D], in_proj_bias[:D])    # (B, Nq, D)
    K = torch.nn.functional.linear(key,   in_proj_weight[D:2*D], in_proj_bias[D:2*D])  # (B, Ns, D)
    V = torch.nn.functional.linear(value, in_proj_weight[2*D:], in_proj_bias[2*D:])    # (B, Ns, D)

    # Reshape to multi-head
    Q = Q.view(B, Nq, Nh, dh).transpose(1, 2)  # (B, Nh, Nq, dh)
    K = K.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)
    V = V.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)

    # Attention logits
    scale = dh ** -0.5
    attn = (Q @ K.transpose(-2, -1)) * scale   # (B, Nh, Nq, Ns)

    # Apply per-query temperature: divide logits row i by tau_i
    # temperature: (Nq,) → reshape to (1, 1, Nq, 1) for broadcast
    tau = temperature.clamp(min=0.1).view(1, 1, Nq, 1)   # clamp prevents collapse
    attn = attn / tau                                       # (B, Nh, Nq, Ns)

    attn = attn.softmax(dim=-1)                            # (B, Nh, Nq, Ns)
    attn = torch.nn.functional.dropout(attn, p=self.cross_attn.dropout, training=self.training)

    # Aggregate values
    out = (attn @ V)                                       # (B, Nh, Nq, dh)
    out = out.transpose(1, 2).contiguous().view(B, Nq, D) # (B, Nq, D)
    out = out_proj(out)                                    # (B, Nq, D)
    return out
```

This is a ~30-line implementation in `pose3d_transformer_head.py`, using the existing `cross_attn` module's learned weight matrices and simply inserting the temperature scaling step after the dot-product but before softmax.

**AMP compatibility**: all operations (linear, matmul, softmax) are AMP-safe. The `tau.clamp(min=0.1)` prevents the temperature from collapsing to zero (which would cause overflow in logit scale), and `tau` initialised to 1.0 ensures identical behaviour at the start of training.

---

## Proposed Variations

### Design A — Single temperature per joint query, standard cross-attention temperature (minimal)

One learnable scalar `tau_i` per joint query (70 scalars total) applied to the cross-attention (not self-attention). Temperature initialised to 1.0. This is the minimal-change diagnostic: do learned per-query temperatures help at all?

New parameter: `self.cross_temp = nn.Parameter(torch.ones(num_joints))` in `Pose3dTransformerHead.__init__`.

Self-attention is unchanged (standard `nn.MultiheadAttention`). Only cross-attention uses temperature scaling.

Config kwarg: `use_cross_temp=True` (bool literal). Temperature is an internal `nn.Parameter`; no config kwarg needed for its value (always initialised to 1.0).

### Design B — Temperature + SoftPlus activation for guaranteed positivity + L2 regularization

Same as Design A, but instead of clamping tau, use `F.softplus(self.log_temp)` where `self.log_temp = nn.Parameter(torch.zeros(num_joints))` is the log-space parameterization. `softplus(0) = log(2) ≈ 0.693`, so the effective temperature at init is `softplus(0) ≈ 0.693` — slightly below 1.0, which means slightly sharper attention than baseline at initialisation. This is a more numerically stable parameterization that avoids the need for clamping.

Additionally, add a small L2 regularization term on the log-space temperatures to prevent extreme values:
```
loss_temp_reg = temp_reg_weight * log_temp.pow(2).mean()
```
with `temp_reg_weight=0.01`. This acts like a prior pulling temperatures back toward the baseline (log=0), preventing individual queries from learning degenerate temperatures (τ→0 or τ→∞).

Config kwargs: `use_cross_temp=True`, `temp_log_space=True`, `temp_reg_weight=0.01` (bool/float literals).

### Design C — Temperature scaling applied to both self-attention and cross-attention

Extend the temperature mechanism to self-attention as well: each joint query i has a separate learnable temperature for its self-attention (`self_temp_i`) and its cross-attention (`cross_temp_i`). Self-attention temperature controls how sharply each query attends to other queries — a pelvis query that needs global information might benefit from diffuse self-attention (learning from all other joint queries), while end-effector queries might sharpen their self-attention toward their kinematic ancestors.

New parameters:
- `self.self_temp = nn.Parameter(torch.ones(num_joints))` — 70 scalars
- `self.cross_temp = nn.Parameter(torch.ones(num_joints))` — 70 scalars

Total new parameters: 140 scalars (< 1 KB).

For self-attention: the custom temperature-scaled attention function is applied with `self.self_temp`. For cross-attention: same function with `self.cross_temp`.

This design tests whether allowing the self-attention temperature to differentiate joint specialisation provides additional benefit beyond cross-attention temperature alone.

Config kwargs: `use_cross_temp=True`, `use_self_temp=True` (bool literals).

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

**1. New helper function `_temp_scaled_attn`** (module-level function):

```python
def _temp_scaled_attn(
    mha_module: nn.MultiheadAttention,
    query: torch.Tensor,        # (B, Nq, D)
    key: torch.Tensor,          # (B, Ns, D)
    value: torch.Tensor,        # (B, Ns, D)
    temperature: torch.Tensor,  # (Nq,)
    dropout_p: float = 0.0,
    training: bool = True,
) -> torch.Tensor:
    """Standard MHA cross-attention with per-query temperature scaling of logits."""
    B, Nq, D = query.shape
    _, Ns, _ = key.shape
    Nh = mha_module.num_heads
    dh = D // Nh
    w = mha_module.in_proj_weight
    b = mha_module.in_proj_bias
    Q = torch.nn.functional.linear(query, w[:D],    b[:D] if b is not None else None)
    K = torch.nn.functional.linear(key,   w[D:2*D], b[D:2*D] if b is not None else None)
    V = torch.nn.functional.linear(value, w[2*D:],  b[2*D:] if b is not None else None)
    Q = Q.view(B, Nq, Nh, dh).transpose(1, 2)  # (B, Nh, Nq, dh)
    K = K.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)
    V = V.view(B, Ns, Nh, dh).transpose(1, 2)  # (B, Nh, Ns, dh)
    scale = dh ** -0.5
    attn = (Q @ K.transpose(-2, -1)) * scale   # (B, Nh, Nq, Ns)
    tau = temperature.clamp(min=0.1).view(1, 1, Nq, 1)
    attn = (attn / tau).softmax(dim=-1)
    if training and dropout_p > 0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, D)
    return mha_module.out_proj(out)
```

**2. `_DecoderLayer` changes:**

The constructor gains optional parameters: `cross_temp: nn.Parameter | None = None`, `self_temp: nn.Parameter | None = None`. These are passed in from the head and stored as attributes. In `forward()`:

```python
# Cross-attention (replace self.cross_attn call)
q = self.norm2(queries)
if self.cross_temp is not None:
    q2 = _temp_scaled_attn(
        self.cross_attn, q, spatial_tokens, spatial_tokens,
        self.cross_temp, dropout_p=self.cross_attn.dropout, training=self.training)
else:
    q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
queries = queries + self.dropout2(q2)
```

For Design C, same pattern applied to `self_attn` with `self.self_temp`.

**3. `Pose3dTransformerHead.__init__` changes:**

```python
# New kwargs (all with defaults for backward compat):
#   use_cross_temp: bool = False
#   use_self_temp: bool = False
#   temp_log_space: bool = False   (Design B)
#   temp_reg_weight: float = 0.0   (Design B)

if use_cross_temp:
    if temp_log_space:
        self.log_cross_temp = nn.Parameter(torch.zeros(num_joints))
    else:
        self.cross_temp = nn.Parameter(torch.ones(num_joints))

if use_self_temp:
    self.self_temp = nn.Parameter(torch.ones(num_joints))

# Pass temperatures to decoder layer:
cross_t = getattr(self, 'cross_temp', None) or (
    torch.nn.functional.softplus(self.log_cross_temp)
    if hasattr(self, 'log_cross_temp') else None)
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout,
                                   cross_temp=cross_t,
                                   self_temp=getattr(self, 'self_temp', None))
```

Note: passing `nn.Parameter` to `_DecoderLayer` as a reference (not a copy) ensures gradient flows correctly.

**4. `loss()` changes (Design B only):**

```python
if self.temp_reg_weight > 0 and hasattr(self, 'log_cross_temp'):
    losses['loss/temp_reg/train'] = (
        self.temp_reg_weight * self.log_cross_temp.pow(2).mean())
```

**5. `_init_head_weights()` changes:**

No additional init needed — `nn.Parameter(torch.ones(...))` and `nn.Parameter(torch.zeros(...))` are already correctly initialised (ones → tau=1.0 at start; zeros → softplus(0)=0.693 at start for Design B).

### `config.py`

Add to head kwargs:

**Design A:**
```python
use_cross_temp=True,
use_self_temp=False,
temp_log_space=False,
temp_reg_weight=0.0,
```

**Design B:**
```python
use_cross_temp=True,
use_self_temp=False,
temp_log_space=True,
temp_reg_weight=0.01,
```

**Design C:**
```python
use_cross_temp=True,
use_self_temp=True,
temp_log_space=False,
temp_reg_weight=0.0,
```

All values are bool/float literals. No Python import statements. Fully compliant with MMEngine config constraints.

---

## Expected Outcome

- **Primary gain — pelvis MPJPE**: the pelvis token (query 0) can learn a large τ_0, producing a flat, diffuse attention distribution over all 960 spatial tokens. This gives it a global spatial average — exactly what a depth/UV regression head needs (absolute position requires global scene context). Targeted at `mpjpe_pelvis_val < 610` at stage-1 (vs. baseline 652 mm), `< 330` at stage-2 (vs. baseline 366 mm).

- **Secondary gain — body MPJPE**: distal joint queries (wrists, ankles, toes) can learn τ_i < 1, producing sharper attention. This concentrates gradients on the most relevant spatial tokens for each joint, improving convergence for fine-grained joint localisation. Target: `mpjpe_body_val < 188` at stage-1 (vs. baseline 195.7 mm), `< 155` at stage-2.

- **Design A (cross-attention temperature only)**: diagnostic. Does per-joint temperature help? Expected composite_val < 338 at stage-1.

- **Design B (log-space temperature + regularization)**: more numerically stable parameterization. Regularization prevents extreme temperatures that could destabilize training. Expected composite_val < 332 at stage-1.

- **Design C (self + cross temperature)**: highest expressivity. Proximal joints can attend broadly in both self-attention (learn from all other queries) and cross-attention (aggregate global spatial features). Distal joints sharpen both. Expected composite_val < 328 at stage-1, improving toward best prior (328.14 — idea013/design003), with potential stage-2 < 225.

- **Composite target (stage-2)**: aim for `composite_val < 224` (competitive with best prior 224.52 — idea001/design001), driven primarily by pelvis improvement.

---

## Risk and Mitigation

- **Temperature collapse (τ→0)**: if a query's temperature collapses to near-zero, the logits become very large before softmax, causing `softmax` to produce a near-one-hot distribution. Under AMP float16, logit overflow is possible. Mitigations: (1) `tau.clamp(min=0.1)` in Design A/C prevents tau below 0.1, (2) `softplus` in Design B guarantees tau > 0 smoothly, (3) existing `clip_grad=dict(max_norm=1.0)` bounds temperature gradient magnitude.

- **Temperature explosion (τ→∞)**: if tau becomes very large, the effective logits approach zero before softmax, producing a near-uniform distribution (equivalent to removing cross-attention). This is a degenerate but stable state — the query simply ignores spatial features and relies on its own embedding. Mitigation: Design B's L2 regularization on log_temp pulls tau back toward 1.0; Designs A/C have no upper clamp, but gradient signal should prevent this since the attention loss increases.

- **AMP: float16 precision in temperature division**: dividing logits `(B, Nh, Nq, Ns)` in float16 by a float32 temperature scalar requires explicit casting. In the `_temp_scaled_attn` function: `tau = temperature.clamp(min=0.1).view(1, 1, Nq, 1).to(attn.dtype)`. The Designer must include this cast.

- **`in_proj_weight` access pattern**: `nn.MultiheadAttention.in_proj_weight` is a standard attribute when `_qkv_same_embed_dim=True` (which it is here, since all Q/K/V have the same dim). The Designer should add an assertion `assert self.cross_attn._qkv_same_embed_dim` in `__init__` to catch edge cases.

- **Interaction with idea006 (self-attention additive bias)**: idea006 adds a learnable additive bias to self-attention logits; Design C of this idea adds a multiplicative temperature scaling to self-attention. These two mechanisms are compatible — the bias shifts the logit landscape, the temperature controls its sharpness. A future idea could combine both, but this is not proposed here to keep the mechanism clean.

- **Interaction with idea008 (22-query body decoder)**: temperature scaling is fully compatible with `num_body_queries=22`. The `cross_temp` parameter shape would be `nn.Parameter(torch.ones(22))` instead of 70. This composition (22-query + temperature) is a strong combination that could be explored in a future idea, but is excluded here to isolate the temperature effect.

- **Gradient flow to temperatures**: the temperature parameter is in the denominator of the logit scaling. Its gradient is `d(attn_logits/tau)/d(tau) = -attn_logits/tau^2`, which is non-zero whenever attn_logits are non-zero. Gradient flow is therefore well-conditioned throughout training, as long as tau is not near zero (handled by clamp).

- **MMEngine config constraint**: `use_cross_temp`, `use_self_temp`, `temp_log_space` are bool literals; `temp_reg_weight` is a float literal. No Python import statements required. Fully compliant.

- **Eval/inference compatibility**: temperature parameters are `nn.Parameter` tensors that are part of the model state_dict. They are saved and loaded correctly by MMEngine's `CheckpointHook`. The `predict()` function calls `self.forward(feats)` which routes through the temperature-scaled cross-attention — no special inference mode needed.

- **Parameter count**: 70 scalars per temperature tensor × 1 (Design A/B) or 2 (Design C) = 70–140 additional floats ≈ 280–560 bytes. Negligible relative to the ~200M parameter model.

- **MMEngine config interaction**: the temperatures are internal `nn.Parameters` of the head, not config-visible hyperparameters (beyond the bool/float flags above). MMEngine does not need to be aware of their values — they are learned end-to-end.
