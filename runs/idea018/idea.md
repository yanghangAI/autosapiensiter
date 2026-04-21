**Idea Name:** Depth-Gated Cross-Attention for Depth-Plane-Consistent Joint Regression

**Approach:** Inject a soft depth-plane gate into the decoder's cross-attention by computing, for each spatial token, a scalar depth-consistency weight derived from the raw depth channel (bilinearly sampled at the token's grid location), then scaling cross-attention logits by these weights before softmax — so that each joint query preferentially attends to spatial tokens whose 3D depth is consistent with the human body's expected depth plane, suppressing background tokens that lie at structurally implausible depths.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder performs cross-attention between 70 joint queries and 960 spatial tokens with uniform key-value weighting:

```
attn_logits[b, h, i, j] = (q[b,h,i] · k[b,h,j]) / sqrt(head_dim)
attn_weights = softmax(attn_logits, dim=-1)
output[b,i] = sum_j attn_weights[b,h,i,j] * v[b,h,j]
```

There is no mechanism in the baseline to prefer spatial tokens that correspond to the human body's actual depth plane over background tokens at radically different depths. In an RGBD crop produced by `CropPersonRGBD`, the depth channel encodes per-pixel 3D distance from the camera. The subject (human body) is localised at a known approximate depth `pelvis_depth` (~2–8 m); background pixels at the floor, wall, or near-body objects can span 1–12 m. The backbone processes the full 4-channel RGBD input but the cross-attention has no explicit inductive bias to focus on the body-depth plane.

### The Depth Inconsistency Problem

Across all 17 completed ideas, three patterns consistently appear in the results:

1. **`mpjpe_abs` remains high** (533–833 mm across designs) — absolute pose requires accurate reconstruction of both body structure and pelvis depth. The cross-attention over depth-inconsistent background tokens introduces noise into the depth pathway.

2. **Body MPJPE is bounded ~156–185 mm** despite diverse architectural improvements. This persistent floor suggests that the backbone feature quality is not the bottleneck — the bottleneck is how joint queries aggregate spatial evidence. If cross-attention is diluted by tokens at the wrong depth plane, the body-relevant signal is suppressed.

3. **idea008/design002** achieved the best `mpjpe_rel_val` (333.2 mm vs. baseline 438.7 mm) and `mpjpe_abs` (533.8 mm) by reducing to 22 body-only queries — demonstrating that *restricting attention* to relevant queries helps. The complementary restriction on the **spatial token side** (by depth plane) has not been explored.

### What Depth-Gated Cross-Attention Proposes

Rather than compressing the token set (idea015: super-token pooling) or routing by joint group (idea007), depth-gated cross-attention adds a **per-token, input-conditional** scalar gate derived from the raw depth values in the RGBD sample:

```
depth_map:     (B, H', W')  — bilinearly downsampled from raw depth to feature grid
body_depth:    (B, 1)       — pelvis depth estimate (from pelvis_depth output head)
depth_err:     (B, H'*W')  = |depth_map - body_depth.expand| / σ_depth
gate:          (B, H'*W')  = exp(-depth_err²)  or  softplus(-depth_err * τ)
attn_logits_gated = attn_logits + log(gate).unsqueeze(1).unsqueeze(1)  # (B,1,1,H'*W') broadcast over heads and queries
attn_weights = softmax(attn_logits_gated, dim=-1)
```

The gate is a Gaussian-shaped soft mask centered at the estimated pelvis depth, with learnable scale parameter `σ_depth`. Spatial tokens closer to the pelvis depth plane receive gate values near 1 (their logits are unchanged), while tokens at distant depths receive gate values near 0 (logits are strongly negative-shifted), effectively suppressing them in the softmax.

This is an **additive logit-space gate** (adding `log(gate)` to attention logits is equivalent to multiplicative masking of attention weights before softmax normalization), which is numerically stable, differentiable, and backed by the masked attention pattern used in BERT/GPT attention for causal masking.

### Why `body_depth` Is Available in Forward

A key implementation insight: the pelvis depth is predicted from token 0 of the decoded queries — but decoding is preceded by cross-attention. This creates a chicken-and-egg issue: we want the depth gate to depend on the current predicted depth, but the depth is predicted after the gate is applied.

**Solution**: use a lightweight **depth probe** — a single linear layer applied to the global average pool of the projected spatial tokens — to predict a preliminary depth estimate `z_hat (B, 1)` *before* the decoder. This estimate does not need to be accurate; it just needs to capture the coarse depth scale (it is trained with a small auxiliary loss on pelvis depth or simply inherits the depth signal from the pooled spatial features). The gate is then computed from `z_hat`. The full pelvis depth output from token 0 remains unchanged (no architectural change to the pelvis path).

Alternatively, a fixed gate derived from the **raw depth map** (not from a predicted depth) avoids the chicken-and-egg problem entirely: we can compute the gate directly from the bilinearly downsampled depth channel values in the RGBD input (which are available in the backbone feature map's depth slice via `feats[-1]` from the depth channel of the input, or from `data_sample.metainfo`). In the baseline, `feats[-1]` is the backbone output of the 4-channel input; the raw depth map is passed through `CropPersonRGBD` and available in `data_sample.inputs` (the depth_map tensor). However, the head's `forward()` only receives `feats` (backbone outputs), not raw inputs.

**Pragmatic solution used in this idea**: derive the depth gate from the backbone's own spatial features using a one-layer depth probe (shared for all designs), which avoids touching `data_sample` in `forward()` and keeps the implementation confined to `pose3d_transformer_head.py`.

### Differentiation from All Prior Ideas

| Idea | Spatial Token Treatment | Key Difference |
|---|---|---|
| idea004 | Depth added to positional encoding | Positional, not attention-weight-level |
| idea007 | Soft spatial routing by joint group | Joint-group masks, not depth-plane gate; not content-adaptive |
| idea009 | Random spatial token dropout | Unstructured; no depth content; hurts performance |
| idea015 | Slot attention compresses 960 → K tokens | Learned aggregation; not depth-plane gated |
| idea016 | FiLM modulation of all tokens by global pool | Channel-wise feature rescaling; gate applies uniformly to all tokens |
| **idea018** | **Per-token depth gate on cross-attention logits** | **Content-adaptive, depth-plane-specific; per-token, per-sample** |

Idea018 is the **first mechanism that directly uses the depth-plane content of individual spatial tokens to modulate cross-attention**. All prior spatial-token ideas (004, 007, 009, 015, 016) treat spatial tokens either uniformly, by learned aggregation, or by spatial position — none by the depth value at each token's location.

### Grounding in Observed Results

- **idea004** (depth positional encoding, stage-1 best design001: 336.57) improved composite by adding depth to position signals. This validates that depth information helps, but positional encoding is a weak proxy because it modulates position rather than attention weights.
- **idea007** (joint-group spatial routing, stage-1 design002: 339.72) showed that routing attention spatially improves composite. Depth-gated routing is complementary: instead of routing by joint anatomy group, route by depth consistency.
- **idea009** (spatial token dropout, all designs 349–375) — random removal hurts. This negative result is the strongest evidence for the importance of *structured* token selection vs. random masking. Depth-gated attention is structured selection based on physical depth consistency.
- **`mpjpe_abs` floor**: the best absolute MPJPE at stage-2 is 533mm (idea008/design002). Absolute pose accuracy is directly coupled to the depth inference quality. Suppressing depth-inconsistent background tokens should reduce the depth estimation noise that contributes to this floor.

---

## Proposed Variations

### Design A — Fixed-sigma Gaussian depth gate from spatial depth probe (minimal)

A single learnable depth probe (`LinearProbe: Linear(hidden_dim, 1)` applied to the global average pool of projected spatial tokens) estimates a preliminary pelvis depth `z_hat (B, 1)`. Each spatial token's depth is estimated by a single-channel linear probe `depth_channel_proj: Linear(hidden_dim, 1)` that projects each spatial token to a scalar depth estimate (initialised to predict zero = no depth bias). The gate is:

```
z_hat = probe_global(spatial.mean(dim=1))    # (B, 1) — preliminary body depth
z_tok = probe_token(spatial)                 # (B, H'*W', 1) — per-token depth estimate
depth_err = (z_tok.squeeze(-1) - z_hat) / sigma   # (B, H'*W'), sigma=1.0 fixed
gate_logit = -0.5 * depth_err ** 2          # (B, H'*W')  — Gaussian log-gate
# Add gate_logit to cross-attention logits (broadcast over heads/queries):
attn_logits_gated = attn_logits + gate_logit.unsqueeze(1).unsqueeze(1)
```

`probe_global` and `probe_token` are both initialised with near-zero weights (output bias at zero), so at initialisation the gate is flat (uniform weight over all tokens), and the model degrades to the baseline. This ensures training starts at the baseline and the gate is learned incrementally.

`sigma = 1.0` is a fixed hyperparameter (not learned in Design A). This controls the soft-masking bandwidth around the estimated body depth plane.

The gated cross-attention requires a custom forward for `_DecoderLayer` that accepts an optional `attn_logit_bias` argument. Since the standard `nn.MultiheadAttention` accepts `attn_mask` as an additive bias (when `attn_mask.dtype` is float), this is fully supported via the existing PyTorch API: `self.cross_attn(q, k, v, attn_mask=gate_logit_broadcast)`.

Parameter count: `probe_global: Linear(hidden_dim, 1)` = 257 params; `probe_token: Linear(hidden_dim, 1)` = 257 params. Total: 514 parameters. Negligible.

Config kwargs: `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0`, `depth_gate_init_zero=True`.

### Design B — Learnable-sigma depth gate with auxiliary depth probe loss

Same as Design A but `sigma` is a learnable `nn.Parameter(torch.ones(1))` (initialised to 1.0 and positive-clamped during forward: `sigma = sigma.abs() + 0.01`). This allows the model to learn the optimal bandwidth: a small sigma means strict depth-plane selection; large sigma approaches the baseline (uniform weighting).

Additionally, add a small auxiliary loss on `z_hat` (the preliminary depth from `probe_global`) to ensure the probe converges to a useful depth estimate:
```
L_probe = lambda_probe * smooth_l1(z_hat, gt_depth)
```
with `lambda_probe = 0.1` — small enough not to distort the main loss but sufficient to give the depth probe a training signal. This auxiliary loss is purely training-time and does not affect inference.

The learnable sigma provides an automatic adaptation: if depth-gating helps, sigma learns to be small (tight depth band); if it hurts in some scenario, sigma learns to be large (relaxed gate). This is an uncertainty-adaptive version of the depth gate, analogous to idea005's uncertainty-weighted loss but applied to the spatial attention gate.

Config kwargs: `depth_gate_type='gaussian_learnable_sigma'`, `depth_probe_loss_weight=0.1`.

### Design C — Depth gate combined with body-only 22-query decoder (compositional)

Apply Design A's depth gate in the setting of idea008's body-only 22-query decoder. Run 22 body-only queries through a single decoder layer with depth-gated cross-attention. Hand joints are recovered via a linear projection from body query features (same as idea008/design002). This design tests whether depth-gated attention and query-side body isolation compound their respective benefits:

- **idea008/design002 (alone)**: composite 333.63 stage-1, 241.14 stage-2.
- **idea018/design003 (combined)**: target composite < 320 stage-1, < 230 stage-2.

Architectural interaction: with 22 body queries, the cross-attention is `(B, 22, 256)` × `(B, 960, 256)`. The depth gate `gate_logit (B, 960)` is broadcast over the 22 query dimension — same gate for all 22 queries (a shared depth-plane assumption for the full body, which is appropriate since a standing person occupies a narrow depth range). The gate adds no query-specific logic, making the combination clean.

Implementation: inherit from the combined body-focused decoder (idea008/design002) and add the `attn_mask=gate_logit_broadcast` argument to `cross_attn` in `_DecoderLayer`.

Config kwargs: `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0`, `depth_gate_init_zero=True`.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or `train.py` wrapper.

### `pose3d_transformer_head.py`

**`_DecoderLayer` changes:**

Modify `_DecoderLayer.forward()` to accept an optional `attn_logit_bias` argument (a pre-computed additive bias to cross-attention logits):

```python
def forward(self, queries, spatial_tokens, attn_logit_bias=None):
    # Self-attention (unchanged)
    q = self.norm1(queries)
    q2 = self.self_attn(q, q, q)[0]
    queries = queries + self.dropout1(q2)

    # Cross-attention with optional logit bias
    q = self.norm2(queries)
    if attn_logit_bias is not None:
        # attn_logit_bias: (B, num_spatial) → broadcast to (B*num_heads, num_queries, num_spatial)
        # PyTorch MHA accepts float attn_mask as additive bias
        B_nq, Nq, D = q.shape
        num_heads = self.cross_attn.num_heads
        # Expand: (B, 1, 1, N_spatial) → MHA expects (B*num_heads, Nq, N_spatial) or (Nq, N_spatial)
        # Use per-sample bias via unsqueeze; MHA attn_mask shape: (B*num_heads, Nq, N_spatial)
        B = attn_logit_bias.shape[0]
        N_spatial = attn_logit_bias.shape[1]
        # Broadcast: (B, Nq, N_spatial)
        mask = attn_logit_bias.unsqueeze(1).expand(B, Nq, N_spatial)
        # Reshape to (B*num_heads, Nq, N_spatial) by repeating heads
        mask = mask.unsqueeze(1).expand(B, num_heads, Nq, N_spatial).reshape(B * num_heads, Nq, N_spatial)
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens, attn_mask=mask)[0]
    else:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
    queries = queries + self.dropout2(q2)

    # FFN (unchanged)
    queries = queries + self.ffn(self.norm3(queries))
    return queries
```

**`Pose3dTransformerHead.__init__` changes:**

```python
# New constructor kwargs:
#   depth_gate_type: str = 'none'        ('none' | 'gaussian' | 'gaussian_learnable_sigma')
#   depth_gate_sigma: float = 1.0        (fixed sigma for 'gaussian')
#   depth_probe_loss_weight: float = 0.0 (auxiliary probe loss weight for Design B)
#   depth_gate_init_zero: bool = True    (near-zero init for probe networks)

self.depth_gate_type = depth_gate_type
if depth_gate_type != 'none':
    # Global depth probe: pool spatial tokens → scalar depth estimate
    self.depth_probe_global = nn.Linear(hidden_dim, 1)
    # Per-token depth probe: spatial token → scalar depth estimate
    self.depth_probe_token = nn.Linear(hidden_dim, 1)
    # Fixed or learnable sigma
    if depth_gate_type == 'gaussian_learnable_sigma':
        self.log_sigma = nn.Parameter(torch.zeros(1))  # log(sigma), init → sigma=1.0
    else:
        self.register_buffer('depth_gate_sigma',
                             torch.tensor(depth_gate_sigma, dtype=torch.float32))
```

**`forward()` changes:**

After computing `spatial = input_proj(feat) + pos_enc`, and before `decoder_layer(queries, spatial)`:

```python
attn_logit_bias = None
if self.depth_gate_type != 'none':
    z_hat = self.depth_probe_global(spatial.mean(dim=1))  # (B, 1)
    z_tok = self.depth_probe_token(spatial).squeeze(-1)    # (B, H'*W')
    if self.depth_gate_type == 'gaussian_learnable_sigma':
        sigma = torch.exp(self.log_sigma).clamp(min=0.01)
    else:
        sigma = self.depth_gate_sigma
    depth_err = (z_tok - z_hat) / sigma                    # (B, H'*W')
    attn_logit_bias = -0.5 * depth_err ** 2               # (B, H'*W') — log Gaussian gate
    # Store z_hat for auxiliary loss in loss()
    self._depth_probe_z_hat = z_hat

decoded = self.decoder_layer(queries, spatial, attn_logit_bias=attn_logit_bias)
```

**`loss()` changes:**

For Design B (`depth_probe_loss_weight > 0`):

```python
if self.depth_probe_loss_weight > 0.0 and hasattr(self, '_depth_probe_z_hat'):
    losses['loss/depth_probe/train'] = self.depth_probe_loss_weight * self.loss_depth_module(
        self._depth_probe_z_hat, gt_depth)
```

**`_init_head_weights()` changes:**

```python
if self.depth_gate_type != 'none':
    # Near-zero init: gate is flat at start → behaves like baseline
    nn.init.zeros_(self.depth_probe_global.weight)
    nn.init.zeros_(self.depth_probe_global.bias)
    nn.init.zeros_(self.depth_probe_token.weight)
    nn.init.zeros_(self.depth_probe_token.bias)
```

With zero-weight init, `z_hat = 0` and `z_tok = 0` at step 0, so `depth_err = 0` and `gate_logit = 0` for all tokens. This recovers exactly the baseline cross-attention (zero additive logit bias). The gate emerges as the probe networks learn.

### `config.py`

Add to head kwargs (all literals, no imports):

```python
head=dict(
    type='Pose3dTransformerHead',
    # ... existing args ...
    depth_gate_type='gaussian',         # Design A/C: 'gaussian'; Design B: 'gaussian_learnable_sigma'
    depth_gate_sigma=1.0,               # Design A/C only (float literal)
    depth_probe_loss_weight=0.0,        # Design A/C: 0.0; Design B: 0.1
    # Design C additionally:
    num_body_queries=22,
    hand_aux_loss_weight=0.1,
)
```

All values are str/float/int literals. No Python import statements. Fully compliant with MMEngine config constraints.

---

## Expected Outcome

- **Primary gain — `mpjpe_abs` and `mpjpe_rel_val`**: by suppressing cross-attention to spatial tokens at implausible depths (background walls, floor, near-body objects), each joint query concentrates on body-relevant spatial tokens. This directly improves the reconstruction of body geometry relative to the camera — impacting both relative (root-removed) and absolute pose accuracy.
  - Target: `mpjpe_rel_val < 380` at stage-2 (best prior: 333mm — idea008/design002; typical: 420–440mm)
  - Target: `mpjpe_abs < 480` at stage-2 (best prior: 533mm — idea008/design002)
- **Secondary gain — body MPJPE**: with background tokens suppressed, joint queries aggregate cleaner body-region features. Expected moderate improvement: `mpjpe_body_val < 175` at stage-2 (best prior: 156mm — idea002/design003; most designs: 180–195mm).
- **Pelvis MPJPE**: the preliminary depth probe `z_hat` provides an explicit depth signal before the decoder, which may improve the final depth regression (since the full decoder now operates on depth-consistent tokens). Target: `mpjpe_pelvis_val < 300` at stage-2 (best prior: 322mm — idea001/design001).
- **Composite target (stage-1)**: aim for `composite_val < 330` (best prior: 328.14 — idea013/design003).
- **Composite target (stage-2)**: aim for `composite_val < 215` (best prior: 224.52 — idea001/design001).
- **Design A**: diagnostic — does depth-gated attention help vs. flat gate? The clean test of the core hypothesis.
- **Design B**: learnable sigma + auxiliary depth probe supervision. Should learn the optimal bandwidth and provide a stronger depth signal. Expected slightly better than Design A after the probe converges.
- **Design C**: combines depth gating with body-only 22-query decoder. Expected best result if depth gating and query isolation compound their gains independently.

---

## Risk and Mitigation

- **Zero-init gate gives flat start**: the near-zero probe initialization ensures `gate_logit = 0` at step 0, recovering the baseline. As training proceeds, the probe networks learn to distinguish body vs. background tokens by depth. If the probes fail to learn anything (gradient vanishing), the gate stays flat and the design is equivalent to the baseline — a safe fallback. Monitoring `log_sigma.item()` (Design B) and the mean absolute gate logit during training provides a diagnostic.

- **Depth probe conflict with body depth output**: the preliminary depth probe `z_hat` and the final pelvis depth `pelvis_depth` (from token 0 of the decoded output) are separate and non-conflicting. The auxiliary loss on `z_hat` (Design B) does not interfere with the main pelvis depth loss because they use independent output pathways. There is no gradient cycle: `z_hat` is computed *before* the decoder, and `pelvis_depth` is computed *after* the decoder. Their gradients flow through separate parameters.

- **AMP numerical stability**: `gate_logit = -0.5 * depth_err^2` is bounded above by 0 (negative or zero). In float16, negative log-space gates are safe. The `attn_mask` argument in PyTorch's `nn.MultiheadAttention` is applied in the same dtype as the query/key tensors — AMP auto-casts to float16, which is numerically stable for bounded negative logits. If float16 underflows (very large `depth_err`), the gate becomes very negative → softmax weight approaches 0 → this is the correct desired behavior (far-depth tokens are fully suppressed).

- **Chicken-and-egg for depth probe convergence**: the probe networks start at zero and must learn to distinguish body depth from background depth using only the projected spatial token features. The spatial features come from a backbone trained on RGBD input, so they do contain depth-relevant activations. However, the probe might be slow to converge in the first few epochs. Design B's auxiliary loss (`lambda_probe = 0.1 * smooth_l1(z_hat, gt_depth)`) directly addresses this by providing a direct supervision signal for the probe from epoch 1.

- **`nn.MultiheadAttention` attn_mask broadcasting**: PyTorch's `nn.MultiheadAttention` with `batch_first=True` accepts `attn_mask` in shape `(tgt_len, src_len)` or `(B*num_heads, tgt_len, src_len)`. We use the latter to apply per-sample depth gates. The Designer should verify the exact shape expansion in the `_DecoderLayer` modification, especially under AMP. A safe alternative is to detach the mask from the computation graph for the backward pass through self-attention (the gate only needs to modulate forward attention, not receive gradient from the self-attention path).

- **Gate applies to all joint queries uniformly**: the gate is `(B, N_spatial)` broadcast over the query dimension, meaning all joint queries share the same depth gate. This is appropriate under the assumption that all body joints lie in roughly the same depth plane (valid for BEDLAM2's in-frame full-body crops). For edge cases where limbs extend significantly toward the camera (stretched arms, kicked legs), the uniform gate may suppress some valid tokens. Design B's learnable sigma handles this by widening the gate bandwidth when the depth spread is high.

- **Design C memory**: combining 22-body-queries + depth gate adds `probe_global` (257 params) and `probe_token` (257 params) to the 22-query design. The depth gate computation adds `B × 960 × 1` = 3840 FLOPs per batch element — negligible. Net memory is lower than the baseline (22 × 960 cross-attention vs. 70 × 960, with 514 extra params). No OOM risk.

- **Interaction with idea015 (super-token pooling)**: the depth gate could be applied either before or after super-token pooling. Before pooling: depth-gated raw tokens are compressed into super-tokens. After pooling: super-tokens are gated by their mean depth. The combination is valid but left for a future idea — idea018 applies the gate to the raw 960 tokens as they appear in the baseline.

- **Interaction with idea017 (body-focused multi-layer decoder)**: the depth gate can be applied to each decoder layer in a multi-layer stack. Design C provides a minimal composition (22 queries + gate). A full composition with 2-layer decoder + gate is promising but increases implementation complexity — left to Designer if Design C shows strong results.

- **MMEngine config constraint**: `depth_gate_type` is a str literal; `depth_gate_sigma` and `depth_probe_loss_weight` are float literals. No Python import statements required. Fully compliant with MMEngine no-Python-imports restriction.

- **Eval/inference compatibility**: `predict()` calls `self.forward(feats)`, which applies the depth gate inside. Output tensor shapes `(B, 70, 3)`, `(B, 1)`, `(B, 2)` are unchanged. `BedlamMPJPEMetric`, `TrainMPJPEAveragingHook`, and `MetricsCSVHook` see identical interfaces. No downstream changes.

- **`_depth_probe_z_hat` attribute caching in `loss()`**: `loss()` calls `forward()` which sets `self._depth_probe_z_hat`. The `loss()` function then reads this attribute. This pattern is already used in the baseline for `self._train_mpjpe`. No thread-safety issues in single-GPU training.
