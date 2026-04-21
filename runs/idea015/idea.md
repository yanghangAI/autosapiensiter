**Idea Name:** Spatial Super-Token Aggregation via Learned Pooling

**Approach:** Replace the flat cross-attention over all H'×W'=960 spatial tokens with a two-stage mechanism: first, compress the 960 spatial tokens into K learned "super-tokens" via differentiable attention pooling (a small set of K slot-query vectors cross-attends over the full 960 tokens); then, run the joint queries' cross-attention against only these K super-tokens — reducing the cross-attention key-value set by 10–30× and enabling additional decoder capacity within the same memory budget.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder performs cross-attention with the following dimensions:

```
Q: (B, 70, 256)   — joint queries
K, V: (B, 960, 256)  — spatial tokens (H'=24, W'=40, so 24×40=960)
```

Cross-attention complexity is O(70 × 960) = O(67,200) products per head per layer. The 960 spatial tokens are a flat, unstructured set — every query attends to all 960 tokens with no spatial grouping or multi-scale structure. This has three concrete consequences:

### 1. Wasted attention bandwidth on uninformative tokens

In a 640×384 RGBD crop, the 24×40 feature grid covers the entire image at 1/16 resolution. The human body typically occupies ≈30–60% of the crop after `CropPersonRGBD`. This means 40–70% of the 960 spatial tokens correspond to background regions (floor, objects, sky) with no discriminative information for joint regression. Each joint query allocates uniform key-value bandwidth to these background tokens, wasting attention capacity.

### 2. Shallow spatial feature hierarchy

The baseline uses a single-resolution feature map `feats[-1]` at H'=24, W'=40. There is no aggregation of global context (coarse scale) that summarises the full body, nor fine-grain focus on small joints. In ViT-based pipelines without a feature pyramid, a learned pooling step can create a coarse-to-fine hierarchy at no additional backbone cost — purely in the head.

### 3. Memory barrier to stacking more decoder layers

The top-performing ideas (idea001, idea013) showed that increasing decoder depth helps (idea001 with 2–4 layers; idea013 with kinematic chain). But each added decoder layer recomputes cross-attention over all 960 tokens. With batch=4, hidden_dim=256, and num_heads=8, the cross-attention for a single layer uses approximately `4 × 8 × 70 × 960 × 32B ≈ 67 MB` of activation memory (rough estimate with AMP). On the 2080 Ti (10.57 GB), this limits stacking. Reducing the key-value set to K=32 super-tokens brings this to `4 × 8 × 70 × 32 × 32B ≈ 2.2 MB` — a 30× reduction — enabling 2–4 additional decoder layers within the same memory budget.

### The Super-Token Pooling Mechanism

The pooling uses K learnable slot vectors `S ∈ R^{K×hidden_dim}` that cross-attend over the 960 spatial tokens:

```
slots:  S (1, K, hidden_dim) → expanded to (B, K, hidden_dim)
super_tokens = cross_attn(S, spatial_tokens, spatial_tokens)  → (B, K, hidden_dim)
```

where `cross_attn` is a standard `nn.MultiheadAttention` with `batch_first=True`. The softmax over 960 keys is computed for each of the K slot queries, learning to pool semantically similar tokens into each slot. The output K super-tokens are then used as keys/values for the joint query cross-attention:

```
decoded = decoder_layer(joint_queries, super_tokens)  → (B, 70, hidden_dim)
```

This is a strict superset of direct cross-attention: if K=960 and the slots are initialised to identity, it recovers the baseline exactly. In practice, K << 960, so the slots must compress the spatial information.

### Why This Is Different from All Prior Ideas

| Prior Idea | Mechanism | Key Difference |
|---|---|---|
| idea001 | Stack multiple decoder layers | More layers over full 960 tokens; no token compression |
| idea002 | Dedicated pelvis query | Changes queries, not spatial tokens |
| idea003 | Content-adaptive query init | Changes query initialisation, tokens unchanged |
| idea004 | Depth-aware positional encoding | Adds depth signal to spatial tokens, still 960 tokens |
| idea005 | Uncertainty loss weighting | Loss-level change; no structural change |
| idea006 | Skeleton self-attention bias | Query self-attention modification; tokens unchanged |
| idea007 | Joint-group spatial routing | Soft gating of channels per joint group; still 960 tokens |
| idea008 | Body-focused decoder (22 queries) | Reduces queries; still 960 spatial tokens |
| idea009 | Spatial token dropout | Randomly drops tokens; no learned structure |
| idea010 | 2D reprojection loss | Loss modification; tokens unchanged |
| idea011 | Iterative coordinate refinement | Two decoder passes; still 960 tokens in both |
| idea012 | Pairwise distance-matrix loss | Loss modification; tokens unchanged |
| idea013 | Kinematic bone-vector output | Output parameterization; tokens unchanged |
| idea014 | Anchor-based pelvis depth | Pelvis depth discretization; tokens unchanged |

**No prior idea has compressed the spatial token set.** idea009 randomly drops tokens (unstructured) — this idea applies **learned, content-driven aggregation** that preserves the information from all 960 tokens while reducing the effective key-value set that cross-attention must search.

### Grounding in Observed Results

- **idea001 (multi-layer decoder)** achieved the best stage-2 composite (224.52) by stacking decoder layers. The memory savings from super-token pooling would allow 3–4 layers instead of 2, potentially pushing idea001's gain further.
- **idea009 (spatial token dropout)** — the negative result that randomly removing tokens hurts (design001: 375.46, design002: 349.08, design003: 360.65) — directly motivates *learned* pooling rather than random removal. The difference is that super-token pooling aggregates all tokens rather than discarding any.
- **idea008 (body-focused 22-query decoder)** achieved strong relative MPJPE improvement (333.2mm vs. baseline 438.6mm) by focusing the query side. The spatial-side complement — focusing the token side — has not been explored.
- **mpjpe_rel_val** stagnates at 420–440mm across most ideas. This metric reflects how well the model captures body shape and limb structure (root-removed). Reducing cross-attention noise from background tokens via super-token pooling targets this directly.

### Prior Art

Spatial token compression via slot attention / learned pooling is well-established:
- **Perceiver IO** (Jaegle et al., ICML 2021): latent-space cross-attention from a small set of learned latent vectors over a large input, demonstrating that K << N learned latents can effectively compress visual features.
- **Set Transformers** (Lee et al., ICML 2019): Induced Set Attention Blocks use m inducing points to compress N tokens to m-dimensional representation.
- **TokenLearner** (Ryoo et al., NeurIPS 2021): learns to reduce video/image token sets from 196 to 8 tokens with minimal information loss.
- **SMCA** (Gao et al., ICCV 2021): constrained cross-attention for DETR that focuses attention spatially; this idea applies the complementary approach of compressing the key-value set.

For 3D pose estimation on BEDLAM2, the key motivation is that RGBD crops contain strong structural redundancy in the background, and learned pooling can extract the body-relevant spatial information into a compact, noise-reduced super-token representation.

---

## Proposed Variations

### Design A — K=32 super-tokens, single slot-attention layer (minimal)

Compress 960 spatial tokens to K=32 super-tokens via a single `nn.MultiheadAttention` slot layer (num_heads=8, embed_dim=256). The decoder's cross-attention then operates on these 32 super-tokens instead of 960.

Architecture change from baseline:
```
spatial = input_proj(feat) + pos_enc          # (B, 960, 256) — unchanged
super_tokens = slot_attn(S, spatial, spatial) # (B, 32, 256) — new
decoded = decoder_layer(queries, super_tokens) # (B, 70, 256) — unchanged interface
```

`S = self.slot_queries.weight` is a learnable `nn.Embedding(32, 256)`.
`slot_attn` is `nn.MultiheadAttention(256, num_heads=8, batch_first=True)`.

Advantages:
- 30× reduction in cross-attention key-value size → decoder cross-attention is 30× cheaper.
- Can increase number of decoder layers from 1 to 2 within same memory budget.
- Minimal new parameters: K×hidden_dim = 32×256 = 8,192 scalars for slot embeddings, plus the slot attention module (~1.3M params).

Config changes: `num_super_tokens=32` as head kwarg (int literal).

### Design B — K=64 super-tokens with positional super-token initialization (spatially grounded)

Same mechanism as Design A with K=64 super-tokens. Additionally, initialise the K slot embeddings not from random noise but from the K cluster centres of the 2D positional encoding grid. Specifically, divide the 24×40 feature grid into K=64 spatial blocks (4×16 or 8×8 layout), and initialise each slot embedding with the mean 2D sinusoidal positional encoding of its block. This spatially grounds the slots: at initialisation, slot i tends to aggregate tokens from spatial region i, providing a warm start analogous to idea003's content-adaptive query initialisation — but applied to the spatial token compression step.

This avoids the cold-start problem where all K slots initially attend to the same (most salient) spatial region, which can cause training instability in the first few epochs when the backbone features are still adapting.

Positional initialisation is done once at `__init__` time using `_build_2d_sincos_pos_enc` (already available in the module), averaged over each block:
```python
pos_enc = _build_2d_sincos_pos_enc(24, 40, 256)  # (1, 960, 256)
pos_2d = pos_enc.reshape(8, 120, 256).mean(1)     # (8, 256) — not quite; actual block split by Designer
```
The exact averaging logic is left to the Designer (simple numpy operation at init, then assigned as initial weight of `self.slot_queries`).

Config changes: `num_super_tokens=64`, `slot_pos_init=True` as head kwargs.

### Design C — K=32 super-tokens + 2 decoder layers (stacked with super-token pooling)

Use K=32 super-tokens (same as Design A), but exploit the memory savings to stack **2 decoder layers** (as in the best designs of idea001). The decoder runs:
```
super_tokens = slot_attn(S, spatial, spatial)  # (B, 32, 256) — pooling
decoded = decoder_layer_1(queries, super_tokens)  # layer 1
decoded = decoder_layer_2(decoded, super_tokens)  # layer 2 — reuses same super_tokens
```

The super-tokens are computed once and reused by both decoder layers, so the slot-attention cost is paid once. Two decoder layers × smaller cross-attention (32 tokens) costs approximately the same memory as one decoder layer × large cross-attention (960 tokens):
- Baseline: 1 layer × 960 K/V → cost ∝ 960
- Design C: 2 layers × 32 K/V → cost ∝ 64 (33% of baseline)

This is the primary test of whether super-token pooling enables more decoder depth within budget. Intermediate supervision on layer-1 output (aux joint loss weight 0.4, same as idea001/design002) prevents gradient vanishing at the first decoder layer.

Config changes: `num_super_tokens=32`, `num_decoder_layers=2`, `aux_loss_weight=0.4` as head kwargs (int/float literals).

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`.

### `pose3d_transformer_head.py`

**`__init__` changes:**
```python
# New constructor kwargs:
#   num_super_tokens: int = 0  (0 = disabled, use flat 960 tokens as baseline)
#   slot_pos_init: bool = False
#   num_decoder_layers: int = 1
#   aux_loss_weight: float = 0.0

# New modules (when num_super_tokens > 0):
self.slot_queries = nn.Embedding(num_super_tokens, hidden_dim)
self.slot_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
self.slot_norm = nn.LayerNorm(hidden_dim)  # pre-norm for slot attention

# Decoder stack (when num_decoder_layers > 1):
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
# Note: existing self.decoder_layer kept for backward compat when num_decoder_layers=1
```

**`forward()` changes:**
1. After computing `spatial` tokens (input projection + positional encoding), apply slot attention if `num_super_tokens > 0`:
   ```python
   if self.num_super_tokens > 0:
       S = self.slot_queries.weight.unsqueeze(0).expand(B, -1, -1)
       S_norm = self.slot_norm(S)
       super_tokens, _ = self.slot_attn(S_norm, spatial, spatial)
       spatial_for_decoder = super_tokens  # (B, K, hidden_dim)
   else:
       spatial_for_decoder = spatial  # (B, 960, hidden_dim) — baseline
   ```

2. Run decoder layer(s):
   ```python
   decoded = queries
   intermediate_outputs = []
   for layer in self.decoder_layers:
       decoded = layer(decoded, spatial_for_decoder)
       intermediate_outputs.append(decoded)
   ```

3. Read output projections from final `decoded` (unchanged).

**`loss()` changes (Design C only):**
- If `aux_loss_weight > 0`, compute joint loss on intermediate decoder outputs (same pattern as idea001/design002):
  ```python
  for i, inter_decoded in enumerate(intermediate_outputs[:-1]):
      inter_joints = self.joints_out(inter_decoded)
      losses[f'loss/joints_aux_{i}/train'] = self.aux_loss_weight * self.loss_joints_module(
          inter_joints[:, _BODY], gt_joints[:, _BODY])
  ```

**`_init_head_weights()` changes:**
- `nn.init.trunc_normal_(self.slot_queries.weight, std=0.02)` (standard init).
- For `slot_pos_init=True` (Design B): after the standard init, overwrite `slot_queries.weight` with the position-averaged sinusoidal encodings (computed from `_build_2d_sincos_pos_enc(24, 40, hidden_dim)` and spatially blocked).

### `config.py`
- Add `num_super_tokens=32` or `num_super_tokens=64` to head kwargs (int literal).
- Add `slot_pos_init=False` or `slot_pos_init=True` (bool literal).
- Add `num_decoder_layers=1` or `num_decoder_layers=2` (int literal).
- Add `aux_loss_weight=0.0` or `aux_loss_weight=0.4` (float literal).

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

---

## Expected Outcome

- **Primary gain — mpjpe_rel_val**: by filtering background spatial tokens into compact super-tokens, the cross-attention focuses on body-region information. Root-relative joint accuracy (which is directly determined by the quality of cross-attended spatial features) should improve. Target: `mpjpe_rel_val < 400` (best prior 333mm — idea008/design002; typical ~420mm).
- **Secondary gain — body MPJPE**: cross-attention on K=32 well-organised super-tokens should produce cleaner gradients than cross-attention on 960 mixed body/background tokens. Target: `mpjpe_body_val < 155` (best prior 156.6mm — idea002/design003).
- **Design A (K=32, 1 layer)**: diagnostic — does learned spatial aggregation alone help? Tests the compression hypothesis cleanly. Expected composite_val < 335 at stage-1.
- **Design B (K=64, positional slot init)**: tests whether warm-started spatial partition slots help convergence within 20 epochs. Expected composite_val < 330 at stage-1.
- **Design C (K=32 + 2 decoder layers)**: primary bet — combining token compression with depth-enabled stacking. Expected composite_val < 320 at stage-1 (best prior: 328.14 — idea013/design003).
- **Composite target (stage-2)**: aim for `composite_val < 220` (best prior: 224.52 — idea001/design001).

---

## Risk and Mitigation

- **Slot attention cold-start**: at initialisation, all K slot queries attend uniformly to all 960 spatial tokens (random key-query alignment). Early training may produce low-quality super-tokens until the slot attention module converges. Mitigation for Design A/C: use trunc-normal init for slot queries (same std=0.02 as joint queries); the slot attention module is a standard MultiheadAttention which initialises well with default PyTorch init. For Design B, positional super-token initialisation directly addresses this risk by grounding slot queries spatially.
- **Super-token information bottleneck**: K=32 may be too few tokens to represent all the information needed for 22 body joints. Each super-token must serve multiple joint queries. Mitigation: K=32 for body joints only (22 joints) and K=64 for all 70 queries provides 1.5–2.1 tokens per joint on average. In practice, shared context (torso region, limb context) is highly redundant, so 32 super-tokens should suffice. If stage-1 shows degradation, Design B's K=64 is a fallback.
- **Gradient through softmax pooling**: the slot-attention backward pass computes gradients through the 960-key softmax. AMP with dynamic loss scaling is already configured — the softmax gradient is well-behaved (bounded). No additional numerical risk beyond the baseline cross-attention.
- **Memory increase from slot attention module**: the slot attention adds one additional MultiheadAttention (~1.3M params, ~10MB activation) to the forward. This is offset by the 30× reduction in decoder cross-attention activation. Net memory change is strongly negative (less total activation memory than baseline).
- **num_decoder_layers config conflict with baseline decoder_layer**: the Designer should ensure that when `num_decoder_layers=1`, the module falls back to the single `decoder_layer` path (or equivalently, `decoder_layers = nn.ModuleList([_DecoderLayer(...)])` with a single element). Backward compatibility is simple: use `nn.ModuleList` for all designs and set length to 1 or 2.
- **Interaction with idea008 (body-focused 22 queries)**: super-token pooling is strictly orthogonal to query reduction. Combining 22 body queries (idea008) with K=32 super-tokens would reduce cross-attention to O(22×32) — 97% reduction vs. baseline O(70×960). This composition is promising but left to a future idea to avoid overlap.
- **Interaction with idea009 (spatial token dropout)**: idea009 randomly dropped spatial tokens; super-token pooling replaces that with learned aggregation. These are mutually exclusive in their spatial-token handling, so they should not be combined.
- **MMEngine config constraint**: `num_super_tokens`, `num_decoder_layers` are int literals; `slot_pos_init` is a bool literal; `aux_loss_weight` is a float literal. No imports required. Fully compliant with MMEngine no-Python-imports restriction.
- **Eval/inference compatibility**: `predict()` calls `self.forward(feats)` unchanged — the super-token path is inside `forward()`. The output tensor shapes (`(B, 70, 3)` joints, `(B, 1)` depth, `(B, 2)` UV) are unchanged. `BedlamMPJPEMetric` and `TrainMPJPEAveragingHook` see identical interfaces.
- **Feature grid resolution assumption**: the slot positional init in Design B assumes H'=24, W'=40 (consistent with backbone output for 640×384 input at 1/16 stride). This should be verified by the Designer at forward time. If the grid differs, the slot init logic needs adjustment — but the trunc-normal fallback (Design A) still works.
