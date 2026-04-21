**Idea Name:** Learnable Cross-Attention Spatial Bias for Anatomically-Grounded Joint Localization

**Approach:** Add a learnable additive bias to the cross-attention logits for each joint query over the spatial feature grid — implemented via the `attn_mask` argument of `nn.MultiheadAttention` — so that each query can learn a soft spatial prior over the 24×40 feature map indicating where in the image it expects its corresponding body part to appear, initialized to zero for exact baseline equivalence.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Missing Spatial Prior in Cross-Attention

The baseline decoder's cross-attention computes attention weights between joint query `i` and each of the 960 spatial tokens purely via the learned dot-product score:

```
a_{i,j} = (q_i @ k_j^T) / sqrt(d_head)
```

where `q_i ∈ R^{d_head}` is the projection of joint query embedding `i`, and `k_j ∈ R^{d_head}` is the projection of spatial token `j`. The attention is then `softmax_j(a_{i,j})`.

This dot-product score is a **purely semantic** similarity: query `i` learns to attend to spatial tokens whose *content* is most relevant for predicting joint `i`. There is no spatial structure or positional prior — a pelvis query could, in principle, develop maximum attention weight on the upper-left corner of the feature grid if the learned key-query similarity happens to be highest there.

In practice, the model must infer spatial structure implicitly: the query embeddings must encode both *what the joint looks like* (semantic) and *where it should look* (spatial), competing for the same hidden_dim=256 representational budget. This is a fundamental capacity bottleneck: the 256-dimensional query embedding must simultaneously solve the spatial routing problem and the joint-feature encoding problem.

### Anatomical Spatial Regularity in BEDLAM2

After `CropPersonRGBD`, the input is a 640×384 crop centred on the target person. The backbone produces a 24×40 feature grid at 1/16 resolution. Due to the crop centering:

- **Pelvis / root (joint 0)**: appears near the vertical centre of the crop, horizontally centred.
- **Head / neck (joints 9–11)**: appear in the upper-centre region of the crop.
- **Feet (joints 7–8)**: appear in the lower region of the crop.
- **Hands (joints 22–69 and wrists 17–18)**: appear in the middle-to-upper half, laterally offset.
- **Spine / chest**: appear in the central vertical band.

These spatial locations are not fixed (people can lean, crouch, etc.), but they have strong **marginal distributions** in the crop frame. A model that cannot exploit this prior must learn it implicitly from the dot-product scores, consuming query embedding capacity that could be better spent on semantic joint recognition.

### The Proposed Fix: Learnable Additive Cross-Attention Bias

For each of the `num_joints` queries, add a learnable scalar bias `B_i[h, w]` to the cross-attention logit for that query attending to the spatial token at grid position `(h, w)`:

```
a_{i,j}^{biased} = (q_i @ k_j^T) / sqrt(d_head) + B_i[j]
```

where `j` indexes the flattened spatial position and `B_i ∈ R^{H' × W'}` (reshaped to `R^{H'W'}` for the softmax). The attended output is then `softmax_j(a_{i,j}^{biased})`.

This decouples the **spatial routing** problem from the **semantic matching** problem:
- The bias `B_i` handles "where to look" — learned as a spatial attention prior for joint `i`.
- The dot-product `q_i @ k_j^T` handles "what to look for" — learned as semantic feature matching.

Crucially, when `B_i = 0` for all `i` (initialisation), the computation is **exactly identical to the baseline**. The bias introduces zero perturbation at training start and is learned incrementally. This is the same zero-init strategy used successfully in idea006 (self-attention bias), idea011 (coordinate encoder), and idea016 (FiLM conditioning).

### Implementation: Three Lines via `attn_mask`

`nn.MultiheadAttention(batch_first=True)` accepts an optional `attn_mask: Tensor` argument of shape `(tgt_len, src_len)` that is **added to the attention logits before softmax**. This is precisely our bias:

```python
# In _DecoderLayer.forward(), replace:
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]

# With:
bias_flat = cross_attn_bias.view(q.shape[1], -1)   # (num_joints, H'*W')
q2 = self.cross_attn(q, spatial_tokens, spatial_tokens, attn_mask=bias_flat)[0]
```

The `attn_mask` is broadcast over batch and head dimensions by PyTorch's `nn.MultiheadAttention` implementation, so a single `(num_joints, H'W')` parameter serves all batch elements and all attention heads uniformly. This is the correct semantics: the spatial prior for joint `i` is the same regardless of the specific image (it is a prior, not an image-specific attention map).

**Parameter count**: `num_joints × H' × W' = 70 × 24 × 40 = 67,200` scalars ≈ 262KB. Entirely negligible relative to the ~300M parameter model.

### Differentiation from All 20 Prior Ideas

| Prior Idea | Attention Mechanism Modified | Key Difference |
|---|---|---|
| idea006 | Learnable additive bias to **self-attention** logits (query-to-query, `70×70`) | Different: (1) self-attention not cross-attention; (2) captures query-query structural relations, not spatial routing |
| idea007 | Multiplicative gating of cross-attention **output values** (post-softmax, channel-level) | Different: gating is applied to aggregated values *after* softmax, not to logits *before* softmax — fundamentally different effect on the attention distribution shape |
| idea019 | Deformable sparse sampling: selects a small set of query-specific spatial tokens before cross-attention | Different: changes *which tokens exist*, not adds a bias over a full fixed token set |
| idea020 | Per-query temperature scaling: divides all cross-attention logits by a scalar `τ_i` | Different: temperature controls the *sharpness* (entropy) of the distribution uniformly across all positions; bias controls the *location* (which positions get higher weight) |
| idea009 | Random spatial token dropout: randomly removes tokens | Different: unstructured, non-learnable, not location-specific |
| idea015 | Super-token pooling: compresses 960 tokens into K learned slots | Different: reduces the token set size, does not add a position-specific logit bias |
| idea018 | Depth-gated cross-attention | Different: gates the *contribution* of depth-specific tokens; not per-query spatial logit bias |

This is the **first idea to add a learnable additive bias to the cross-attention logits**, operating directly on the spatial attention distribution *per query* over the *full 24×40 spatial grid*. It is orthogonal to every prior idea.

### Grounding in Observed Results

- **idea006** (learnable self-attention bias, zero-init): best design achieved composite_val = 343.74 at stage-1, improvement from baseline 346.58. The self-attention bias helps moderately but is limited because self-attention captures query-to-query structural relationships, not where each query should look in the image. A cross-attention bias directly addresses the routing problem that self-attention bias cannot.

- **idea020** (per-query temperature): the temperature controls attention *sharpness*. The proposed idea controls attention *location*. These two mechanisms are highly complementary: a query for the ankle joint should look in the lower half of the image (`B_ankle` assigns high bias to lower rows) AND look sharply at a small region (`τ_ankle < 1.0`). Together they provide independent location and sharpness control.

- **Body MPJPE floor at ~156mm**: all prior ideas improving body MPJPE do so by (a) reducing query contamination (idea008), (b) adding decoder capacity (idea001), (c) adding structural output priors (idea013). The proposed idea attacks a different bottleneck: the cross-attention routing capacity. If query embeddings are freed from encoding spatial location (by offloading it to the bias), they can allocate more capacity to semantic feature matching, potentially breaking the 156mm floor.

- **`mpjpe_rel_val` at 333–440mm**: relative MPJPE measures pose shape accuracy after Procrustes alignment (removes absolute position/scale). Anatomically-grounded spatial attention should improve the quality of spatial features used for each joint, improving shape accuracy. Target: improve from best prior 333mm (idea008/design002).

---

## Proposed Variations

### Design A — Full Spatial Bias Matrix, Zero-Initialized (diagnostic)

A single learnable `nn.Parameter(torch.zeros(num_joints, H_feat * W_feat))` = `(70, 960)` initialized to zero. No warm-start: the spatial prior is learned entirely from data.

This is the cleanest diagnostic: does a learnable spatial logit bias improve over the baseline? Since the init is zero (exact baseline), any improvement is attributable purely to the learned spatial routing.

Key implementation:
- `self.cross_attn_bias = nn.Parameter(torch.zeros(num_joints, feat_h * feat_w))` in `__init__`. `feat_h=24, feat_w=40` are integer literals in config.
- In `_DecoderLayer.forward()`: pass `attn_mask=cross_attn_bias` to `self.cross_attn(...)`.
- Since the `_DecoderLayer` doesn't own the bias (it lives in the head), pass the bias as an argument: `decoder_layer(queries, spatial_tokens, cross_attn_bias=self.cross_attn_bias)`.

Config kwargs: `use_cross_attn_bias=True`, `feat_h=24`, `feat_w=40` (bool/int literals).

### Design B — Low-Rank Factored Bias (row × column, parameter-efficient)

Instead of a full `(70, 960)` bias matrix, factorize it as an outer product of row-bias and column-bias:
```
B_i[h, w] = u_i[h] + v_i[w]
```
where `u_i ∈ R^{H'=24}` is a per-joint row bias and `v_i ∈ R^{W'=40}` is a per-joint column bias. The full cross-attention logit for query `i` attending to token at (h, w) is:
```
a_{i,(h,w)}^{biased} = dot_product + u_i[h] + v_i[w]
```

This **additive factorization** is the natural low-rank prior for 2D spatial attention: it independently controls row preference (which vertical band of the image to look at) and column preference (which horizontal band). For the human body in a centred crop, many joints have strong row preferences (head=top, feet=bottom) but weak column preferences (most joints are roughly centred). The factored parameterization captures this efficiently.

Parameter count: `num_joints × (H' + W') = 70 × 64 = 4,480` scalars — 15× fewer than Design A.

Implementation:
- `self.cross_attn_bias_row = nn.Parameter(torch.zeros(num_joints, feat_h))` — `(70, 24)`
- `self.cross_attn_bias_col = nn.Parameter(torch.zeros(num_joints, feat_w))` — `(70, 40)`
- Full bias: `bias = bias_row.unsqueeze(-1) + bias_col.unsqueeze(-2)` → `(70, 24, 40)`, then `.view(70, 960)`.

Both initialized to zero → exact baseline at start.

Config kwarg: `cross_attn_bias_type='factored'` (str literal).

### Design C — Low-Rank Factored Bias with Anatomical Warm-Start

Same factored parameterization as Design B, but **warm-start the row biases** based on the approximate vertical position of each of the 22 body joints in the BEDLAM2 RGBD crop.

After `CropPersonRGBD` centres the person and scales to 640×384, the body joints roughly span:
- Row 0 (top of crop, H/16=0): head
- Row 12 (middle): pelvis
- Row 23 (bottom): feet

A Gaussian warm-start: for body joint `i`, set `u_i[h] = α * exp(-(h - μ_i)^2 / (2σ^2))` where `μ_i ∈ [0, 23]` is the expected row (at 1/16 resolution) and `σ=4` (spans ~3 grid cells). The `α = 1.0` controls the initial bias strength. Column biases `v_i` remain zero-initialized.

The 22 expected row positions (at H'=24 scale, so 0=top, 23=bottom) are hardcoded as integer literals in config:
```python
# joint_row_prior: list of 22 floats, one per body joint (0=pelvis to 21=...)
# From top of crop (0) to bottom (23). Zero-indexed.
joint_row_prior = [12.0, 10.0, 14.0, 12.0, 9.0, 15.0, 7.0, 19.0, 21.0, 5.0,
                    3.0, 2.0, 11.0, 13.0, 11.0, 13.0, 9.0, 9.0, 15.0, 15.0, 12.0, 12.0]
```
(Designer should validate these approximate priors from data or skeleton definition; they are soft initializations that the model will refine during training.)

Hand joints (22–69) use zero-initialized row biases (since hand positions are highly variable with arms in different poses).

This warm-start is expected to accelerate convergence in the first ~5 epochs compared to Design B, where the model must learn the spatial prior from scratch.

Config kwargs: `cross_attn_bias_type='factored_warmstart'`, `joint_row_prior=[...]` (str + list-of-float literals).

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

**1. `_DecoderLayer` signature change:**

Add optional `cross_attn_bias` argument to `forward()`:
```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            cross_attn_bias: torch.Tensor | None = None) -> torch.Tensor:
    ...
    # Cross-attention:
    q = self.norm2(queries)
    if cross_attn_bias is not None:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                              attn_mask=cross_attn_bias.to(q.dtype))[0]
    else:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
    queries = queries + self.dropout2(q2)
    ...
```

The `.to(q.dtype)` cast ensures AMP float16 compatibility (bias stored as float32 parameter, cast to match query dtype at runtime).

**2. `Pose3dTransformerHead.__init__` changes:**

New constructor kwargs (all with defaults for backward compatibility):
```python
use_cross_attn_bias: bool = False
cross_attn_bias_type: str = 'full'   # 'full' | 'factored' | 'factored_warmstart'
feat_h: int = 24
feat_w: int = 40
joint_row_prior: list = None         # only used for 'factored_warmstart'
```

Parameter allocation:
```python
if use_cross_attn_bias:
    if cross_attn_bias_type == 'full':
        self.cross_attn_bias = nn.Parameter(
            torch.zeros(num_joints, feat_h * feat_w))
    else:  # 'factored' or 'factored_warmstart'
        self.cross_attn_bias_row = nn.Parameter(
            torch.zeros(num_joints, feat_h))
        self.cross_attn_bias_col = nn.Parameter(
            torch.zeros(num_joints, feat_w))
```

Warm-start logic (in `_init_head_weights()` for Design C):
```python
if cross_attn_bias_type == 'factored_warmstart' and joint_row_prior is not None:
    h_coords = torch.arange(feat_h, dtype=torch.float32)   # (24,)
    sigma = 4.0
    for i, mu in enumerate(joint_row_prior[:22]):
        gauss = torch.exp(-(h_coords - mu) ** 2 / (2 * sigma ** 2))
        self.cross_attn_bias_row.data[i] = gauss   # warm-start body joints
    # hand joints (22-69) remain zero-initialized
```

**3. `Pose3dTransformerHead.forward()` changes:**

Pass the bias to `decoder_layer`:
```python
if self.use_cross_attn_bias:
    if self.cross_attn_bias_type == 'full':
        bias = self.cross_attn_bias     # (num_joints, H'W')
    else:
        # Factored: combine row and column biases via broadcasting
        bias = (self.cross_attn_bias_row.unsqueeze(-1) +
                self.cross_attn_bias_col.unsqueeze(-2))   # (num_joints, feat_h, feat_w)
        bias = bias.view(self.num_joints, -1)              # (num_joints, H'W')
    decoded = self.decoder_layer(queries, spatial, cross_attn_bias=bias)
else:
    decoded = self.decoder_layer(queries, spatial)
```

**4. `loss()` and `predict()`**: unchanged — they call `self.forward(feats)` which routes through the bias path. No change to loss computation or output shape.

**5. `_init_head_weights()`**: zero-init is the default for `nn.Parameter(torch.zeros(...))`. Only Design C adds non-trivial warm-start logic as described above.

### `config.py`

**Design A:**
```python
use_cross_attn_bias=True,
cross_attn_bias_type='full',
feat_h=24,
feat_w=40,
```

**Design B:**
```python
use_cross_attn_bias=True,
cross_attn_bias_type='factored',
feat_h=24,
feat_w=40,
```

**Design C:**
```python
use_cross_attn_bias=True,
cross_attn_bias_type='factored_warmstart',
feat_h=24,
feat_w=40,
joint_row_prior=[12.0, 10.0, 14.0, 12.0, 9.0, 15.0, 7.0, 19.0, 21.0, 5.0,
                  3.0, 2.0, 11.0, 13.0, 11.0, 13.0, 9.0, 9.0, 15.0, 15.0, 12.0, 12.0],
```

All values are bool, str, int, or list-of-float literals. No Python import statements. Fully compliant with the MMEngine config constraint.

---

## Expected Outcome

- **Primary gain — body MPJPE**: freeing query embeddings from encoding spatial routing should allow them to better encode joint-specific semantic features. Body joint embeddings that spend less capacity on "where to look" can spend more on "what to look for." Target: `mpjpe_body_val < 185` at stage-1 (vs. baseline 195.7 mm), `< 152` at stage-2 (vs. best prior 156.6 mm — idea002/design003).

- **Secondary gain — `mpjpe_rel_val`**: root-relative MPJPE (Procrustes-aligned) reflects how well the predicted pose shape matches GT pose. Spatially-grounded cross-attention should produce sharper, more discriminative spatial features per joint, improving shape accuracy. Target: `mpjpe_rel_val < 380` at stage-1 (vs. baseline 438.7 mm), with potential to approach the idea008/design002 outlier of 333 mm at stage-2.

- **Pelvis MPJPE**: expected neutral to mild positive. The pelvis query (index 0) gains its own spatial bias; it can learn to look at the full body region (flattish bias across the crop centre). Pelvis regression does not benefit as much as joint regression from spatial routing since it requires global context — the learned bias is unlikely to help strongly here.

- **Design A (full bias, 67K params)**: diagnostic — does unconstrained spatial learning help? Higher parameter count but no structural assumption. Expected composite_val < 340 at stage-1.

- **Design B (factored bias, 4.5K params)**: tests whether the additive row × column decomposition is sufficient. The factored form imposes a soft "row independence from column" prior, which matches human anatomy (joints have independent vertical and horizontal distributions in centred crops). Expected composite_val < 333 at stage-1.

- **Design C (factored + warm-start)**: anatomical initialisation accelerates convergence. In 20 epochs, a well-initialized spatial prior can be refined more than a cold-started one. Expected to outperform Design B by 5–10 mm body MPJPE within the same epoch budget. Expected composite_val < 328 at stage-1.

- **Composite target (stage-2)**: aim for `composite_val < 218` (vs. best prior 224.52 — idea001/design001).

---

## Risk and Mitigation

- **`attn_mask` semantics in `nn.MultiheadAttention`**: PyTorch's `attn_mask` is added to logits *before* softmax, which is exactly what we want. However, the mask is expected to be additive (not multiplicative), and values of `-inf` produce attention weights of 0. Our bias uses finite float values (from a zero-init or small Gaussian init), so no masking occurs — all spatial positions retain positive attention weight. This is correct.

- **`attn_mask` shape for `batch_first=True`**: for `batch_first=True`, the `attn_mask` is still `(tgt_len, src_len)` (not batch-first for the mask argument). The Designer should verify this in the PyTorch documentation for their torch version (2.x). If needed, the mask can be expanded to `(B * num_heads, tgt_len, src_len)` for explicit batch+head control, but the broadcast form `(tgt_len, src_len)` should suffice.

- **AMP float16 compatibility**: the bias parameter is float32; the `q.dtype` cast in `_DecoderLayer.forward()` (`.to(q.dtype)`) ensures the bias is cast to float16 when AMP is active. Since the bias values are small (near zero at init; Gaussian ~1.0 at most for Design C), there is no float16 overflow risk.

- **Gradient flow to bias**: the gradient of the cross-attention output with respect to `B_i[j]` is well-conditioned — it is the cross-attention output weight `softmax_j(a_{i,j})`, which is always in `[0, 1]`. No gradient explosion risk.

- **Capacity of full bias (Design A)**: 70 × 960 = 67,200 independent scalars is a relatively large parameter set compared to the small decoder head. On 20 epochs × ~800 train steps = 16,000 gradient steps, each parameter receives one gradient per step. Risk of overfitting to training set spatial distributions. Mitigation: the CropPersonRGBD augmentation (`NoisyBBoxTransform`) adds spatial noise to the crop bounding box, creating variation in joint locations. Additionally, the factored alternatives (Designs B/C) are lower-risk fallbacks.

- **Warm-start prior correctness (Design C)**: the `joint_row_prior` values are approximate estimates from BEDLAM2 skeleton anatomy. If they are systematically off, the warm-start could hurt early-epoch convergence. Mitigation: (1) the prior is *soft* (Gaussian with σ=4 grid cells), not hard; (2) the model can correct the prior within a few epochs; (3) the column biases remain zero, so the row prior provides directional but not over-specified guidance. The Designer should validate the prior by sampling a few training examples and computing empirical joint row distributions.

- **Interaction with idea006 (self-attention bias)**: idea006 biases *query-query* self-attention logits; this idea biases *query-spatial* cross-attention logits. These are orthogonal and additive. Combining both in a future idea is possible: the self-attention bias controls query-to-query structural relationships, while the cross-attention bias controls each query's spatial routing. Together they provide full control over both attention stages.

- **Interaction with idea020 (per-query temperature)**: per-query temperature scales the magnitude of all cross-attention logits for query `i` by `1/τ_i`; cross-attention bias adds a spatial pattern to those logits. These are orthogonal: bias controls *location* preference, temperature controls *sharpness* of the spatial distribution. A future idea could combine both mechanisms.

- **Interaction with idea008 (body-focused 22-query decoder)**: if using `num_body_queries=22`, the bias parameter shape becomes `(22, 960)` instead of `(70, 960)`. The Designer should ensure the `num_joints` arg that controls the bias shape matches the number of decoder queries, not the total output joints. For Design C, `joint_row_prior` has 22 entries (matching the 22 body joints).

- **`feat_h=24, feat_w=40` hardcoded**: for the 640×384 input at 1/16 backbone stride, `H'=40, W'=24` (height=640/16=40, width=384/16=24). **Note**: the current architecture has `H'=40, W'=24` (height-first ordering in the feature map), while the spatial tokens are flattened row-major as `H'*W'=960`. The Designer should verify the exact ordering convention used in `feat.flatten(2).transpose(1, 2)` in the baseline `forward()` — if `feat` is `(B, C, H, W)` with `H=40, W=24`, then `feat_h=40, feat_w=24`. The `attn_mask` shape must match this exactly. Config should set `feat_h=40, feat_w=24` (not 24×40) if this is the convention. The Architect notes this ambiguity and flags it for the Designer to verify.

- **Backward-compatibility**: all new kwargs have default values (`use_cross_attn_bias=False` etc.), so existing baseline configs that don't specify these kwargs continue to work without modification.

- **MMEngine config constraint**: `use_cross_attn_bias` is a bool literal; `cross_attn_bias_type` is a str literal; `feat_h`, `feat_w` are int literals; `joint_row_prior` is a list of float literals. No Python `import` statements required. Fully compliant.

- **Memory**: one `(70, 960)` float32 parameter = 268KB; or `(70, 24) + (70, 40)` = 11KB for factored designs. The outer-product expansion `(70, 24, 40)` is computed once per forward pass from two small parameter tensors — O(70 × 64) extra FLOPs. Negligible on 2080 Ti.

- **Eval/inference compatibility**: `predict()` calls `self.forward(feats)`, which applies the cross-attention bias. The bias is part of the model's `state_dict` and is loaded correctly by `CheckpointHook`. No special inference mode needed.
