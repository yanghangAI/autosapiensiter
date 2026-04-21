**Idea Name:** Per-Query Deformable Spatial Sampling for Joint-Specific Cross-Attention

**Approach:** Replace the standard dense cross-attention over all 960 spatial tokens with a lightweight deformable sampling mechanism: each joint query predicts a small set of 2D reference point offsets (K_s=8 points), samples the spatial feature map at those locations via bilinear interpolation, and cross-attends only to the K_s sampled features — giving each joint its own anatomically-grounded, query-conditioned attention footprint rather than sharing a uniform full-map attention.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder's cross-attention operates as follows:

```
Q: (B, 70, 256)    — joint queries
K: (B, 960, 256)   — full 24×40 spatial token set (projected backbone features)
V: (B, 960, 256)   — same spatial tokens
attn_weights = softmax(Q @ K^T / sqrt(head_dim))   # (B, num_heads, 70, 960)
output = attn_weights @ V                           # (B, 70, 256)
```

Every joint query attends softmax over all 960 spatial tokens uniformly. This creates three concrete problems.

### Problem 1: Diffuse attention in early training

At initialisation, `Q @ K^T` produces near-random logits; the softmax over 960 tokens is nearly uniform — each token receives weight ≈ 1/960 ≈ 0.1%. The gradient signal flowing back to individual spatial tokens is correspondingly small. The model must learn, over 20 epochs, to concentrate attention onto the ~30–60% of tokens that correspond to the human body region. This is a hard attention-focusing problem from a cold start.

Contrast with **idea003 (content-adaptive query init)**, which addressed the difficulty of cold-starting the joint *queries*. This idea addresses the complementary problem: the difficulty of cold-starting the *spatial attention distribution* — helping the model find the right tokens, not just produce the right queries.

### Problem 2: All joint queries share the same full spatial context

The knee joint query and the shoulder joint query both cross-attend over the same 960 tokens, including each other's body region and the entire background. There is no inductive bias that the knee should attend to lower-body spatial tokens and the shoulder to upper-body tokens. The model must learn this purely from data — a task made harder by the short 20-epoch training window and the strong structural priors that RGBD data contains.

This is distinct from **idea007 (joint-group spatial routing)**, which applied soft, per-group gating weights to the cross-attention values (a multiplicative bias shared within joint groups). In that design, all joints in a group still attend over all 960 tokens with modified weighting; the attention is still dense. In this idea, each joint query predicts its own sparse set of sampling locations — a per-query, per-sample, learned reference point selection that reduces the effective spatial set from 960 to K_s=8 points per query.

### Problem 3: Cross-attention memory scales with full 960-token set

The full 70×960 cross-attention is the bottleneck for adding decoder layers (as confirmed by idea015's analysis). Deformable sampling reduces the effective key-value set from 960 to K_s per query — a 120× reduction (960÷8). This opens headroom for 2–3 decoder layers with deformable sampling within the same VRAM budget as a single standard cross-attention layer.

### The Deformable Sampling Mechanism

Drawing on Deformable DETR (Zhu et al., ICLR 2021), for each joint query `q_i ∈ R^{hidden_dim}`, a small **offset network** predicts K_s 2D offsets around a learnable reference point `r_i ∈ R^2` (initialised to the centre of the feature grid, normalised to [0, 1]²):

```
offset_pred_i = offset_net(q_i)     # (K_s, 2) — predicted offsets from reference
sample_locs_i = r_i + offset_pred_i  # (K_s, 2) — K_s sampling locations in [0,1]²
sampled_features_i = grid_sample(spatial_grid, sample_locs_i)  # (K_s, hidden_dim)
```

Then, the joint query cross-attends only to its K_s sampled features:

```
output_i = cross_attn_sparse(q_i, sampled_features_i, sampled_features_i)  # (hidden_dim,)
```

The total cross-attention for a batch has dimensions `(B, num_joints, K_s, hidden_dim)` instead of `(B, num_joints, 960, hidden_dim)` — a 120× reduction in the attention matrix.

### Key Implementation Insight

The spatial tokens in the baseline are already in a grid structure: `feats[-1]` has shape `(B, 1024, 24, 40)` before `flatten(2)`. For deformable sampling, we **do not flatten the spatial tokens** — instead, we keep the `(B, hidden_dim, H', W')` representation after input projection, and use `torch.nn.functional.grid_sample` to bilinearly interpolate at arbitrary `(u, v)` positions.

The offset network is a small two-layer MLP per query (shared weights across queries):
```
offset_net: Linear(hidden_dim, hidden_dim//4) → GELU → Linear(hidden_dim//4, K_s * 2)
```
Initialised with near-zero output weights so that at start, all K_s offsets are ≈ 0 and all queries sample near the grid centre (equivalent to a coarse but stable initialisation). The reference points `r_i` are learnable `nn.Parameter(torch.zeros(num_joints, 2))` initialised at grid centre (0.5, 0.5), providing per-joint anchors that the model learns to move toward each joint's expected anatomical region.

### Why K_s = 8

- 8 points per query × 70 queries = 560 total sampling locations vs. 960 full-grid tokens → 42% of the standard cross-attention key-value cost, with the added benefit of per-query spatial specificity.
- In Deformable DETR (4 attention heads × 4 sampling points = 16 per query), 8 points per query is a conservative choice that trades coverage for efficiency.
- For joint queries in a 640×384 crop at 1/16 stride (24×40 grid), 8 points cover approximately an 8-pixel-radius neighbourhood at grid resolution — sufficient to capture a joint and its immediate limb context.
- The offset network output shape is `K_s=8` × 2 coordinates. At K_s=4 (smallest sensible), spatial context may be insufficient; at K_s=16, memory cost equals the original for 60 queries. K_s=8 is the stable middle.

### Differentiation from All Prior Ideas

| Idea | Spatial Token Treatment | Key Difference |
|---|---|---|
| idea007 | Soft group-level gating of cross-attn values | Shared dense attention with group-level weight; still 960 tokens |
| idea009 | Random spatial token dropout | Unstructured removal; not per-query, not learned |
| idea015 | Slot attention: 960 → K super-tokens | Learned *compression* shared by all queries; not per-query |
| idea018 | Depth-gate on cross-attn logits | Shared per-token gate (not per-query); still dense attention |
| **idea019** | **Per-query learned reference + sparse bilinear sampling** | **Each joint query independently selects its K_s spatial locations; no dense 960-token attention** |

idea019 is the **first mechanism where cross-attention is explicitly per-query in its spatial footprint**. Every prior idea (including idea007, idea015, idea018) has maintained a shared spatial token set (possibly compressed or gated, but always shared across queries). Here, the sampled token set is *different for every joint query in every batch element*.

### Grounding in Observed Results

1. **idea001 (multi-layer decoder)**: best stage-2 composite 224.52, by stacking 2 decoder layers. The gain is attributed to progressive refinement. Deformable sampling enables 2–3 decoder layers within the same VRAM budget (Design C), potentially amplifying this gain.

2. **idea007 (joint-group spatial routing)**: design002 at stage-1 composite 339.72 (third-best stage-1). The spatial routing concept helps — confirming that spatially-directed cross-attention is beneficial. Deformable sampling is a stronger form of spatial routing: instead of group-level channel gating, it is per-query location selection.

3. **idea003 (content-adaptive query init)**: design002 composite 225.44 stage-2 (second best). Content-adaptive query initialization helps convergence. Combining learned reference points (this idea) with content-adaptive query init could be synergistic, but that is a composition for a future idea.

4. **idea009 (spatial token dropout)**: all designs performed poorly (349–375 composite at stage-1). This confirms that **unstructured** removal of spatial tokens hurts because it randomly discards relevant information. Deformable sampling is the complementary insight: *structured, learned* spatial selection preserves and focuses the relevant signal.

5. **Body MPJPE floor ~155–195mm** across all 18 ideas. The persistent floor suggests that the cross-attention is not saturated by better queries or losses; the spatial information integration mechanism itself may be limiting. A fundamentally different spatial integration (deformable sampling) is a new lever.

---

## Proposed Variations

### Design A — K_s=8 deformable sampling, single decoder layer (minimal)

Replace the standard cross-attention in `_DecoderLayer` with a deformable sampling module. A single decoder layer with deformable sampling over K_s=8 points. The offset network is a shared 2-layer MLP applied per joint query.

Architecture change relative to baseline:
```
# Baseline:
decoded = decoder_layer(queries, spatial_flat)   # spatial_flat: (B, 960, 256)

# Design A:
decoded = deform_decoder_layer(queries, spatial_grid, ref_points, offset_net)
# spatial_grid: (B, 256, 24, 40) — NOT flattened; kept as 2D map
# ref_points: (num_joints, 2) learnable, initialised to (0.5, 0.5)
# offset_net: Linear(256, 256//4) → GELU → Linear(256//4, K_s*2), output near-zero init
```

Self-attention (over queries) is unchanged. Only the cross-attention step is replaced by deformable sampling + sparse MHA.

Memory: K_s=8 cross-attn per query → (B, num_heads, 70, K_s=8) attention matrix per head vs. (B, num_heads, 70, 960) baseline. VRAM for cross-attn activation: 70×8 / (70×960) = 0.83% of baseline cross-attn activation. Overall head VRAM reduces by ~30–40%.

Config kwargs: `deform_num_points=8`, `deform_hidden_dim=64` (int literals).

### Design B — K_s=8 deformable sampling, body-only 22 queries, linear hand recovery (compositional with idea008)

Apply deformable sampling (K_s=8) in the 22-query body-only decoder setting from idea008/design002. The 22 body joint queries each learn their own reference point and offset prediction. Hand joints (22–69) are recovered via the same linear projection as idea008/design002. The pelvis token (query 0) gets its own reference point — the model can learn to focus the pelvis depth token on the global spatial context by placing its reference near the image centre.

Rationale for combination: idea008/design002 achieved outstanding `mpjpe_rel_val` (333.2mm) and `mpjpe_abs` (533.8mm) at stage-2 by removing hand-query contamination from cross-attention. Adding deformable sampling on top of the 22-query decoder gives each of the 22 body queries its own spatial attention footprint — potentially further improving relative pose accuracy.

This is the compositional design with the highest empirical support: both 22-query body decoding (idea008) and spatial routing (idea007) improved results independently; their combination with deformable sampling (per-query spatial selection) has not been tried.

Config kwargs: `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `deform_num_points=8`, `deform_hidden_dim=64`.

### Design C — K_s=8 deformable sampling + 2-layer decoder + intermediate supervision (max capacity within budget)

Stack 2 deformable decoder layers using K_s=8 points per query and 22 body queries (Design B setting). The memory savings from (a) 22 queries instead of 70 and (b) K_s=8 sparse sampling instead of 960 tokens enable 2 decoder layers at significantly less VRAM than the baseline single layer with 70 queries.

Cross-attention VRAM per layer (rough estimate):
- Baseline: B × 8_heads × 70 × 960 = 537,600 elements per layer
- Design C: B × 8_heads × 22 × 8 = 1,408 elements per layer — 382× less VRAM for cross-attention

With 2 layers: 2 × 1,408 = 2,816 vs. baseline 537,600 — still 191× less. This budget is easily absorbed on the 2080 Ti.

Intermediate body joint supervision at layer 1 (weight 0.4) to prevent gradient vanishing at the first deformable layer.

Hand linear recovery with auxiliary loss weight 0.1 (same as Design B).

This design tests whether the compounded memory savings of 22 queries + sparse sampling can fund 2 decoder layers of progressive refinement — the same combination that drove idea017's design motivation but with deformable sampling replacing standard cross-attention.

Config kwargs: `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`, `deform_num_points=8`, `deform_hidden_dim=64`.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

#### New module: `_DeformableDecoderLayer`

```python
class _DeformableDecoderLayer(nn.Module):
    """Transformer decoder layer with deformable sparse cross-attention.

    Self-attention is unchanged (all queries attend to each other).
    Cross-attention is replaced by per-query deformable sampling + sparse MHA.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.1, num_points: int = 8,
                 deform_hidden_dim: int = 64, num_queries: int = 70):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        # Self-attention (unchanged)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Deformable offset network (shared across all queries)
        # Input: query feature (B, num_queries, embed_dim)
        # Output: (B, num_queries, num_points*2) offset values
        self.offset_net = nn.Sequential(
            nn.Linear(embed_dim, deform_hidden_dim),
            nn.GELU(),
            nn.Linear(deform_hidden_dim, num_points * 2),
        )
        # Learnable reference points: (num_queries, 2) initialised to grid centre
        self.ref_points = nn.Parameter(
            torch.full((num_queries, 2), 0.5))  # [0,1]² grid coords

        # Sparse cross-attention: q × (K_s sampled features)
        # Implemented as a linear attention over K_s points
        self.attn_weight_net = nn.Linear(embed_dim, num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sample_spatial_features(
        self,
        queries: torch.Tensor,      # (B, num_queries, embed_dim)
        spatial_grid: torch.Tensor, # (B, embed_dim, H', W')
    ) -> torch.Tensor:
        """Sample K_s features per query via bilinear interpolation.

        Returns:
            (B, num_queries, num_points, embed_dim)
        """
        B, Nq, D = queries.shape
        _, _, H, W = spatial_grid.shape

        # Predict offsets from query features
        offsets = self.offset_net(queries)                    # (B, Nq, num_points*2)
        offsets = offsets.view(B, Nq, self.num_points, 2)    # (B, Nq, K_s, 2)

        # Add reference points (broadcast across batch)
        ref = self.ref_points.unsqueeze(0).unsqueeze(2)      # (1, Nq, 1, 2)
        sample_locs = ref + offsets * 0.1                    # scale offsets to ±10% of grid
        # Clamp to valid [0,1] range
        sample_locs = sample_locs.clamp(0.0, 1.0)
        # Convert to [-1,1] for grid_sample (expects (x, y) = (W, H) ordering)
        # spatial_grid is (B, D, H, W); grid_sample needs grid of shape (B, H_out, W_out, 2)
        # We reshape: sample_locs → (B, Nq*K_s, 1, 2) → grid_sample outputs (B, D, Nq*K_s, 1)
        grid = sample_locs.view(B, Nq * self.num_points, 1, 2) * 2.0 - 1.0  # [-1,1]
        # grid_sample expects grid in (x, y) order: x=W, y=H
        # our coords are (u, v) where u→W, v→H → already correct as (x, y) for grid_sample
        sampled = torch.nn.functional.grid_sample(
            spatial_grid, grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )  # (B, D, Nq*K_s, 1)
        sampled = sampled.squeeze(-1).transpose(1, 2)       # (B, Nq*K_s, D)
        sampled = sampled.view(B, Nq, self.num_points, D)   # (B, Nq, K_s, D)
        return sampled

    def forward(
        self,
        queries: torch.Tensor,      # (B, num_queries, embed_dim)
        spatial_grid: torch.Tensor, # (B, embed_dim, H', W') — NOT flattened
    ) -> torch.Tensor:
        # Self-attention (unchanged)
        q = self.norm1(queries)
        q2 = self.self_attn(q, q, q)[0]
        queries = queries + self.dropout1(q2)

        # Deformable cross-attention
        q = self.norm2(queries)
        sampled = self._sample_spatial_features(q, spatial_grid)
        # sampled: (B, Nq, K_s, embed_dim)

        # Per-query attention weights over K_s sampled features
        attn_w = self.attn_weight_net(q)                    # (B, Nq, K_s)
        attn_w = attn_w.softmax(dim=-1).unsqueeze(-1)       # (B, Nq, K_s, 1)

        # Project sampled values
        values = self.value_proj(sampled)                   # (B, Nq, K_s, D)

        # Weighted sum: (B, Nq, D)
        attended = (attn_w * values).sum(dim=2)             # (B, Nq, D)
        attended = self.out_proj(attended)                  # (B, Nq, D)

        queries = queries + self.dropout2(attended)

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries
```

#### `Pose3dTransformerHead.__init__` changes

```python
# New constructor kwargs:
#   deform_num_points: int = 0       (0 = disabled, use standard cross-attn)
#   deform_hidden_dim: int = 64      (bottleneck in offset network)
#   num_body_queries: int = 70       (22 for Design B/C)
#   num_decoder_layers: int = 1      (2 for Design C)
#   hand_aux_loss_weight: float = 0.0
#   aux_body_loss_weight: float = 0.0

self.use_deform = deform_num_points > 0
if self.use_deform:
    self.decoder_layer = _DeformableDecoderLayer(
        hidden_dim, num_heads, dropout,
        num_points=deform_num_points,
        deform_hidden_dim=deform_hidden_dim,
        num_queries=num_body_queries)
else:
    self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)

# For multi-layer designs:
if num_decoder_layers > 1:
    self.decoder_layers = nn.ModuleList([
        _DeformableDecoderLayer(hidden_dim, num_heads, dropout,
                                num_points=deform_num_points,
                                deform_hidden_dim=deform_hidden_dim,
                                num_queries=num_body_queries)
        for _ in range(num_decoder_layers)
    ])
```

#### `forward()` changes

The key change: for deformable designs, do NOT flatten the spatial feature into (B, 960, D). Instead, keep it as a 2D grid (B, D, H', W') after `input_proj`:

```python
feat = feats[-1]          # (B, C, H, W)
B, C, H, W = feat.shape

if self.use_deform:
    # Project backbone features to hidden_dim, keep 2D grid
    spatial_flat = feat.flatten(2).transpose(1, 2)   # (B, H*W, C)
    spatial_proj = self.input_proj(spatial_flat)      # (B, H*W, hidden_dim)
    pos_enc = self._get_pos_enc(H, W, feat.device)
    spatial_proj = spatial_proj + pos_enc             # (B, H*W, hidden_dim)
    # Reshape back to 2D grid for grid_sample
    spatial_grid = spatial_proj.transpose(1, 2).view(B, self.hidden_dim, H, W)
    # (B, hidden_dim, H', W')

    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
    intermediate_outputs = []
    for layer in self.decoder_layers:
        queries = layer(queries, spatial_grid)
        intermediate_outputs.append(queries)
else:
    # Standard baseline path
    spatial = feat.flatten(2).transpose(1, 2)
    spatial = self.input_proj(spatial) + self._get_pos_enc(H, W, feat.device)
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
    intermediate_outputs = []
    decoded = self.decoder_layer(queries, spatial)
    intermediate_outputs.append(decoded)
    queries = decoded
```

#### `_init_head_weights()` changes

For the offset network: initialise the final linear output near-zero so that at step 0, all offsets are ≈ 0:
```python
if self.use_deform:
    for layer_mod in self.decoder_layers:
        nn.init.zeros_(layer_mod.offset_net[-1].weight)
        nn.init.zeros_(layer_mod.offset_net[-1].bias)
        nn.init.zeros_(layer_mod.attn_weight_net.weight)
        nn.init.zeros_(layer_mod.attn_weight_net.bias)
        nn.init.trunc_normal_(layer_mod.value_proj.weight, std=0.02)
        nn.init.trunc_normal_(layer_mod.out_proj.weight, std=0.02)
```
With zero offset and zero attention-weight init, all K_s sampled features are identical (all at grid centre), and `attn_w` is uniform (1/K_s each) → `attended = value_proj(sampled_centre)` — a linear transform of the centre feature, equivalent to cross-attending to a single global average pool at init. This gives a stable warm-start.

### `config.py`

Add to head kwargs (all literals, no imports):

**Design A:**
```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=1024,
    hidden_dim=256,
    num_joints=70,
    num_heads=8,
    dropout=0.1,
    deform_num_points=8,
    deform_hidden_dim=64,
    num_body_queries=70,
    num_decoder_layers=1,
    hand_aux_loss_weight=0.0,
    aux_body_loss_weight=0.0,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
)
```

**Design B:** `num_body_queries=22`, `hand_aux_loss_weight=0.1`

**Design C:** `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`

All values are int/float/str literals. No Python import statements. Fully compliant with MMEngine config constraints.

---

## Expected Outcome

- **Primary gain — body MPJPE**: per-query deformable sampling gives each joint query a learned spatial footprint aligned to its anatomical region. This is the strongest attention-level structural prior proposed so far. Target: `mpjpe_body_val < 180` at stage-1 (best prior: 183.16 — idea002/design003 stage-1), `< 160` at stage-2.
- **Secondary gain — mpjpe_rel_val**: relative MPJPE benefits from joint queries attending to consistent, body-relevant spatial tokens. Following idea008's pattern (333mm stage-2), deformable sampling should further improve relative structure. Target: `mpjpe_rel_val < 380` at stage-2 (best prior: 333mm — idea008/design002; deformable is additive on top).
- **Design A (deformable, 70 queries, 1 layer)**: the diagnostic test. Does learned sparse sampling alone beat the baseline? Expected composite_val < 340 at stage-1.
- **Design B (deformable + 22 body queries, 1 layer)**: combines spatial query isolation (idea008) with per-query attention footprint. Expected composite_val < 330 at stage-1 (best prior: 328.14 — idea013/design003), < 220 at stage-2.
- **Design C (deformable + 22 body queries + 2 layers)**: the primary high-potential bet. VRAM savings from 22 queries × K_s=8 sampling (vs. 70 queries × 960 tokens) are massive, enabling 2 decoder layers with progressive refinement. Expected composite_val < 320 at stage-1, < 210 at stage-2 (improving on best prior 224.52 — idea001/design001).
- **Pelvis MPJPE**: the pelvis token (query 0) learns its own reference point and offset. Over training, it can specialise to attend to depth-informative spatial tokens (body silhouette boundaries, depth gradient regions). This provides a natural depth-signal concentration that standard dense cross-attention lacks. Expected mild improvement.

---

## Risk and Mitigation

- **`grid_sample` AMP compatibility**: `torch.nn.functional.grid_sample` with `mode='bilinear'` is AMP-compatible (all ops float16-safe). However, if the input features are in float16 and the grid coordinates are float32, an explicit cast `grid = grid.to(spatial_grid.dtype)` may be needed before `grid_sample`. The Designer should add this cast to prevent dtype mismatch errors under AMP.

- **Near-zero offset init produces degenerate sampling**: at init, all K_s offsets ≈ 0, so all K_s sampled features are at the same grid location (centre). This means `value_proj(sampled_features)` produces K_s identical vectors, and the weighted sum (with attn_w initialised to uniform 1/K_s) is equivalent to `value_proj(spatial_centre)`. The gradient from the body joint loss flows back through `grid_sample` to the `ref_points` and `offset_net` parameters, pushing them toward the body region. Convergence is expected in the first 3–5 epochs as reference points migrate toward anatomically relevant grid locations. If convergence is slow, the Designer may increase the offset scaling from 0.1 to 0.2 (larger exploration around reference points).

- **`grid_sample` gradient numerical stability**: `grid_sample` gradients can become large when sampling near grid boundaries (`border` padding_mode is used to clamp without discontinuities). The existing `clip_grad=dict(max_norm=1.0)` in the config provides the necessary safety net.

- **Reference point initialisation at (0.5, 0.5)**: all joint queries start with the same reference point (grid centre). This is intentional — the offset network learns per-query specialization during training. Alternative: initialise reference points at anatomically plausible 2D locations on the 24×40 grid (e.g., the head at top-centre, feet at bottom). However, this requires knowledge of BEDLAM2's crop geometry and is a Designer-level tuning choice. The uniform centre init is safe and gradient-driven specialization should occur naturally within 20 epochs.

- **`num_queries` must match `joint_queries.weight.shape[0]`**: `_DeformableDecoderLayer.__init__` takes `num_queries` as a constructor argument, and `ref_points` is an `nn.Parameter` of shape `(num_queries, 2)`. The Designer must ensure this matches `num_body_queries` from the head config. A simple assert in `_DeformableDecoderLayer.__init__` can catch mismatches early.

- **`decoder_layers` vs. `decoder_layer` naming**: the baseline uses `self.decoder_layer` (singular). For deformable designs, use `self.decoder_layers = nn.ModuleList(...)` for all designs including single-layer to avoid conditional logic. The Designer can keep `self.decoder_layer = self.decoder_layers[0]` as an alias for backward compatibility with any code that references the singular attribute, but `forward()` should always iterate over `decoder_layers`.

- **Interaction with idea017 (body-focused multi-layer decoder)**: Design C of this idea (deformable + 22 queries + 2 layers) is the compositional upgrade of idea017's Design A (standard cross-attn + 22 queries + 2 layers). If idea017 is training when this idea is submitted, the Orchestrator should schedule idea019 after idea017 completes to avoid redundant overlap in the training queue. The two ideas are orthogonal in mechanism (cross-attention implementation) so direct comparison is scientifically useful once results are available.

- **Interaction with idea018 (depth-gated cross-attention)**: idea018 adds a depth gate to *dense* cross-attention logits. Deformable sampling in idea019 replaces the dense attention entirely with per-query sparse sampling. These two ideas are mutually exclusive in their cross-attention implementation — do not combine them directly. A future idea could explore depth-gated reference point initialization (using depth information to warm-start reference points toward body-depth regions), but this is beyond scope here.

- **Memory feasibility (Design C)**: with AMP, batch=4, hidden_dim=256, num_heads=8, num_body_queries=22, K_s=8:
  - Self-attention: (B, 8_heads, 22, 22) × 2 layers = small
  - Deformable cross-attention: K_s=8 sampled features per query → (B, 22, 8, 256) sampled tensor × 2 layers
  - FFN: (B, 22, 1024) intermediate × 2 layers
  - Total estimate well within 2080 Ti 10.57 GB budget.
  No OOM risk expected.

- **MMEngine config constraint**: `deform_num_points`, `deform_hidden_dim`, `num_body_queries`, `num_decoder_layers` are int literals; `hand_aux_loss_weight`, `aux_body_loss_weight` are float literals. No Python import statements required. Fully compliant.

- **Eval/inference compatibility**: `predict()` calls `self.forward(feats)`, which now runs deformable layers. Output shapes `(B, 70, 3)` joints, `(B, 1)` depth, `(B, 2)` UV are unchanged. `BedlamMPJPEMetric`, `TrainMPJPEAveragingHook`, and `MetricsCSVHook` see identical interfaces. No downstream changes.
