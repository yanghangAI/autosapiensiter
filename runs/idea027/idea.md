**Idea Name:** Spatial Token Context Enrichment via Depthwise Separable Convolution

**Approach:** Before joint queries cross-attend to spatial tokens, reshape the 960 projected spatial tokens back to the 40×24 feature grid and apply a lightweight depthwise-separable 2D convolution (3×3 depthwise + 1×1 pointwise, residual connection, zero-initialized so training starts from baseline) to give each spatial token local 2D context from its immediate neighborhood — so that cross-attention keys and values encode spatially-coherent local structures (limb segments, body contours) rather than independent per-position feature vectors.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Spatial Token Independence Problem

In the baseline, after the backbone feature map is projected and positional-encoded, each of the 960 spatial tokens is an independent feature vector. When joint queries cross-attend to these tokens, each query Q_j selects from K=960 independently-encoded keys. The key for spatial position (h, w) encodes only the features at that exact location — it has **zero knowledge of what is happening at neighboring positions (h±1, w±1)**.

This is a structural limitation. Consider what a joint query needs to find in the spatial token grid:

- A **limb** (e.g., forearm) extends across multiple spatial grid cells. Its local evidence is distributed across several adjacent tokens, not concentrated at a single point.
- A **body contour** (e.g., shoulder-to-torso boundary) is a spatial gradient across neighboring cells — impossible to capture from a single token's isolated features.
- **Depth edges** (transitions in the depth channel, now fused into the backbone via RGBD) are inherently local structures spanning 2-3 adjacent tokens.

With independent token representations, a query searching for the elbow must rely on cross-attention to simultaneously identify and aggregate the relevant evidence from multiple adjacent tokens — a hard task for a single linear attention layer. The attention over 960 tokens is noisy because no individual token is "rich enough" to attract decisive attention.

### What Prior Ideas Have Done

Every prior idea has modified the **query side** or the **loss/supervision side**:

| Category | Ideas |
|---|---|
| Query count / content | idea001, 002, 003, 008, 017, 023 |
| Query attention routing | idea006, 007, 019, 020, 021, 022 |
| Loss / supervision | idea005, 010, 012, 013, 024, 025, 026 |
| Output parameterization | idea013 |
| Spatial token count / pooling | idea015 (compresses to fewer super-tokens) |
| Spatial token dropout | idea009 (drops tokens randomly) |
| Depth-aware token features | idea004, 016, 018 |

**No prior idea has processed spatial tokens to incorporate 2D neighborhood context before cross-attention.** idea004 and idea016 modify individual token features using depth channel information, but still produce independent-per-token representations. idea015 reduces token count by pooling, which is the opposite direction. No idea applies a spatial convolution to propagate information between neighboring tokens.

### The Depthwise-Separable Convolution Approach

After `spatial = input_proj(feat.flatten(2).transpose(1,2)) + pos_enc`, the spatial tokens have shape `(B, 960, hidden_dim)`. We can reshape to `(B, hidden_dim, 40, 24)`, apply a lightweight 2D convolution, and reshape back.

**Architecture of the spatial enrichment module:**

```
SpatialContextNet(hidden_dim):
    depthwise: Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False)
    pointwise: Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
    norm:      GroupNorm(num_groups=32, num_channels=hidden_dim)
    act:       GELU
    residual:  spatial_enriched = spatial + net(spatial_reshaped)
```

The residual connection is critical: **zero-initializing the pointwise convolution weight and bias ensures the module outputs exactly zero at training start**, so the enriched spatial tokens are identical to the baseline spatial tokens at initialization. Training can only improve from this point.

**Zero-initialization strategy:**
```python
nn.init.zeros_(self.spatial_ctx_pw.weight)
nn.init.zeros_(self.spatial_ctx_pw.bias)
```
With zeros in the pointwise conv, `net(spatial_reshaped) = 0` for all inputs at init → `spatial_enriched = spatial + 0 = spatial`. The model starts at exactly baseline behaviour and learns to use spatial context when it is beneficial.

**The depthwise convolution** (`groups=hidden_dim`) applies an independent 3×3 spatial filter to each of the 256 feature channels, propagating context from the 8 neighbors of each grid cell. It has only `hidden_dim × 3 × 3 = 256 × 9 = 2304` parameters — negligible.

**The pointwise convolution** mixes the spatially-aggregated per-channel features: `hidden_dim × hidden_dim = 256 × 256 = 65536` parameters — same order as a single attention layer weight, but applied once globally rather than per-query.

**GroupNorm + GELU**: a single GroupNorm normalizes the convolved features before the pointwise mixing, preventing scale explosion. GELU introduces a mild nonlinearity consistent with the transformer's FFN blocks.

### Why This Will Help

**1. Richer cross-attention keys/values.** After enrichment, each token's key/value encodes not only its own backbone feature but also a weighted summary of its 8 spatial neighbors. A joint query searching for an elbow region will find that the most relevant key is a spatially-aggregated representation of the forearm region — more discriminative than a single-point feature.

**2. Reduced cross-attention noise.** With 960 independent tokens, a single-layer cross-attention must allocate its attention mass across many similar-looking background tokens and only a few relevant foreground tokens. After spatial context propagation, body region tokens become more distinctive (their features now reflect a consistent body-structure context), making the relevant tokens more "peaky" and easier for queries to find.

**3. Depth-structure coherence.** The RGBD backbone already produces spatially-structured features that encode depth information. Applying a spatial convolution on these features explicitly captures depth gradient structure (depth edges, depth planes) that spans adjacent spatial positions — relevant for pelvis depth estimation where the depth boundary of the torso region spans multiple tokens.

**4. No additional cross-attention overhead.** The enrichment is computed once before any decoder layer, producing the same 960 tokens. Cross-attention cost is unchanged. The conv cost is O(H'×W'×hidden_dim²) per forward pass — trivial compared to the backbone.

### Grounding in Observed Results

- **idea023/design001** (heatmap-guided query init): best stage-1 at 323.75, body MPJPE 183.4mm. This works by routing each query to a better spatial region using a heatmap. But even with the right spatial routing, the quality of cross-attention is limited by the independence of spatial tokens. A query that correctly focuses on the elbow region still only aggregates features from isolated token positions.

- **Stage-1 body MPJPE floor at ~183mm** across all 26 ideas: the floor is consistent regardless of query sophistication. The shared limitation is the spatial token side: no matter how well a query is initialized or routed, it cross-attends to the same independent, context-free spatial token representations.

- **idea008/design002** (body-only decoder): best mpjpe_rel_val at 362mm (vs. baseline 438mm) by removing query contamination. Spatial context enrichment would give body queries even better key/value representations to attend to, potentially breaking the rel_val floor further.

- **idea001/design001** (2-layer decoder, best stage-2 at 224.52): multi-layer decoding allows progressive refinement. The second decoder layer's cross-attention would benefit from enriched spatial tokens: at the second layer, the queries already carry partial pose estimates from layer 1, and richer spatial context in the keys/values would help the refinement step zero in on precise joint locations.

- The spatial token enrichment is **composable with every prior idea** — it operates purely on the key/value side of cross-attention and requires no changes to queries, losses, or output heads. It could be combined with idea001 (multi-layer decoder), idea023 (heatmap query init), or idea017 (body-focused multi-layer) in future ideas.

---

## Proposed Variations

### Design A — Depthwise-Separable Conv, No Norm, λ=0 (minimal baseline-equivalent start)

Minimal enrichment module: depthwise 3×3 + pointwise 1×1, **no normalization**, GELU activation, residual. Zero-init on pointwise ensures baseline-equivalent training start.

```python
use_spatial_ctx=True,
spatial_ctx_norm='none',
spatial_ctx_act='gelu',
spatial_ctx_kernel=3,
spatial_ctx_layers=1,
```

No GroupNorm is the safest starting point: avoids the risk that normalization changes the gradient landscape early in training. The model starts exactly at baseline (zero pointwise output) and learns to use context only where it helps.

This design answers the core question: does local 2D spatial context in the cross-attention keys/values improve pose estimation?

### Design B — Depthwise-Separable Conv + GroupNorm, Single Layer

Same as Design A but add GroupNorm(32, hidden_dim) applied to the depthwise-convolved features before the pointwise mixing:

```python
use_spatial_ctx=True,
spatial_ctx_norm='groupnorm',
spatial_ctx_groups=32,
spatial_ctx_act='gelu',
spatial_ctx_kernel=3,
spatial_ctx_layers=1,
```

GroupNorm normalizes across channels within each spatial group, preventing scale drift in the convolved features. This is particularly important for the depthwise component which processes each channel independently (GroupNorm prevents any single channel from dominating the mixed output). The tradeoff: GroupNorm introduces a scale-bias pair (32 groups × 2 params × hidden_dim/32 channels_per_group = hidden_dim × 2 = 512 extra params — negligible).

Zero-init is applied to the pointwise weight and bias, so GroupNorm at init normalizes a zero-variance tensor (all zeros from depthwise conv at init with trivial weight). This is safe because the residual means the model starts at the baseline; GroupNorm only activates once the depthwise conv has learned meaningful filters.

Design B is expected to be the most stable and achieve the best results, as GroupNorm prevents early gradient issues.

### Design C — Two-Layer Depthwise-Separable Conv Stack

Stack two successive enrichment modules (each = depthwise + pointwise + GroupNorm + GELU + residual), allowing a receptive field of 5×5 grid cells (3×3 composed twice, minus 1 on each side). This covers a spatial region of approximately 80×80 pixels at the 40×24 grid scale (each grid cell is 640/40 = 16 pixels), sufficient to capture an entire forearm or lower leg within the receptive field.

```python
use_spatial_ctx=True,
spatial_ctx_norm='groupnorm',
spatial_ctx_groups=32,
spatial_ctx_act='gelu',
spatial_ctx_kernel=3,
spatial_ctx_layers=2,
```

Zero-init only applies to the second layer's pointwise conv — the first layer's pointwise is trunc-normal initialized (std=0.02). This ensures the module starts at baseline (second layer's residual = 0) but allows the first layer to immediately begin learning spatial context from the first gradient step.

The two-layer design is motivated by the fact that a single 3×3 convolution only captures immediate neighbors. For joints that span larger regions (e.g., the spine spanning the entire height of the torso ≈ 20 grid cells), a wider receptive field provides more relevant context.

Parameter cost: 2 × (2304 + 65536) = 135,680 parameters ≈ 0.5% of the backbone — fully within budget.

---

## Implementation Scope

All changes confined to **`pose3d_transformer_head.py`** and **`config.py`**. No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or training infrastructure.

### `pose3d_transformer_head.py`

**1. New `_SpatialContextNet` module class**

```python
class _SpatialContextNet(nn.Module):
    """Lightweight depthwise-separable conv for spatial token context enrichment.

    Applies one (or two) depthwise-separable 2D convolution layers with a
    residual connection to the projected spatial token grid.  The pointwise
    weights are zero-initialized so the module starts as an identity (residual
    output = 0 → spatial tokens unchanged at init).

    Args:
        hidden_dim: Feature dimension (= number of channels in the grid).
        kernel_size: Depthwise kernel size (3 for 3×3 conv).
        num_layers: Number of depthwise-separable blocks to stack.
        norm: 'none' or 'groupnorm'.
        num_groups: Number of groups for GroupNorm (only used if norm='groupnorm').
        act: Activation: 'gelu' or 'relu'.
        zero_init_last: If True, zero-init the last pointwise layer only (for
            multi-layer stacks).
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        norm: str = 'none',
        num_groups: int = 32,
        act: str = 'gelu',
        zero_init_last: bool = True,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            # Depthwise conv (per-channel spatial filtering)
            dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                           padding=kernel_size // 2, groups=hidden_dim, bias=False)
            nn.init.kaiming_normal_(dw.weight, mode='fan_out', nonlinearity='relu')

            # Optional GroupNorm
            if norm == 'groupnorm':
                gn = nn.GroupNorm(num_groups, hidden_dim)
            else:
                gn = nn.Identity()

            # Activation
            act_fn = nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True)

            # Pointwise conv (channel mixing)
            pw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True)
            # Zero-init: last layer always; first layer only if num_layers == 1
            if is_last or (not zero_init_last):
                nn.init.zeros_(pw.weight)
                nn.init.zeros_(pw.bias)
            else:
                nn.init.trunc_normal_(pw.weight, std=0.02)
                nn.init.zeros_(pw.bias)

            layers.extend([dw, gn, act_fn, pw])

        self.net = nn.Sequential(*layers)

    def forward(self, spatial: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            spatial: (B, H*W, hidden_dim) — flattened spatial tokens.
            h, w: Height and width of the spatial grid.

        Returns:
            (B, H*W, hidden_dim) — enriched spatial tokens with residual.
        """
        B, _, D = spatial.shape
        # Reshape to 2D grid: (B, D, H, W)
        x = spatial.transpose(1, 2).reshape(B, D, h, w)
        # Apply spatial context network
        delta = self.net(x)                         # (B, D, H, W)
        delta = delta.reshape(B, D, -1).transpose(1, 2)  # (B, H*W, D)
        return spatial + delta
```

**2. `Pose3dTransformerHead.__init__` additions**

New kwargs with defaults matching baseline (all False/0 = baseline):

```python
use_spatial_ctx: bool = False
spatial_ctx_kernel: int = 3
spatial_ctx_layers: int = 1
spatial_ctx_norm: str = 'none'       # 'none' or 'groupnorm'
spatial_ctx_groups: int = 32
spatial_ctx_act: str = 'gelu'
```

When `use_spatial_ctx=True`:

```python
self.spatial_ctx_net = _SpatialContextNet(
    hidden_dim=hidden_dim,
    kernel_size=spatial_ctx_kernel,
    num_layers=spatial_ctx_layers,
    norm=spatial_ctx_norm,
    num_groups=spatial_ctx_groups,
    act=spatial_ctx_act,
    zero_init_last=True,
)
```

**3. `forward()` additions**

After `spatial = spatial + pos_enc` and before the decoder:

```python
if self.use_spatial_ctx:
    spatial = self.spatial_ctx_net(spatial, H, W)
# (spatial is still (B, H*W, hidden_dim) after enrichment)
```

That is all. The decoder receives the enriched spatial tokens as keys/values. Queries are unchanged.

**4. `loss()` and `predict()`** — no changes needed.

### `config.py`

**Design A:**
```python
use_spatial_ctx=True,
spatial_ctx_kernel=3,
spatial_ctx_layers=1,
spatial_ctx_norm='none',
spatial_ctx_act='gelu',
```

**Design B:**
```python
use_spatial_ctx=True,
spatial_ctx_kernel=3,
spatial_ctx_layers=1,
spatial_ctx_norm='groupnorm',
spatial_ctx_groups=32,
spatial_ctx_act='gelu',
```

**Design C:**
```python
use_spatial_ctx=True,
spatial_ctx_kernel=3,
spatial_ctx_layers=2,
spatial_ctx_norm='groupnorm',
spatial_ctx_groups=32,
spatial_ctx_act='gelu',
```

All values are bool/int/str literals. No Python import statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — mpjpe_body_val**: richer spatial context in cross-attention keys/values allows joint queries to more accurately attend to multi-cell body structures. Target: `mpjpe_body_val < 188` at stage-1, `< 170` at stage-2 (matching best prior of 168.79mm from idea010/design002).

- **Secondary gain — mpjpe_rel_val**: relative MPJPE depends on inter-joint spatial precision. Better spatial context should narrow the gap between adjacent joints' attention regions, improving the relative structure. Target: `mpjpe_rel_val < 420` at stage-1 (vs. baseline 438.7mm).

- **Pelvis MPJPE**: the pelvis token (query 0) cross-attends to enriched spatial tokens that now encode depth-gradient structure around the torso. Expected mild improvement in `mpjpe_pelvis_val` as the depth-encoding spatial tokens become more coherent.

- **Design A** (no norm, 1 layer): minimal intervention — tests whether any spatial context helps. Most conservative; minimal risk of training instability from the absence of normalization. Expected composite_val < 345 at stage-1.

- **Design B** (GroupNorm, 1 layer): normalized spatial context aggregation. The GroupNorm prevents any channel from dominating the mixed features. Expected to outperform Design A by providing more stable gradient flow through the spatial context module. Expected composite_val < 340 at stage-1.

- **Design C** (2 layers, 5×5 effective receptive field): widest spatial context; captures full-limb structures in the enriched representation. Highest potential but also highest risk of adding noise for small, precise joint locations. Expected composite_val < 335 at stage-1 if 2 layers synergize, or potentially worse than Design B if the wider receptive field blurs precise single-joint signals.

- **Composite target (stage-1)**: `composite_val < 330`, competitive with best prior stage-1 of 323.75 (idea023/design001).
- **Composite target (stage-2)**: `composite_val < 220`, competitive with best stage-2 of 224.52 (idea001/design001).

---

## Risk and Mitigation

- **Zero-init baseline equivalence**: the residual `spatial + delta` with zero-init on the last pointwise layer guarantees `delta = 0` at training start for Designs A and B (single-layer). For Design C (two-layer), the second pointwise is zero-initialized → the second layer contributes zero delta, but the first layer is randomly initialized → the first layer can immediately affect the spatial tokens. Mitigation: in Design C, also zero-init the first pointwise's output weight (set `zero_init_last=False` inverts the logic — the Designer should zero-init the final layer as the sole zero-init). Actually: for multi-layer, set the *last* pointwise to zero-init (end of the stack is zero → stack total is zero from baseline perspective). See `zero_init_last=True` in constructor — the Designer should verify that the zero-init is applied to the final pointwise layer only in the multi-layer case.

- **GroupNorm with zero-input at init**: if the depthwise conv output is near-zero at init (due to the kaiming_normal init producing small outputs for near-zero inputs), GroupNorm may encounter near-zero variance across groups. Mitigation: GroupNorm with `eps=1e-5` (PyTorch default) handles near-zero variance gracefully. At init, the groupnorm input is small but not identically zero (kaiming_normal init ≠ zero-init); the normalization is stable.

- **Memory**: the `_SpatialContextNet` for Design B holds 2304 (depthwise) + 65536 (pointwise) + 512 (GroupNorm) = 68,352 parameters ≈ 274 KB at float32. The intermediate tensors are `(B, D, H, W) = (4, 256, 40, 24)` = 983,040 float16 values ≈ 2 MB. Negligible on the 2080 Ti (10 GB). For Design C (2 layers): ~546 KB params, ~4 MB intermediate tensors. Still negligible.

- **Speed**: depthwise conv `(B=4, 256, 40, 24)` with kernel 3×3: approximately `4 × 256 × 40 × 24 × 9 = 22M multiply-adds`. Pointwise conv `(B=4, 256, 40, 24)` with kernel 1×1: `4 × 256 × 256 × 40 × 24 = 2.5G multiply-adds`. Pointwise is O(D²) per spatial position — at `D=256`, this is 256K multiply-adds per spatial position × 960 positions = 246M ops. On a 2080 Ti with ~10 TFLOPS at float16: ≈ 0.025 ms per forward pass. For Design C (2 layers): ~0.05 ms. Negligible compared to the backbone (~1s per iteration).

- **Interaction with idea023 (heatmap-guided query init)**: idea023 computes a soft heatmap from `spatial` tokens and pools per-joint features from them. In idea027, if both were combined, the heatmap would be computed from the enriched spatial tokens rather than raw spatial tokens — which would actually be more informative (the enriched tokens know about their spatial neighborhood). This is a natural future composition.

- **Interaction with idea001 (multi-layer decoder)**: both decoder layers would use the same enriched spatial tokens as keys/values. Each refinement pass benefits from the spatial context. The interaction is purely additive.

- **Interaction with idea008/idea017 (body-focused decoder)**: no interaction — the spatial context enrichment is applied to the token side, which is independent of how many queries are used. The body-focused decoder with 22 queries would benefit equally from enriched tokens as the full 70-query decoder.

- **Feature grid dimensions**: the code uses `H, W = feat.shape[2], feat.shape[3]` from the backbone feature map, and passes `H, W` to `spatial_ctx_net.forward()`. The Designer must ensure these are correct (H=40, W=24 for 640×384 input with stride=16). The `SpatialContextNet.forward()` accepts `h, w` as arguments for flexibility — no hardcoding of grid dimensions.

- **AMP / float16 safety**: Conv2d with float16 inputs is natively supported by PyTorch AMP. GroupNorm is also supported. The zero-init pointwise layer at float16 outputs exact zeros (no float16 precision issue). Safe.

- **MMEngine config constraint**: all new kwargs are bool/int/str literals. No Python import statements in `config.py`. Fully compliant.

- **Output invariance**: the spatial context enrichment only modifies the `spatial` variable inside `forward()`. The output dict `{'joints': ..., 'pelvis_depth': ..., 'pelvis_uv': ...}` is unchanged in shape and semantics. `predict()`, `bedlam_metric.py`, and all hooks see identical interfaces.
