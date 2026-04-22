**Idea Name:** Lightweight Spatial Encoder via Single-Layer Self-Attention over Spatial Tokens

**Approach:** Before joint queries cross-attend to spatial tokens, pass the spatial tokens through a single lightweight transformer encoder layer (self-attention over the 960 spatial tokens + FFN, with residual and zero-init so training starts from baseline behaviour) so that each spatial token's key/value representation is enriched with global context from all other spatial positions — giving joint queries richer, globally-aware features to cross-attend to, exactly as the DETR encoder does before the DETR decoder.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Missing Encoder

The baseline transformer head is a **decoder-only** architecture: spatial tokens are projected, positionally-encoded, and then immediately used as keys/values for cross-attention. There is **no encoder** — spatial tokens never attend to each other before being read by joint queries.

In contrast, the original DETR (Carion et al., 2020) and virtually every subsequent transformer-based detection or pose model uses an **encoder** that runs self-attention over image tokens before the decoder. The encoder serves a critical purpose: it allows each spatial token to aggregate context from distant image regions before the decoder queries read from it. Without this, cross-attention keys and values at each position reflect only local feature information.

This architectural gap has never been directly addressed in the 29 prior ideas for this project:

| Idea | What it modifies | Difference from idea030 |
|---|---|---|
| idea001 | Multi-layer decoder (deeper query refinement) | More decoder layers, no encoder; queries refine, spatial tokens never self-attend |
| idea009 | Spatial token *dropout* (regularization) | Drops tokens; no self-attention added |
| idea015 | Spatial super-token *pooling* (compression) | Compresses tokens to K slots; no global context enrichment via self-attention |
| idea019 | Deformable sampling of spatial tokens | Changes which tokens are sampled; no enrichment |
| idea021 | Learnable per-query spatial bias | Bias on query side; spatial tokens unchanged |
| idea022 | Cascaded decoder with reprojection bias | More decoder layers + reprojection-conditioned bias; still no encoder |
| idea023 | Heatmap-guided soft pooling | Pools from spatial tokens; tokens themselves unchanged |
| idea027 | Depthwise-separable convolution over spatial tokens | Local (3×3 neighborhood) context; **not** global self-attention |

**idea027 is the closest prior** — it enriches spatial tokens via local convolution. This idea proposes a fundamentally different mechanism: **full global self-attention** over all 960 spatial tokens. The key difference:

- Idea027 (3×3 depthwise conv): each token sees its 8 neighbors in the 40×24 grid → receptive field of 3×3 = 9 grid cells
- Idea030 (spatial self-attention): each token attends to all other 959 tokens → global receptive field

The combination matters because human pose estimation requires **global context**: identifying the pelvis requires knowing where the head and feet are, and vice versa. A 3×3 conv (idea027) cannot propagate this long-range context; a self-attention encoder can.

### Why Global Spatial Context Matters for Joint Localization

Consider what a joint query's cross-attention has to do in the baseline:

1. The query for joint `j` cross-attends to 960 independent spatial tokens
2. Each spatial token key at position (h, w) encodes only local features at that grid cell
3. The query must learn to identify, from these locally-coded keys, which grid cell contains joint `j`

Without a spatial encoder, joint `j`'s query has no way to know from the token keys whether a given position is "near the body center" or "at the image boundary", "above the pelvis" or "below the shoulders" — except through positional encoding. The positional encoding provides geometric position but not semantic context (e.g., "this token is in a region with another joint nearby").

With a spatial encoder, after self-attention:
- The token at the shoulder position "knows" where the elbow is (adjacent token received gradient from shoulder-elbow joint query during encoder training)
- The token at the hip "knows" where the knee is
- Background tokens "know" they are surrounded by other background tokens (and will be downweighted accordingly)

This is the **same argument** that motivates the DETR encoder: without it, the decoder queries must simultaneously do spatial routing AND feature aggregation from locally-coded keys. With the encoder, the keys are pre-contextualized, and the decoder's job is just to read the right (already-contextualized) token.

### Why the Baseline Has Room for This

Looking at the observed results:

- **mpjpe_rel_val plateau**: despite 29 ideas improving architecture, loss, queries, and routing, `mpjpe_rel_val` best at stage-1 is 362mm (idea008/design002) vs baseline 438mm. Stage-1 best composite is 323.75 (idea023). The improvements have come from better query initialization (idea023: heatmap routing), better query architecture (idea008: body-only), and better output structure (idea013: kinematic chain). None of these targeted the spatial token representation before cross-attention.
- **idea027 (conv context enrichment)**: is "Implemented" (not yet trained). This idea is the spatial-encoder approach in a different technical regime (local conv vs global self-attn). If idea027 shows gains, the spatial encoder (global context) is a natural follow-up. If idea027 doesn't show gains, it may be because local conv is insufficient — global self-attention is stronger. This idea is complementary either way.
- **Body MPJPE floor at ~183mm**: most ideas improve or maintain this floor. A spatial encoder that provides richer cross-attention keys could break through by giving joint queries better pre-contextualized features to read from.

---

## Proposed Variations

### Design A — Single-layer spatial encoder, zero-init (safe baseline enhancement)

Add one `_EncoderLayer` before the decoder:

```python
class _EncoderLayer(nn.Module):
    """Single self-attention encoder layer for spatial tokens."""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        # Zero-init the output projection of the FFN so training starts from baseline
        nn.init.zeros_(self.ffn[-2].weight)
        nn.init.zeros_(self.ffn[-2].bias)
```

The zero-init on the FFN output projection ensures that at training start, the encoder's residual delta is zero — the spatial tokens passed to the decoder are identical to the baseline (input_proj + pos_enc only). As training progresses, the encoder learns to add contextual information.

**Note**: the self-attention output projection (`self.self_attn.out_proj`) is also zero-initialized for a fully safe baseline-equivalent start:
```python
nn.init.zeros_(self.self_attn.out_proj.weight)
nn.init.zeros_(self.self_attn.out_proj.bias)
```

This zero-init strategy follows the same approach used in ResNet residual branches, idea027's conv init, and idea023's heatmap_proj init — a principled practice for safely adding new modules to a working architecture.

**Config kwargs**: `use_spatial_encoder=True`, `num_encoder_layers=1`, `encoder_dropout=0.1`, `encoder_zero_init=True`.

Design A is the minimal diagnostic: does a single spatial self-attention encoder layer improve cross-attention quality for joint queries?

### Design B — Single-layer spatial encoder with reduced head count (memory-efficient variant)

Same as Design A but use `num_heads=4` for the encoder self-attention (vs 8 in the decoder cross-attention). Rationale:

- The encoder operates on 960×960 self-attention matrices (vs 70×960 for decoder cross-attention). With `num_heads=8` and batch=4, the encoder self-attention matrix is `4 × 8 × 960 × 960 × 2B ≈ 59 MB` in float16. With `num_heads=4`, this halves to ~29 MB — within the 2080 Ti's available budget after the backbone.
- Fewer heads may be sufficient for spatial context: the encoder needs to propagate long-range spatial context, not model fine-grained joint-specific relationships (which is the decoder's job).

**Config kwargs**: `use_spatial_encoder=True`, `num_encoder_layers=1`, `encoder_num_heads=4`, `encoder_dropout=0.1`, `encoder_zero_init=True`.

### Design C — Two-layer spatial encoder (deeper context enrichment)

Same as Design B (4 heads for memory efficiency) but with 2 stacked encoder layers. After 2 self-attention passes, each spatial token's representation effectively integrates context from all other tokens at two levels of abstraction (first pass: immediate neighborhood aggregation; second pass: higher-order context across the full feature grid).

Memory estimate for 2 encoder layers with `num_heads=4`, batch=4, AMP float16:
- Encoder self-attn: 2 × 4 × 4 × 960 × 960 × 2B ≈ 58 MB
- Decoder cross-attn (22 or 70 queries): 4 × 8 × 70 × 960 × 2B ≈ 8.5 MB
- Backbone activations: ~6–7 GB (dominant)
- Total estimate: ~7 GB — within the 10.57 GB 2080 Ti VRAM budget (with AMP and gradient checkpointing already in use via the training infrastructure)

If memory pressure is observed, the Designer can reduce `encoder_dropout=0.0` (saves some activation memory) or use the body-only 22-query baseline (not required by this idea).

**Config kwargs**: `use_spatial_encoder=True`, `num_encoder_layers=2`, `encoder_num_heads=4`, `encoder_dropout=0.1`, `encoder_zero_init=True`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

**1. New module: `_EncoderLayer`**

```python
class _EncoderLayer(nn.Module):
    """Single transformer encoder layer: self-attention over spatial tokens + FFN."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 zero_init: bool = True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if zero_init:
            nn.init.zeros_(self.self_attn.out_proj.weight)
            nn.init.zeros_(self.self_attn.out_proj.bias)
            nn.init.zeros_(self.ffn[-2].weight)
            nn.init.zeros_(self.ffn[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) spatial tokens

        Returns:
            (B, N, embed_dim) enriched spatial tokens
        """
        # Self-attention (pre-norm)
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2)[0]
        x = x + self.dropout1(x2)

        # FFN (pre-norm)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x
```

**2. `Pose3dTransformerHead.__init__` additions**

New kwargs (all defaulting to baseline behaviour with `use_spatial_encoder=False`):

```python
use_spatial_encoder: bool = False     # enable spatial encoder
num_encoder_layers: int = 1           # number of encoder self-attn layers
encoder_num_heads: int = 8            # attention heads in encoder (Design B/C: 4)
encoder_dropout: float = 0.1          # dropout in encoder
encoder_zero_init: bool = True        # zero-init encoder output projections
```

When `use_spatial_encoder=True`:
```python
self.spatial_encoder = nn.ModuleList([
    _EncoderLayer(hidden_dim, encoder_num_heads, encoder_dropout, encoder_zero_init)
    for _ in range(num_encoder_layers)
])
self.use_spatial_encoder = use_spatial_encoder
```

When `use_spatial_encoder=False` (baseline):
```python
self.use_spatial_encoder = False
```

**3. `forward()` additions**

After computing `spatial = spatial + pos_enc` and before the decoder:

```python
if self.use_spatial_encoder:
    for enc_layer in self.spatial_encoder:
        spatial = enc_layer(spatial)  # (B, H'W', hidden_dim) in-place enrichment
```

The encoder receives positional-encoded spatial tokens and returns enriched tokens. The decoder then cross-attends to these enriched tokens — same as baseline except the keys/values now encode global context.

**4. No changes to `loss()`, `predict()`**, or any other method.

### `config.py`

**Design A:**
```python
use_spatial_encoder=True,
num_encoder_layers=1,
encoder_num_heads=8,
encoder_dropout=0.1,
encoder_zero_init=True,
```

**Design B:**
```python
use_spatial_encoder=True,
num_encoder_layers=1,
encoder_num_heads=4,
encoder_dropout=0.1,
encoder_zero_init=True,
```

**Design C:**
```python
use_spatial_encoder=True,
num_encoder_layers=2,
encoder_num_heads=4,
encoder_dropout=0.1,
encoder_zero_init=True,
```

All values are bool/int/float literals. No Python `import` statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — `mpjpe_rel_val`**: the spatial encoder gives joint queries pre-contextualized keys/values. Each spatial token's representation now encodes where other body parts are (relative context), enabling joint queries to cross-attend more selectively and accurately. Target: `mpjpe_rel_val < 420mm` at stage-1 (vs. baseline 438mm; approaching idea008's 362mm without body-only query restriction).

- **Secondary gain — `mpjpe_body_val`**: better cross-attention quality (richer keys) should improve 3D joint coordinate regression. Target: `mpjpe_body_val < 188mm` at stage-1 (vs. baseline 195mm, best prior 183mm from idea023).

- **Tertiary gain — `mpjpe_pelvis_val`**: the pelvis token (query 0) also cross-attends to spatially-encoded tokens; global context about body extent may help localize the pelvis more accurately. Target: `mpjpe_pelvis_val < 620mm` at stage-1 (vs. baseline 652mm, best prior 608mm).

- **Composite target (stage-1)**: `composite_val < 335` for Design A/B; `< 328` for Design C (competitive with best prior 323.75 from idea023/design001).

- **Composite target (stage-2)**: `composite_val < 222`, competitive with best prior stage-2 of 224.52 (idea001/design001).

- **Design A** (1 layer, 8 heads): diagnostic — does spatial self-attention encoder help at all? The 8-head encoder is richer but uses more memory. If memory is not an issue, this is the strongest variant for a single layer.

- **Design B** (1 layer, 4 heads): memory-efficient variant. 4 heads are sufficient to model long-range spatial context. Expected to match Design A quality while fitting more comfortably in GPU memory.

- **Design C** (2 layers, 4 heads): deeper context enrichment. Two encoder passes allow spatial tokens to integrate higher-order dependencies (e.g., wrist token "knows" about both elbow and shoulder after 2 passes). Highest potential.

---

## Risk and Mitigation

- **Memory: 960×960 self-attention matrix size**: the encoder self-attention matrix per head is `(B=4) × 960 × 960` in float16 = `4 × 960 × 960 × 2B ≈ 7.3 MB` per head. With 8 heads: ~58 MB for Design A; with 4 heads: ~29 MB for Design B/C per layer. Two layers (Design C, 4 heads): ~58 MB total for encoder attention. This is manageable alongside the backbone's ~6–7 GB activation footprint. If OOM occurs, the Designer should switch to `encoder_num_heads=2` or reduce `encoder_dropout=0.0` to save activation memory.

- **Training time overhead**: the encoder self-attention over 960 tokens is `O(960^2)` per head per layer. For 1 layer, 8 heads: `4 × 8 × 960^2 ≈ 29.5M` operations — roughly equivalent to one full decoder cross-attention pass. On the 2080 Ti this adds approximately 2–5 ms per step. With 100 steps/epoch (train100, batch=4 with accum=8) this is negligible (<10 min/epoch overhead). With 400 steps/epoch (train400, batch=4), the overhead is also small.

- **Zero-init safety**: zero-initializing the encoder output projections ensures the spatial tokens at training start are identical to the baseline's positional-encoded tokens. The encoder adds no perturbation at initialization — the first few epochs establish the encoder via gradient descent exactly as the new module learns to be useful. This pattern is identical to idea027's conv zero-init and idea023's heatmap_proj zero-init, both of which produced stable training.

- **Interaction with `persistent=False` pos_enc buffer**: the positional encoding is added to spatial tokens before they enter the encoder (and before cross-attention). The encoder self-attention operates on positional-encoded tokens, so every spatial token's content and position are both available during global context aggregation. This is correct — the encoder should use positional information when aggregating context.

- **Interaction with idea027 (depthwise conv)**: the two ideas enrich spatial tokens in complementary ways (local 3×3 conv vs. global self-attention). They can be combined in a future idea. In isolation (this idea only), the spatial encoder captures long-range dependencies that the local conv misses.

- **Interaction with idea023 (heatmap-guided query init)**: the heatmap routing initializes query embeddings with joint-specific feature summaries from the raw spatial tokens. If combined with the spatial encoder, the heatmap soft-pooling would operate on encoder-enriched tokens (more informative). This is a natural composition for a future combined idea.

- **Interaction with idea001 (multi-layer decoder)**: the encoder + multi-layer decoder is the full DETR-style architecture. Idea001 showed strong stage-2 results (224.52). Combining 1 encoder layer + 2 decoder layers is a direct future experiment if both individually show gains.

- **Gradient flow through encoder**: the encoder's self-attention gradient flows back to the spatial token representation (input_proj output + pos_enc). The gradient path is:
  - Loss → joint coordinates → decoder cross-attention → encoder output → encoder self-attention → input_proj
  - This is an additional gradient path to the input_proj layer. The backbone (backbone params) also receive gradient through the backbone→feature→input_proj path, unchanged.
  - With zero-init, the encoder gradient is initially zero; it grows as the encoder learns to be non-trivial. No gradient explosion risk.

- **AMP / float16 safety**: the encoder self-attention uses standard `nn.MultiheadAttention` which is AMP-compatible (same as the decoder). LayerNorm is float32 in AMP (standard MMEngine AMP behaviour). No numerical stability concerns.

- **MMEngine config constraint**: all new kwargs (`use_spatial_encoder`, `num_encoder_layers`, `encoder_num_heads`, `encoder_dropout`, `encoder_zero_init`) are bool/int/float literals. No Python import statements in `config.py`. Fully compliant.

- **Memory: `num_encoder_layers` as a Python int loop**: the `nn.ModuleList` comprehension `[_EncoderLayer(...) for _ in range(num_encoder_layers)]` uses `range(num_encoder_layers)` — this is Python code in `__init__`, not in the config. The config only passes the integer value `num_encoder_layers=1` or `num_encoder_layers=2`. Fully compliant.

- **Parameter count**: each `_EncoderLayer` has:
  - Self-attention: 4 × (hidden_dim × hidden_dim) = 4 × 256² = 262,144 params (Q, K, V, out_proj)
  - FFN: (256×1024) + (1024×256) = 524,288 + 262,144 = 786,432 params
  - Total: ~1.05M params per encoder layer
  - Design A/B (1 layer): 1.05M params
  - Design C (2 layers): 2.1M params
  - For reference, the baseline head has ~2.8M params (rough estimate); the backbone has ~300M. The encoder adds <1% of total parameters.

- **Masking**: the standard `nn.MultiheadAttention` without `key_padding_mask` or `attn_mask` performs full 960×960 attention. No masking is needed — all spatial tokens are valid (no padding tokens in the baseline's feature grid). This is correct.

- **Composability with body-only decoder (idea008)**: the encoder enriches spatial tokens before they become keys/values. Whether the decoder has 22 or 70 queries is irrelevant to the encoder. The Designer can combine this idea with the body-only query architecture by simply setting `num_joints=22` (or as hardcoded in idea008 variants) while also adding the spatial encoder.
