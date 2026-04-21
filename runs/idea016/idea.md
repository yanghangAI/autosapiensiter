**Idea Name:** Depth-Conditional Feature Modulation for Scale-Aware Joint Regression

**Approach:** Insert a FiLM (Feature-wise Linear Modulation) layer between the spatial token projection and the cross-attention in the decoder head: a lightweight sub-network first predicts a scene-scale embedding from the projected spatial tokens, then uses that embedding to produce per-channel affine parameters (γ, β) that rescale and shift the spatial tokens before joint-query cross-attention, making the spatial features explicitly conditioned on the predicted depth scale of the scene.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder projects backbone features to hidden_dim and adds 2D positional encoding, then performs cross-attention between 70 joint queries and these 960 spatial tokens. The spatial tokens contain a mixture of RGB appearance and depth-derived shape information (the backbone processes 4-channel RGB+D input), but the cross-attention treats all 960 tokens identically regardless of the overall depth scale of the scene.

### The Scale Ambiguity Problem

BEDLAM2 samples span a wide range of camera–subject distances. A person's knee joint that is 3 metres away has a very different spatial feature context from the same joint at 8 metres:

- **Pixel area**: closer subjects are larger in the crop; features encode finer joint-level detail.
- **Depth gradient magnitude**: nearby subjects have steeper depth gradients at limb boundaries; distant subjects have shallower gradients.
- **Absolute-to-relative scale**: a 100 mm displacement corresponds to ~2 pixels at 8 m but ~5 pixels at 3 m.

The model must currently infer this scale from the raw feature content — something the backbone can do implicitly (depth channel contains this information), but the head has no explicit mechanism to use. The pelvis depth output (`pelvis_depth`) is ultimately the model's best estimate of this scale, but it is predicted *after* the cross-attention, so the cross-attention itself has no access to the depth-scale signal.

### What FiLM Does

Feature-wise Linear Modulation (Perez et al., 2018, AAAI) provides a generic mechanism to condition a feature representation on an auxiliary signal:

```
FiLM(h; γ, β) = γ ⊙ h + β
```

where `h ∈ R^{hidden_dim}` is a feature vector, and `(γ, β)` are channel-wise affine parameters predicted by a small conditioning network from the auxiliary signal. FiLM has been successfully applied in:

- **Visual question answering** (original FiLM paper): conditioning image features on question embeddings.
- **Multi-domain image translation**: conditioning features on domain labels.
- **Few-shot learning**: conditioning feature extractors on task embeddings.
- **3D detection** (ImVoxelNet and variants): conditioning spatial features on depth priors.

Applied here, the "auxiliary signal" is a **global scale embedding** pooled from the spatial tokens themselves — a compact summary of the depth scale implicit in the backbone's output. This creates a **self-conditioning loop**: the spatial features first produce a scale estimate, which then conditions those same features before they are consumed by cross-attention. This is similar in spirit to squeeze-and-excitation (SE) networks but:

1. Operates on the full hidden-dim channel-wise (SE uses a channel-bottleneck; FiLM is per-channel affine).
2. Pools a *depth-scale* summary rather than a channel saliency summary.
3. Outputs both scale (γ) and shift (β) rather than just scale.

### Why This Is Unexplored

Fourteen completed ideas address: decoder depth (idea001, idea015), pelvis decoupling (idea002), query initialisation (idea003), depth positional encoding (idea004), loss balancing (idea005), skeleton attention bias (idea006), spatial routing (idea007), query reduction (idea008), spatial dropout (idea009), reprojection loss (idea010), iterative refinement (idea011), distance loss (idea012), kinematic parameterization (idea013), anchor pelvis depth (idea014). One idea (idea015) is designed, not yet trained.

**Idea004** (Depth-Aware Spatial Positional Encoding) is the most related: it adds depth values to the *positional* component of spatial tokens. This idea is orthogonal and complementary:

| | idea004 | idea016 |
|---|---|---|
| What is modulated | Positional encoding | Feature content (γ·h + β) |
| Conditioning signal | Raw per-pixel depth value (from data) | Global scale embedding pooled from spatial tokens |
| Where applied | Per-token, before input_proj | Per-channel, after input_proj |
| Dependency on depth data | Requires depth values in metainfo | Does not require additional data; uses feature pool |
| Interaction with backbone | Independent of backbone | Operates on backbone output only |

Idea004's best stage-1 result (design001: composite_val=336.57) beat the baseline (346.58) but is below idea013/design003 (328.14) and idea001/design001 (338.78). The depth positional encoding helps but is limited because it modulates position rather than feature content. FiLM modulation of feature content provides a richer conditioning mechanism.

### Grounding in Observed Results

- **`mpjpe_abs` floor**: the best observed absolute MPJPE (idea008/design002/stage2: 533.77mm) is still far above baseline (833.75mm at stage-1). Absolute MPJPE depends on both relative joint accuracy and the reconstruction quality of absolute 3D from depth+UV predictions. Scale-aware feature conditioning targets this by giving the joint decoder explicit knowledge of the depth scale before computing joint offsets.
- **Body MPJPE floor**: ~156mm across many ideas. Structural ideas (kinematic parameterization, body-focused decoder, multi-layer decoder) have pushed this floor but not broken it decisively. A different lever — feature-level conditioning — may unlock gains that architectural changes cannot.
- **Pelvis MPJPE at stage-1**: ranges 611–862mm; best design at stage-1 is idea005/design003 (611.88mm). Pelvis prediction requires the model to infer absolute depth. If the cross-attention features are explicitly scale-conditioned, the pelvis depth output (which reads from token 0 of the decoded queries) has richer depth-informative features to work with.

### Intuition Check

Consider two training samples:
1. A person at 3 m: spatial tokens have large depth values, large feature activations near body pixels (large image footprint), steep depth gradients at limb boundaries.
2. A person at 8 m: spatial tokens have small depth values, small activations (person is small in the crop), shallow depth gradients.

In the baseline, the cross-attention applies the same transformation regardless of which type of sample is present. The FiLM conditioning network learns a summary `z` (pooled from all 960 spatial tokens) that implicitly encodes the scale-context of the scene and produces (γ, β) that amplify depth-sensitive feature channels for sample 1 and suppress them for sample 2, or vice-versa. Crucially, γ and β can shift the *direction* of feature activation, not just the magnitude, allowing the conditioning to reshape the feature manifold for scale-specific joint regression.

---

## Proposed Variations

### Design A — Global Average Pool Conditioning (minimal FiLM)

The simplest FiLM conditioning: pool the 960 spatial tokens by global average pooling to produce a single `(B, hidden_dim)` context vector, then apply a 2-layer MLP to produce `(γ, β) ∈ R^{2 × hidden_dim}`:

```python
# After: spatial = input_proj(feat) + pos_enc  → (B, 960, hidden_dim)
context = spatial.mean(dim=1)                    # (B, hidden_dim) — global average pool
gamma, beta = self.film_net(context).chunk(2, dim=-1)  # each (B, hidden_dim)
gamma = gamma + 1.0                              # residual: identity at init
spatial = spatial * gamma.unsqueeze(1) + beta.unsqueeze(1)  # FiLM: (B, 960, hidden_dim)
# → decoder_layer(queries, spatial_film)
```

`film_net` is a 2-layer MLP: `Linear(hidden_dim, hidden_dim//2) → GELU → Linear(hidden_dim//2, 2*hidden_dim)`, with the output weight initialized to near-zero so that at initialisation, `gamma ≈ 1` and `beta ≈ 0` (i.e., the identity transform). This ensures training starts exactly at the baseline configuration and the FiLM modulation is learned incrementally.

The `+1.0` offset on γ implements the residual initialisation: since the MLP output is near-zero at init, γ starts at 1.0 (multiplicative identity) and β starts at 0.0 (additive identity).

Parameter count: `hidden_dim×(hidden_dim//2) + (hidden_dim//2)×(2*hidden_dim)` = `256×128 + 128×512` = 32768 + 65536 ≈ 100K parameters. Negligible on a 300M parameter model.

Config kwarg: `film_pool_type='avg'` (str literal), `film_hidden_dim=128` (int literal).

### Design B — Depth-Pool Conditioning (depth-max pool + global avg pool concat)

Rather than pooling the full hidden-dim feature, construct the conditioning signal from two complementary pools:
1. **Global max pool**: captures the peak activations (most salient foreground tokens).
2. **Global average pool**: captures the mean context.

Concatenate them → `(B, 2*hidden_dim)` → MLP → `(γ, β)`:

```python
ctx_avg = spatial.mean(dim=1)           # (B, hidden_dim)
ctx_max = spatial.max(dim=1).values     # (B, hidden_dim)
ctx = torch.cat([ctx_avg, ctx_max], dim=-1)  # (B, 2*hidden_dim)
gamma, beta = self.film_net(ctx).chunk(2, dim=-1)
gamma = gamma + 1.0
spatial = spatial * gamma.unsqueeze(1) + beta.unsqueeze(1)
```

The max pool highlights the highest-activation tokens (typically the most body-proximate tokens) while the average pool provides global scene context. Together they form a richer depth-scale descriptor than average pooling alone.

`film_net`: `Linear(2*hidden_dim, hidden_dim) → GELU → Linear(hidden_dim, 2*hidden_dim)`.

Parameter count: `512×256 + 256×512` = 131K + 131K ≈ 262K parameters. Still negligible.

Config kwarg: `film_pool_type='avg_max'`.

### Design C — Hierarchical FiLM: Spatial-Group Conditioning (body-region aware)

Rather than one global conditioning signal for all 960 spatial tokens uniformly, divide the 24×40 feature grid into a 4×4 = 16 spatial blocks and compute the FiLM parameters independently per block. Each block's context vector (average of the 60 tokens in that block) produces its own `(γ_block, β_block)`, modulating only the tokens within that block:

```python
# Reshape spatial tokens to (B, 4, 4, 60, hidden_dim) using view, then:
ctx_blocks = spatial_4d.mean(dim=3)    # (B, 4, 4, hidden_dim) — 16 block contexts
film_params = self.film_net(ctx_blocks)  # (B, 4, 4, 2*hidden_dim)
gamma_blocks, beta_blocks = film_params.chunk(2, dim=-1)  # each (B, 4, 4, hidden_dim)
gamma_blocks = gamma_blocks + 1.0
# Scatter back: each token in block (r,c) gets (gamma_blocks[:,r,c,:], beta_blocks[:,r,c,:])
spatial_film = ...  # (B, 960, hidden_dim)
```

The 16-block spatial decomposition allows the FiLM conditioning to be spatially non-uniform: upper-body blocks (torso, head region) may have different depth gradients than lower-body blocks (legs, feet). This spatial specificity is especially useful for RGBD crops where depth gradients cluster at body boundaries.

The `film_net` is shared across all 16 blocks (to save parameters and encourage generalization): `Linear(hidden_dim, hidden_dim//2) → GELU → Linear(hidden_dim//2, 2*hidden_dim)`. Applied independently per block context.

Parameter count: same as Design A (~100K). Computation: 16 independent MLP applications vs 1 in Design A — modest overhead.

Config kwarg: `film_pool_type='spatial_block'`, `film_num_blocks=16`.

The 4×4 block decomposition of 24×40 tokens: exact block boundaries may produce non-uniform block sizes (24÷4=6 rows, 40÷4=10 columns → 6×10=60 tokens per block). The Designer should verify the exact reshape logic. Alternative: use 3×5=15 blocks (each 8×8=64 tokens), which may be more uniform. This choice is left to the Designer.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py` and `config.py`.

### `pose3d_transformer_head.py`

**`__init__` changes:**
```python
# New constructor kwargs:
#   film_pool_type: str = 'none'   ('none' = disabled, 'avg', 'avg_max', 'spatial_block')
#   film_hidden_dim: int = 128     (bottleneck dim in the MLP)
#   film_num_blocks: int = 16      (only used when film_pool_type='spatial_block')

self.film_pool_type = film_pool_type
if film_pool_type == 'avg':
    in_dim = hidden_dim
elif film_pool_type == 'avg_max':
    in_dim = 2 * hidden_dim
elif film_pool_type == 'spatial_block':
    in_dim = hidden_dim  # per-block context
else:
    in_dim = 0  # disabled

if in_dim > 0:
    self.film_net = nn.Sequential(
        nn.Linear(in_dim, film_hidden_dim),
        nn.GELU(),
        nn.Linear(film_hidden_dim, 2 * hidden_dim),
    )
    # Near-zero init for output layer → identity transform at start
    nn.init.zeros_(self.film_net[-1].weight)
    nn.init.zeros_(self.film_net[-1].bias)
```

**`forward()` changes:**
After computing `spatial = self.input_proj(spatial_flat) + pos_enc`, and before `decoder_layer(queries, spatial)`, apply FiLM:

```python
if self.film_pool_type == 'avg':
    ctx = spatial.mean(dim=1)          # (B, hidden_dim)
    film = self.film_net(ctx)          # (B, 2*hidden_dim)
    gamma, beta = film.chunk(2, dim=-1)
    gamma = gamma + 1.0
    spatial = spatial * gamma.unsqueeze(1) + beta.unsqueeze(1)

elif self.film_pool_type == 'avg_max':
    ctx = torch.cat([spatial.mean(1), spatial.max(1).values], dim=-1)
    film = self.film_net(ctx)
    gamma, beta = film.chunk(2, dim=-1)
    gamma = gamma + 1.0
    spatial = spatial * gamma.unsqueeze(1) + beta.unsqueeze(1)

elif self.film_pool_type == 'spatial_block':
    B, L, D = spatial.shape  # L=960, D=hidden_dim
    # Designer chooses block decomposition; example: 6 row-blocks × 10 col-blocks
    spatial_blocks = spatial.view(B, 6, 10, 10, 4, D)  # reshape to blocks
    # (actual reshape logic depends on H'=24, W'=40 layout; left to Designer)
    ctx_blocks = spatial_blocks.mean(dim=...)  # (B, 16, D)
    film_params = self.film_net(ctx_blocks)    # (B, 16, 2*D)
    gamma_b, beta_b = film_params.chunk(2, -1)
    gamma_b = gamma_b + 1.0
    # scatter back; each of 960 tokens gets the FiLM params of its block
    spatial = ...  # FiLM-modulated spatial tokens
```

The `forward()` change is entirely contained between the positional encoding addition and the `decoder_layer` call — a surgical insertion with no other side effects.

**`loss()` and `predict()`**: unchanged. FiLM modulation happens inside `forward()`, which both `loss()` and `predict()` call.

**`_init_head_weights()`**: add the zeros-init for `film_net[-1]` as shown above (already included in `__init__`).

### `config.py`
- Add `film_pool_type='avg'` (str literal, Design A)
- Add `film_pool_type='avg_max'` (str literal, Design B)
- Add `film_pool_type='spatial_block'` and `film_num_blocks=16` (str + int literals, Design C)
- Add `film_hidden_dim=128` (int literal, all designs)

No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or `train.py` wrapper.

---

## Expected Outcome

- **Primary gain — `mpjpe_abs`**: scale-aware cross-attention features give the decoded joint representations an explicit depth-scale prior. Absolute pose reconstruction combines relative joints + pelvis depth; if the relative joints are predicted with scale-appropriate features, the absolute MPJPE should improve. Target: `mpjpe_abs < 500` at stage-1 (vs. best prior 747.25mm at stage-1 — idea008/design002).
- **Secondary gain — body MPJPE**: cross-attention features conditioned on scene scale should produce sharper, more discriminative representations for joint localisation. Target: `mpjpe_body_val < 185` at stage-1 (vs. best prior 183.16mm — idea002/design003).
- **Pelvis MPJPE**: FiLM conditioning provides the pelvis depth regression head (which reads from token 0 of decoded queries) with scale-informed features. Expected mild improvement: `mpjpe_pelvis_val < 600` at stage-1 (vs. best prior 611.88mm — idea005/design003).
- **Composite target (stage-1)**: aim for `composite_val < 320` (vs. best prior 328.14 — idea013/design003).
- **Composite target (stage-2)**: aim for `composite_val < 215` (vs. best prior 224.52 — idea001/design001).
- **Design A**: diagnostic for whether any FiLM conditioning helps. Minimal complexity baseline within the idea.
- **Design B**: tests the richer dual-pool conditioning descriptor. Expected to outperform Design A if max-pool captures salient foreground depth information not in average pool.
- **Design C**: spatial FiLM allows body-region-specific scale conditioning. Highest potential but most implementation complexity. If the spatial blocking introduces useful non-uniformity, this should outperform both Design A and B.

---

## Risk and Mitigation

- **Identity initialisation of FiLM**: the near-zero output weight initialisation ensures `gamma ≈ 1, beta ≈ 0` at step 0. Training starts identical to the baseline (no FiLM effect initially). As training proceeds, the FiLM network learns to inject useful conditioning. This is analogous to how residual connections are initialised at zero in ResNets. If the FiLM net learns nothing useful, the model degrades gracefully to the baseline (γ=1, β=0).

- **Self-conditioning circularity**: the conditioning signal is pooled from the same spatial tokens that are being modulated. This is a form of self-attention with global pooling — not circular in the problematic sense because the pool is computed before the modulation (feedforward, not iterative). The gradient of the FiLM parameters flows through both the pool path and the modulated path, which is the standard FiLM training protocol.

- **Pooling losing spatial information**: global average pooling discards spatial structure. Mitigation: Design B (avg+max) retains more spatial peak information; Design C directly addresses this with spatial-block pooling.

- **Design C reshape complexity**: the exact reshape of 960 tokens to spatial blocks depends on H'=24, W'=40 and the block layout. A 4×4 block decomposition (24÷4=6 row-block size, 40÷4=10 col-block size) gives 16 blocks of 60 tokens each. The Designer should use `spatial.view(B, 24, 40, D).view(B, 4, 6, 4, 10, D).mean([2, 4])` → `(B, 4, 4, D)` reshaped to `(B, 16, D)`. The scatter-back is: expand `(B, 4, 4, D)` to `(B, 4, 6, 4, 10, D)` via `.unsqueeze(2).expand(-1,-1,6,-1,10,-1)`, then `.reshape(B, 960, D)`. This is pure tensor ops, no loops.

- **AMP compatibility**: FiLM involves element-wise multiply and add in `float16` (AMP ON). These operations are numerically stable; the near-zero output init prevents overflow at the start. The GELU and Linear ops are standard AMP-compatible ops.

- **Memory**: FiLM forward adds one global pool + one small MLP forward. The pool over 960 tokens is a `mean(dim=1)` — O(960 × hidden_dim) = O(245K) FLOPs per batch element, negligible. The MLP is O(128K) parameters total. No additional spatial tensor allocation beyond the `(B, 960, hidden_dim)` spatial tensor itself (FiLM modifies it in-place or creates a view + result, same memory footprint as a LayerNorm).

- **MMEngine config constraint**: `film_pool_type` is a str literal; `film_hidden_dim`, `film_num_blocks` are int literals. No Python imports required. Fully compliant.

- **Interaction with idea004 (depth positional encoding)**: orthogonal. idea004 adds depth values to positional encoding (spatial token position signals); idea016 modulates feature content (the token representation itself). Combining both would give spatial tokens both depth-aware positions and depth-conditioned feature content. This composition is valid but left to a future idea.

- **Interaction with idea008 (body-focused 22-query decoder)**: FiLM modulation is applied to spatial tokens before cross-attention. It is agnostic to whether there are 22 or 70 queries. Combining idea016 FiLM + idea008 body-focused decoder would give: scale-conditioned spatial tokens + query-side pollution removal. This composition is promising but left to future ideas.

- **Interaction with idea015 (super-token pooling)**: FiLM modulation could be applied to spatial tokens before or after super-token pooling. Before pooling: the FiLM-modulated tokens are then compressed into super-tokens, potentially improving pool quality. After pooling: the super-tokens are FiLM-modulated. Both are valid; the Designer should apply FiLM before any token compression in future combinations.

- **Eval/inference compatibility**: `predict()` calls `self.forward(feats)`, which now applies FiLM inside. Output tensor shapes are unchanged. `BedlamMPJPEMetric` and `TrainMPJPEAveragingHook` see identical interfaces.
