**Idea Name:** Geometry-Guided Cascaded Decoder with Reprojection-Conditioned Inter-Layer Attention

**Approach:** Stack two transformer decoder layers; after the first layer produces an intermediate 3D pose prediction, project those predicted joints to 2D image coordinates via the camera intrinsics K and construct a dynamic Gaussian attention bias over the spatial feature grid centred on each predicted 2D joint location, which is injected into the second layer's cross-attention — so each joint query's second-pass spatial attention is automatically focused on the image region where the first-pass prediction expects the joint to appear.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Static Attention Bottleneck

The baseline single-layer decoder must solve two fundamentally different sub-problems in a single pass:
1. **Spatial routing**: each joint query must attend to the right spatial region of the 40×24 feature grid to extract relevant appearance features.
2. **3D coordinate regression**: the attended features must be transformed into a 3D coordinate estimate.

Both sub-problems compete for the same cross-attention computation. The single-pass result is a compromise: the attention distribution is simultaneously trying to route to the right spatial region and aggregate the right features for regression.

The natural solution — adopted by idea001 (multi-layer decoder) — is to distribute these sub-problems across layers: early layers establish coarse spatial routing, later layers refine. idea001/design001 achieved the best stage-2 composite (224.52 vs. baseline 243.72 — an 8% improvement). But its success used only 2 layers with no mechanism to guide the second layer's cross-attention — the second layer had to *learn* where to look independently from the first layer's output, with no explicit coupling.

### The Proposed Closed-Loop Refinement

This idea adds an explicit geometric **feedback signal** from the first decoder layer to the second: after layer 1 produces an intermediate 3D pose estimate, each joint's predicted 3D position is projected to 2D image coordinates (using the camera intrinsic K and the predicted pelvis depth from layer 1). The resulting 2D locations are used to construct a **dynamic Gaussian attention bias** for layer 2's cross-attention: a per-joint bell-shaped additive bias centred on the projected 2D location on the 40×24 spatial grid.

Formally, for joint `i` with intermediate predicted 3D position `p_i^(1)` (root-relative) and predicted pelvis depth `d^(1)`:
1. Compute absolute predicted position: `P_i = p_i^(1) + pelvis_3d^(1)` where `pelvis_3d^(1)` is recovered from `(d^(1), uv^(1))` via `recover_pelvis_3d`.
2. Project to image coordinates: `(u_i, v_i) = project(P_i, K)` → normalised to `[-1, 1]`, then convert to feature grid coordinates `(h_i, w_i)` in `[0, H'=40)` × `[0, W'=24)`.
3. Construct a Gaussian bias over the spatial grid: `B_i[h, w] = γ * exp(-((h - h_i)^2 + (w - w_i)^2) / (2σ^2))` where `σ` is a learnable or fixed standard deviation.
4. Pass `B_i` as an `attn_mask` additive bias to layer 2's cross-attention.

This creates a **closed feedback loop**: layer 1 predicts → geometry informs where to look → layer 2 refines with focused attention. The key novelty is that the attention bias for layer 2 is **dynamically computed from intermediate predictions**, not a static learnable bias.

### Why This is Different from All 21 Prior Ideas

| Prior Idea | Mechanism | Key Difference |
|---|---|---|
| idea001 | 2-layer decoder, no inter-layer coupling | No geometric feedback: layer 2 must independently re-learn spatial routing |
| idea010 | 2D reprojection *loss* | Loss term on final output; no effect on intermediate attention |
| idea011 | Iterative refinement: 2nd decoder pass reads first-pass coords as query offset | Coords are passed as *query offset* (query space), not as a cross-attention bias over spatial tokens |
| idea019 | Deformable sampling: predicted 2D offsets select sparse spatial tokens at *input* | Applied at entry to the first decoder pass; no cascaded feedback between layers |
| idea021 | Learnable *static* cross-attention spatial bias | Bias is a fixed learned parameter (same for every image); this idea computes the bias dynamically from per-image intermediate predictions — entirely image-specific and geometry-grounded |
| idea020 | Per-query temperature scaling | Controls attention *sharpness* uniformly over all spatial positions; does not localize to predicted joint locations |
| idea007 | Multiplicative gating of cross-attention values | Channel-level gating of output values, not logit-level localization of attention regions |

**This is the first idea to:**
1. Explicitly couple the output of one decoder layer to the spatial attention pattern of the next via geometric projection.
2. Compute cross-attention biases *dynamically per image* (as opposed to all prior attention-bias ideas which use static learned or fixed parameters).

The dynamic nature is the key differentiator: because the bias is computed from intermediate predictions (not from GT), the model learns to (a) produce accurate intermediate predictions in layer 1 (which will yield accurate biases for layer 2) and (b) use those biases effectively in layer 2 for refinement. These are complementary optimization pressures that reinforce each other.

### Grounding in Observed Results

**idea001/design001** is the best stage-2 result (composite=224.52). It achieves this with a 2-layer decoder and no intermediate supervision. Adding geometric feedback between these same two layers is a direct enhancement: we keep what works (2 decoder layers) and add a novel coupling (reprojection-conditioned attention).

**idea010/design002** achieved the best body MPJPE at stage-2 (168.79 mm) via a 2D reprojection loss coupling the joint and pelvis pathways. The reprojection loss demonstrates that geometric constraints (2D projection from 3D) provide a strong training signal. This idea uses the same geometric projection operation, but as attention guidance rather than as a loss term.

**idea021** (learnable static cross-attention spatial bias) proposes adding spatial priors to cross-attention logits. The limitation is that the prior must be the same for all images (static learned parameter) — it cannot be tailored to a specific image's predicted pose. This idea supersedes it by making the spatial bias fully image-specific.

**The 8% plateau**: Despite 21 ideas, the best stage-2 improvement over baseline is ~8% (224.52 vs 243.72). The common thread in top ideas is that they provide better spatial routing or geometric coupling. This idea directly synthesises both into a single, principled mechanism.

---

## Proposed Variations

### Design A — Fixed Gaussian Bandwidth, No Intermediate Loss

Two decoder layers. After layer 1's intermediate joint and pelvis predictions, construct a dynamic Gaussian cross-attention bias with **fixed bandwidth** `σ = 4.0` (grid cells on the 40×24 feature map) and a **fixed amplitude** `γ = 2.0`. The bias is added to layer 2's cross-attention logits via `attn_mask`.

No intermediate loss on layer 1's output — layer 1 is supervised only through the gradient that backpropagates through the bias construction and layer 2's final output. This is the minimal-change test: does dynamic geometric feedback improve over plain multi-layer decoding?

The Gaussian bandwidth σ=4 spans approximately 4 feature cells ≈ 64 pixels at the 640×384 input scale — a reasonable uncertainty radius for a first-pass prediction. With `γ=2.0`, the bias boosts attention at the predicted joint location by `e^2 ≈ 7×` relative to distant tokens (before softmax normalisation).

Config kwargs (all literals): `num_decoder_layers=2`, `use_reproj_bias=True`, `reproj_bias_sigma=4.0`, `reproj_bias_gamma=2.0`, `aux_loss_weight=0.0`.

### Design B — Fixed Gaussian Bandwidth with Auxiliary Intermediate Loss

Same 2-layer decoder and geometric feedback as Design A, but add an **auxiliary joint regression loss** on the layer 1 intermediate output:

```
loss_total = final_losses + aux_loss_weight * loss_joints(layer1_joints[:, 0:22], gt_joints[:, 0:22])
```

with `aux_loss_weight=0.4`. No intermediate loss on depth/UV (to avoid the pelvis regression degradation observed in idea001/designs 002-003 with 3-4 layer intermediate supervision).

The auxiliary loss serves two purposes:
1. It directly supervises layer 1's joint predictions, making the intermediate 3D estimates (and hence the reprojection bias for layer 2) more accurate earlier in training.
2. It provides gradient signal to the layer 1 FFN and attention weights without requiring the full layer 2 computation.

Design B is expected to outperform Design A because the auxiliary loss bootstraps the quality of the reprojection bias from early training epochs, when the unsupervised feedback (Design A) would be noisy.

Config kwargs: `num_decoder_layers=2`, `use_reproj_bias=True`, `reproj_bias_sigma=4.0`, `reproj_bias_gamma=2.0`, `aux_loss_weight=0.4`.

### Design C — Learnable Gaussian Bandwidth with Auxiliary Loss

Same 2-layer decoder, geometric feedback, and auxiliary loss as Design B, but replace the fixed `σ` and `γ` with **learnable per-joint scalars**:
- `self.bias_sigma = nn.Parameter(torch.ones(num_joints) * 4.0)` — per-joint bandwidth
- `self.bias_gamma = nn.Parameter(torch.ones(num_joints) * 2.0)` — per-joint amplitude

`σ` is passed through `F.softplus` to ensure positivity. The model can learn narrow bandwidths for distal joints (wrists, ankles — small, precisely localised) and wide bandwidths for proximal joints (spine, pelvis — larger region of uncertainty). This directly mirrors the insight from idea020 (per-query temperature: focal for distal, diffuse for proximal), but applied to the geometric reprojection bias rather than the attention logits directly.

The learnable parameters are initialized to sensible defaults (σ=4, γ=2) so the model starts from the same state as Design B.

Config kwargs: `num_decoder_layers=2`, `use_reproj_bias=True`, `reproj_bias_learnable=True`, `aux_loss_weight=0.4`.

---

## Implementation Scope

All changes are confined to **two** allowed files: `pose3d_transformer_head.py` and `config.py`. One new helper is added to `pelvis_utils.py`.

### `pelvis_utils.py`

Add a new helper function `project_joints_to_feat_grid`:

```python
def project_joints_to_feat_grid(
    joints_abs: torch.Tensor,   # (B, J, 3) absolute camera-frame joints
    K: np.ndarray,               # (3, 3) crop intrinsic
    crop_h: int,
    crop_w: int,
    feat_h: int = 40,
    feat_w: int = 24,
) -> torch.Tensor:
    """Project absolute 3D joints to feature grid coordinates (h, w).

    Returns:
        (B, J, 2) float tensor: (h_frac, w_frac) in feature grid units [0, feat_h) x [0, feat_w)
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = joints_abs[..., 0].clamp(min=0.01)   # (B, J) depth (forward)
    Y = joints_abs[..., 1]                    # (B, J) lateral
    Z = joints_abs[..., 2]                    # (B, J) vertical

    u_px = -Y / X * fx + cx   # (B, J) — pixel u in crop
    v_px = -Z / X * fy + cy   # (B, J) — pixel v in crop

    # Convert to feature grid coordinates (feature stride = 16)
    h_frac = v_px / crop_h * feat_h   # (B, J)
    w_frac = u_px / crop_w * feat_w   # (B, J)

    return torch.stack([h_frac, w_frac], dim=-1)   # (B, J, 2)
```

This is a small (15-line) fully differentiable helper. It uses the same projection convention as `recover_pelvis_3d` (already in `pelvis_utils.py`) and the `project_joints_to_2d` pattern from idea010.

### `pose3d_transformer_head.py`

**1. New helper: `_build_gaussian_bias`** (module-level function):

```python
def _build_gaussian_bias(
    joint_feat_coords: torch.Tensor,   # (B, J, 2) — (h, w) in feature grid
    feat_h: int,
    feat_w: int,
    sigma: torch.Tensor,               # (J,) or scalar
    gamma: torch.Tensor,               # (J,) or scalar
) -> torch.Tensor:
    """Build dynamic Gaussian cross-attention bias.

    Returns:
        (B, J, feat_h * feat_w) additive bias for cross-attention logits.
    """
    B, J, _ = joint_feat_coords.shape
    # Grid coordinates for all spatial tokens
    grid_h = torch.arange(feat_h, device=joint_feat_coords.device,
                           dtype=joint_feat_coords.dtype)  # (feat_h,)
    grid_w = torch.arange(feat_w, device=joint_feat_coords.device,
                           dtype=joint_feat_coords.dtype)  # (feat_w,)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing='ij')   # each (feat_h, feat_w)
    grid = torch.stack([gh, gw], dim=-1).view(-1, 2)          # (feat_h*feat_w, 2)

    # joint_feat_coords: (B, J, 2), grid: (feat_h*feat_w, 2)
    mu = joint_feat_coords.unsqueeze(-2)    # (B, J, 1, 2)
    g = grid.view(1, 1, -1, 2)             # (1, 1, H'W', 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)    # (B, J, H'W')

    # sigma: (J,) → (1, J, 1); gamma: (J,) → (1, J, 1)
    s = sigma.view(1, -1, 1).clamp(min=0.5)
    g_ = gamma.view(1, -1, 1)
    bias = g_ * torch.exp(-dist2 / (2.0 * s ** 2))   # (B, J, H'W')
    return bias
```

**2. `_DecoderLayer` changes:**

Add optional `cross_attn_bias` argument to `forward()` (same pattern as idea021):

```python
def forward(self, queries, spatial_tokens, cross_attn_bias=None):
    ...
    q = self.norm2(queries)
    if cross_attn_bias is not None:
        # Expand bias across batch: (B, J, H'W') or (J, H'W')
        # nn.MultiheadAttention attn_mask: (tgt_len, src_len) or (B*nheads, tgt_len, src_len)
        # For per-sample dynamic bias, use (B*nheads, J, H'W') by repeating over heads
        B, J, _ = q.shape
        nheads = self.self_attn.num_heads
        mask = cross_attn_bias.unsqueeze(1).expand(-1, nheads, -1, -1)  # (B, nheads, J, H'W')
        mask = mask.reshape(B * nheads, J, -1)                           # (B*nheads, J, H'W')
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                              attn_mask=mask.to(q.dtype))[0]
    else:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
    ...
```

**3. `Pose3dTransformerHead.__init__` changes:**

```python
# New kwargs (all with defaults):
num_decoder_layers: int = 1    # 1 (baseline) or 2 (this idea)
use_reproj_bias: bool = False  # whether to compute dynamic Gaussian feedback
reproj_bias_sigma: float = 4.0
reproj_bias_gamma: float = 2.0
reproj_bias_learnable: bool = False   # Design C only
aux_loss_weight: float = 0.0          # intermediate supervision weight
feat_h: int = 40
feat_w: int = 24
```

When `num_decoder_layers > 1`, create `nn.ModuleList` of decoder layers:
```python
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

When `reproj_bias_learnable=True`, create `nn.Parameter` tensors for sigma and gamma.

**4. `Pose3dTransformerHead.forward()` changes:**

```python
decoded = self.decoder_layers[0](queries, spatial)   # (B, J, D) — layer 1

if self.num_decoder_layers > 1 and self.use_reproj_bias:
    # Compute intermediate predictions from layer 1
    layer1_joints = self.joints_out(decoded)         # (B, J, 3)
    layer1_depth  = self.depth_out(decoded[:, 0])    # (B, 1)
    layer1_uv     = self.uv_out(decoded[:, 0])       # (B, 2)

    # Build per-sample reprojection biases
    # (This is done in loss() where K is available; forward() uses the stored bias)
    # NOTE: forward() does not have access to K — it is in batch_data_samples.
    # For training, the bias is built in loss() and stored in self._reproj_bias.
    # For forward() called from predict(), no bias is applied (test-time simplification).
    bias = getattr(self, '_reproj_bias', None)
    decoded = self.decoder_layers[1](decoded, spatial, cross_attn_bias=bias)
elif self.num_decoder_layers > 1:
    decoded = self.decoder_layers[1](decoded, spatial)
```

**Handling the K-availability problem**: Camera intrinsics `K` are available in `batch_data_samples` (as `ds.metainfo['K']`), but `forward()` is called without `batch_data_samples`. The cleanest solution is to compute the bias in `loss()` before calling `forward()`, temporarily store it as `self._reproj_bias`, and clear it after `forward()` returns. This is a controlled side-channel (not a regression in test-time behaviour since `predict()` runs without the bias — a conservative test-time choice that avoids any inference-time K parsing).

An alternative (cleaner) design: add a `context` optional argument to `forward()` that carries `K` when available, and builds the bias inline. This avoids the side-channel. The Designer should evaluate both options.

**5. `loss()` additions:**

For the auxiliary intermediate loss (Design B/C):
```python
if self.aux_loss_weight > 0.0:
    layer1_joints = self.joints_out(layer1_decoded)
    losses['loss/joints_aux/train'] = (
        self.aux_loss_weight * self.loss_joints_module(
            layer1_joints[:, _BODY], gt_joints[:, _BODY]))
```

For the reprojection bias:
```python
if self.use_reproj_bias:
    # Assemble absolute 3D positions from layer 1 predictions
    abs_joints_list = []
    for i in range(B):
        K = np.asarray(batch_data_samples[i].metainfo['K'], dtype=np.float32)
        img_shape = batch_data_samples[i].metainfo.get('img_shape', (640, 384))
        pelvis = recover_pelvis_3d(
            layer1_depth[i:i+1], layer1_uv[i:i+1], K,
            int(img_shape[0]), int(img_shape[1]))   # (1, 3)
        abs_j = layer1_joints[i] + pelvis           # (J, 3)
        abs_joints_list.append(abs_j)
    abs_joints = torch.stack(abs_joints_list)        # (B, J, 3)

    # Project to feature grid coordinates
    feat_coords_list = []
    for i in range(B):
        K = np.asarray(batch_data_samples[i].metainfo['K'], dtype=np.float32)
        img_shape = batch_data_samples[i].metainfo.get('img_shape', (640, 384))
        fc = project_joints_to_feat_grid(
            abs_joints[i:i+1], K,
            int(img_shape[0]), int(img_shape[1]),
            self.feat_h, self.feat_w)   # (1, J, 2)
        feat_coords_list.append(fc[0])
    feat_coords = torch.stack(feat_coords_list)      # (B, J, 2)

    # Build dynamic Gaussian bias
    sigma = (torch.nn.functional.softplus(self.bias_sigma)
             if self.reproj_bias_learnable else
             torch.full((self.num_joints,), self.reproj_bias_sigma,
                        device=feat_coords.device))
    gamma = (self.bias_gamma
             if self.reproj_bias_learnable else
             torch.full((self.num_joints,), self.reproj_bias_gamma,
                        device=feat_coords.device))
    self._reproj_bias = _build_gaussian_bias(
        feat_coords, self.feat_h, self.feat_w, sigma, gamma)   # (B, J, H'W')
```

### `config.py`

**Design A:**
```python
num_decoder_layers=2,
use_reproj_bias=True,
reproj_bias_sigma=4.0,
reproj_bias_gamma=2.0,
reproj_bias_learnable=False,
aux_loss_weight=0.0,
feat_h=40,
feat_w=24,
```

**Design B:**
```python
num_decoder_layers=2,
use_reproj_bias=True,
reproj_bias_sigma=4.0,
reproj_bias_gamma=2.0,
reproj_bias_learnable=False,
aux_loss_weight=0.4,
feat_h=40,
feat_w=24,
```

**Design C:**
```python
num_decoder_layers=2,
use_reproj_bias=True,
reproj_bias_sigma=4.0,
reproj_bias_gamma=2.0,
reproj_bias_learnable=True,
aux_loss_weight=0.4,
feat_h=40,
feat_w=24,
```

All values are bool/int/float literals. No Python import statements. Fully compliant with the MMEngine config constraint.

---

## Expected Outcome

- **Primary mechanism — body MPJPE**: the second decoder layer attends to the correct spatial regions for each joint (guided by first-pass predictions), extracting sharper, more discriminative joint features. Target: `mpjpe_body_val < 188` at stage-1 (vs. baseline 195.7 mm), `< 168` at stage-2 (matching or improving on best prior 168.79 mm from idea010).

- **Secondary mechanism — pelvis MPJPE**: the pelvis query (index 0) benefits from two decoder passes, and the second pass is guided by the first-pass pelvis projection. The pelvis query in layer 2 attends to a focused region around its predicted position, potentially reducing the influence of distant noisy spatial tokens. Target: `mpjpe_pelvis_val < 630` at stage-1, `< 320` at stage-2 (vs. best prior 322.05 mm from idea003/design002).

- **Design A (no intermediate loss)**: diagnostic — does dynamic geometric feedback help without auxiliary supervision? The layer-1 predictions may be noisy early in training (before the backbone learns meaningful features), producing biases that are initially random. Expected composite_val < 340 at stage-1, potentially matching idea001/design001's stage-2 of 224.52.

- **Design B (with auxiliary loss)**: the auxiliary joint loss on layer 1 ensures that layer 1 produces useful intermediate predictions from early training epochs, bootstrapping the quality of the reprojection bias. Expected composite_val < 332 at stage-1, `< 222` at stage-2 — competitive with or better than idea001/design001.

- **Design C (learnable bandwidth)**: per-joint learnable bandwidth allows narrow attention focus for distal joints (wrists, ankles) and broad attention for proximal/pelvis. Expected to outperform Design B by 3–8 mm body MPJPE. Expected composite_val < 328 at stage-1, target `< 220` at stage-2.

- **Composite target (stage-2)**: primary target `composite_val < 220`, improving on the current best of 224.52 (idea001/design001) by ~2%.

---

## Risk and Mitigation

- **K unavailability in `forward()` at training time**: the reprojection bias requires camera intrinsics K from `batch_data_samples`, which are not passed to `forward()`. Mitigation: compute the bias in `loss()` where `batch_data_samples` is available, and store as `self._reproj_bias` for the internal `forward()` call. A cleaner alternative is to refactor `forward()` to accept an optional `reproj_bias` argument. Designer should implement the cleaner approach. At test-time (`predict()`), no reprojection bias is applied — the model falls back to standard 2-layer decoding. This is conservative and safe (the model does not depend on K at inference).

- **Noisy layer-1 predictions early in training**: in the first few epochs, layer-1 joint predictions are random, producing reprojection biases that point to arbitrary image locations. This could disrupt layer-2 attention and slow early convergence. Mitigation: (a) Design A (no aux loss) lets the gradient guide layer 1 organically; (b) Design B's auxiliary loss directly supervises layer 1, improving bias quality from epoch 1. Additionally, the Gaussian bias amplitude γ=2.0 corresponds to a ~7× boost at the predicted location — reasonable but not dominant, so layer 2 retains a broad attention signal even when the bias is inaccurate.

- **Gradient flow through bias construction**: the bias is a function of `layer1_joints`, `layer1_depth`, and `layer1_uv` — all differentiable w.r.t. the layer 1 decoder parameters. The chain rule flows: `d loss_final / d layer1_params = (d loss_final / d bias) * (d bias / d layer1_joints) * (d layer1_joints / d layer1_params)`. This gradient path is differentiable everywhere (the Gaussian is smooth; `recover_pelvis_3d` is differentiable; `project_joints_to_feat_grid` is differentiable). The Designer should verify that AMP does not produce NaN in this path when `layer1_joints` contains very large values early in training — adding a `torch.clamp` on `layer1_depth` (already present in `recover_pelvis_3d`) and on `feat_coords` (clamp to grid bounds) prevents unstable projections.

- **Per-sample K loop overhead**: the bias construction loops over B samples (batch size 4), performing one `recover_pelvis_3d` and one `project_joints_to_feat_grid` call per sample. Total overhead: ~8 numpy scalar ops and one `(1, J, 3)` tensor op per sample. Negligible on 2080 Ti (< 0.5 ms for batch 4). The Gaussian distance computation is a fully vectorised `(B, J, H'W')` tensor operation.

- **AMP float16 compatibility**: the Gaussian bias is computed in float32 (intermediate tensor operations) and cast to query dtype in `_DecoderLayer.forward()` via `.to(q.dtype)`. The Designer must include this cast. The `clamp(min=0.5)` on `sigma` prevents near-zero bandwidth that would produce very sharp biases potentially causing float16 overflow.

- **Feature grid orientation**: the baseline code uses `feat.flatten(2).transpose(1, 2)` where `feat` is `(B, C, H, W)` with `H=40, W=24`. The spatial tokens are therefore in row-major order of the (40, 24) feature grid. The `project_joints_to_feat_grid` function must use `feat_h=40, feat_w=24` consistently, and the Gaussian bias must be shaped `(B, J, 40*24=960)` = `(B, J, 960)`. Config sets `feat_h=40, feat_w=24`. Designer must verify this orientation experimentally.

- **`attn_mask` shape for `batch_first=True` with per-sample bias**: PyTorch's `nn.MultiheadAttention` with `batch_first=True` accepts `attn_mask` of shape `(tgt_len, src_len)` (shared across batch and heads) or `(B*nheads, tgt_len, src_len)` (per-head per-sample). Since our bias is per-sample (dynamic), the correct form is `(B*nheads, J, H'W')`. The Designer must implement this shape correctly. Note: nheads=8, B=4 → mask shape `(32, 70, 960)`. This is a `32*70*960 = ~2.15M` element float16 tensor per forward pass — approximately 4.3 MB per batch with AMP. Well within the 10.57 GB 2080 Ti memory budget.

- **Interaction with body-only decoder (idea008/022)**: if `num_body_queries=22` (reducing decoder to body joints only), the reprojection bias should only be constructed for 22 body joint positions. The `num_joints` parameter controls both the query set size and the bias dimensions. This composition is left for a future idea.

- **Test-time accuracy without bias**: at test time, `predict()` calls `forward()` without `batch_data_samples`, so no reprojection bias is computed. The second decoder layer runs standard cross-attention. This conservative design ensures the model is robust to bias absence at inference — the layer 1 predictions and the learned weights in layer 2 are sufficient even without the dynamic bias. If the Designer finds a significant gap between training (with bias) and test (without bias), an alternative is to add a lightweight K-extraction path to `predict()` that reads K from the `batch_data_samples` passed to `predict()` (which does have `batch_data_samples`).

- **Interaction with idea001/design001**: this idea intentionally reuses the 2-layer decoder structure from idea001/design001 (the best prior result). The only addition is the dynamic reprojection bias between layers. The baseline for comparison should be idea001/design001's stage-1 composite (338.78) and stage-2 composite (224.52). The Designer should ensure that Design A (no intermediate loss) is a clean replication of idea001/design001 + the reprojection bias, so any delta is directly attributable to the bias mechanism.

- **MMEngine config constraint**: all new kwargs are bool/int/float literals. No Python import statements required. The `project_joints_to_feat_grid` import in `pose3d_transformer_head.py` uses the standard `from pelvis_utils import ...` (which is fine since this is a regular Python file, not an MMEngine config file). Fully compliant.

- **Eval/inference compatibility**: `bedlam_metric.py` and `TrainMPJPEAveragingHook` see identical tensor shapes from `forward()` — `{'joints': (B, 70, 3), 'pelvis_depth': (B, 1), 'pelvis_uv': (B, 2)}`. The intermediate layer-1 outputs are internal to the loss computation. No change to metric invariants.

- **Memory**: the primary memory cost is the Gaussian bias tensor `(B*nheads, J, H'W') = (32, 70, 960)` ≈ 4.3 MB (float16). The second decoder layer adds ~same memory as the first (cross-attention + FFN with hidden_dim=256). Net addition: ~8.6 MB total. Well within the 10.57 GB 2080 Ti budget.
