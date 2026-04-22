**Idea Name:** 2D Spatial Heatmap Classification Head for Pelvis UV Localization

**Approach:** Replace the scalar `Linear(hidden_dim, 2)` pelvis UV regression head with a 2D spatial heatmap classification over the H'xW' feature grid (soft-argmax produces continuous UV), supervised by a soft Gaussian target centered on the GT pelvis pixel — turning the UV head into a classification-with-expectation problem (same principle as idea014's depth bin classification, but applied to the 2D UV output).

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Pelvis UV Regression Bottleneck

The baseline predicts pelvis UV via a direct scalar regression from the pelvis-token embedding:

```python
pelvis_uv = self.uv_out(pelvis_token)   # Linear(hidden_dim, 2) -> (B, 2)
```

Supervision is a SmoothL1 loss against GT UV normalized to `[-1, 1]` in crop coordinates. This is analogous to how the baseline also predicts pelvis depth via a `Linear(hidden_dim, 1)` scalar regression.

`mpjpe_pelvis_val = 0.5 * pelvis_depth_err + 0.5 * pelvis_uv_err` (approximately, both contribute to the 3D pelvis position via unprojection through K). The pelvis MPJPE metric has plateaued at 608–740 mm across all 30 prior ideas at stage-1 (baseline 652, best prior 608 from idea023/design001). Even aggressive pelvis-targeted interventions (idea002 decoupled pelvis queries, idea014 depth classification, idea028 fully decoupled pelvis decoder) have moved this number only marginally, and every attempt to improve body MPJPE has either stalled pelvis or made it worse.

**idea014 identified the core structural insight** for depth: scalar regression from a transformer token onto an unbounded continuous scalar is a poor match for what transformer features encode naturally (attention-weighted mixtures of spatial evidence). Classification with soft-argmax over bins converts this to a distribution-matching problem, which empirically works better in monocular depth literature (DORN, AdaBins, BinsFormer). idea014 proposed this for **depth** but not for **UV**.

### Why 2D UV Classification is a Natural Fit (and Structurally Distinct from idea014)

The UV head has a key property the depth head does not have: **UV lives in the same 2D space as the spatial feature grid**. Specifically:

- The spatial tokens are laid out on a 40×24 grid corresponding to the 16× downsampled input crop (640×384 → 40×24).
- The pelvis's GT pixel location projects directly onto this grid: `(u_grid, v_grid) = (u_norm * 12 + 12, v_norm * 20 + 20)` (approximate mapping from `[-1, 1]` to the 40×24 grid).
- Therefore a 2D heatmap of size H'×W' = 40×24 is a **literal spatial distribution over where the pelvis is in the image**, with a direct geometric interpretation.

This is fundamentally different from idea014's depth classification:
- idea014: 1D classification over abstract log-depth bins `[1, 15]m`, no spatial grounding, bins are a learned abstraction.
- idea031 (this idea): 2D classification over the literal feature-map grid, spatially grounded, bins *are* the spatial positions.

The 2D UV heatmap can be predicted directly from the spatial tokens themselves (not just from the pelvis-token embedding), which means the UV head can leverage the full 40×24 = 960 spatial features rather than compressing everything through a single 256-dim pelvis token. This is architecturally cleaner for a spatial localization task.

### How the 2D Heatmap UV Head Works

**Prediction:**
1. Project spatial tokens to a per-location scalar logit: `uv_logits = Linear(hidden_dim, 1)(spatial)` → `(B, H'W', 1)`. Reshape to `(B, H', W')`.
2. Softmax over spatial dimensions: `uv_attn = softmax(uv_logits.flatten(1)).reshape(B, H', W')` — a proper probability distribution over the 960 grid cells.
3. Soft-argmax to recover continuous `(u_norm, v_norm)` in `[-1, 1]`:
```python
u_grid = sum(uv_attn * W_index_grid) / W'  ∈ [0, 1)
u_norm = u_grid * 2 - 1                   ∈ [-1, 1]
v_norm = v_grid * 2 - 1                   ∈ [-1, 1]
```
The output shape `(B, 2)` is preserved — downstream `recover_pelvis_3d`, `compute_mpjpe_abs`, and `bedlam_metric` see the same tensor interface as the baseline.

**Supervision (dual-mode):**
- **Soft-argmax L1 loss** (inherits the baseline's regression signal, now backed by a distribution): `SmoothL1(soft_argmax_uv, gt_uv)` at weight 1.0. This is the baseline loss path, preserving training dynamics.
- **Classification heatmap loss**: build a soft Gaussian target heatmap centered on the GT pixel location with σ=2 grid cells; compute KL divergence between `uv_attn` and the GT heatmap. This provides dense per-cell gradient to the spatial tokens (not just the pelvis token), directly training the spatial features to encode "where is the pelvis" information. Weight: λ_heatmap = 0.5.

Both losses coexist. The soft-argmax loss ensures the continuous UV stays accurate (the actual quantity used by `recover_pelvis_3d`); the heatmap loss provides the dense classification-style gradient to the spatial tokens, which is the key mechanism enabling the classification advantage.

**Initialization for baseline-equivalent start:**
The `uv_logits_proj: Linear(hidden_dim, 1)` is initialized to **zero weights and zero bias** → `uv_logits ≡ 0` at training start → `uv_attn` is uniform (1/960 per cell) → `soft_argmax` returns (0, 0) in `[-1, 1]` (grid centre). This matches the GT mean of UV in BEDLAM2 (roughly centred on the person crop, which is a close approximation to GT mean). Not literally identical to baseline at step 0, but close (within a few hundred pixels), and the smooth-argmax gradient immediately steers the distribution toward GT within the first few steps — exactly the same dynamics as idea014's zero-init of the bin expectation.

### Why This is Different from All Prior Ideas

| Prior Idea | Mechanism | Difference from idea031 |
|---|---|---|
| idea014 (anchor-based depth classification) | 1D depth bins, soft-argmax | Operates on depth (scalar axis), not UV (2D spatial). Bins are abstract log-depth values, not spatial grid cells. |
| idea023 (heatmap-guided query init) | 2D heatmap over spatial grid for **body joint queries**, used as pooling weights for query warm-start | Operates on body joint queries (indices 0-21), for **input** to decoder. idea031 operates on pelvis UV as the **output** head. Distinct architectural position (pre-decoder vs post-decoder) and distinct target (22 body joints vs 1 pelvis UV). |
| idea002 (dedicated pelvis query) | Architectural pelvis decoupling | Pelvis still regressed from a token via scalar `Linear(hidden_dim, 2)`. No output-representation change. |
| idea028 (decoupled pelvis coordinate queries with axis-specific cross-attention) | Dedicated pelvis queries with their own cross-attention pass | Pelvis output is still a scalar regression. idea031 changes the output head itself, orthogonal to idea028's decoupling. |
| idea010 (2D reprojection consistency loss) | Loss-level coupling via 2D projection of body joints | Adds a loss term; does not change the UV head output representation. |

**idea031 is the first output-representation change for the pelvis UV head.** It is the natural 2D spatial analogue of idea014 (which was a 1D depth output change) and composes cleanly with every prior idea that targets queries, attention, or losses.

### Grounding in Observed Results

- **Pelvis MPJPE floor at 608 mm** (stage-1, 30 ideas). The pelvis MPJPE has the most remaining improvement potential of any tracked metric. idea014 targets the depth half of pelvis MPJPE; idea031 targets the UV half of pelvis MPJPE. Together they cover the full pelvis localization surface.
- **Composite formula**: `0.67 * body + 0.33 * pelvis`. At pelvis_MPJPE = 608 mm, pelvis contributes 201 mm to composite. Reducing pelvis MPJPE by 50 mm (to 558) would reduce composite by ~16 mm — a substantial jump relative to the 4 mm stage-1 composite gap between idea023 (best 323.75) and the next-best (328.14).
- **idea023 (heatmap over spatial grid for body joints)** validated that 2D heatmap classification over the H'×W' grid is trainable, effective, and produces gradient signal that reaches the spatial tokens. idea023 achieved best stage-1 composite (323.75). idea031 applies the same proven mechanism to a different output (pelvis UV) where it is structurally even more appropriate (single target, directly spatially grounded, simpler supervision).
- **idea008 (body-only decoder)** achieved best stage-2 `mpjpe_abs` (533) and best `mpjpe_rel` (333), showing that restructuring of the head output is a strong lever. idea031 is a restructuring of the UV output.

---

## Proposed Variations

### Design A — Soft-argmax UV heatmap, KL heatmap loss, uniform weight λ_heatmap=0.5 (diagnostic)

Add a `uv_heatmap_proj = nn.Linear(hidden_dim, 1)` applied to the spatial tokens to produce `(B, H'W', 1)` logits. Softmax over spatial axis → spatial distribution → soft-argmax → `(B, 2)` continuous UV. The heatmap loss is `KL(uv_attn || gt_heatmap)` with a Gaussian target (σ=2 grid cells). The continuous UV is supervised by the existing SmoothL1 loss (weight 1.0). Heatmap loss weight: `λ_heatmap = 0.5`.

Zero-init on `uv_heatmap_proj` ensures uniform attention at start (soft-argmax → grid centre). The soft Gaussian target σ=2 gives smoothed gradients near the correct location.

Config kwargs: `use_uv_heatmap=True`, `uv_heatmap_loss_weight=0.5`, `uv_heatmap_sigma=2.0`, `uv_heatmap_target='gaussian'`, `feat_h=40`, `feat_w=24`.

Design A is the minimal diagnostic: does replacing the scalar UV regression with a soft-argmax heatmap classification improve pelvis MPJPE?

### Design B — Soft-argmax UV heatmap + tight Gaussian (σ=1.0) + higher weight λ_heatmap=1.0

Same as Design A but with a tighter Gaussian target (σ=1.0 grid cell, roughly 16 pixels in crop space) and doubled heatmap loss weight. The tighter target provides sharper gradient signal toward the exact GT cell, encouraging the heatmap to learn a highly peaked distribution. The higher weight prioritizes the classification signal over the continuous soft-argmax regression, which can help when the heatmap distribution is noisy in early training.

Rationale: the BEDLAM2 pelvis position in the normalized crop space is very well defined (person-centred crops, so the pelvis is near the image centre and stable across frames). A tight Gaussian target σ=1.0 is appropriate; the soft-argmax's continuous output remains well-defined even for sharply-peaked distributions.

Config kwargs: `use_uv_heatmap=True`, `uv_heatmap_loss_weight=1.0`, `uv_heatmap_sigma=1.0`, `uv_heatmap_target='gaussian'`, `feat_h=40`, `feat_w=24`.

### Design C — Soft-argmax UV heatmap + learnable softmax temperature + KL loss λ_heatmap=0.5

Same as Design A but add a learnable scalar softmax temperature `self.uv_heatmap_temp = nn.Parameter(torch.tensor(1.0))` passed through `F.softplus` to ensure positivity. The softmax applied to `uv_logits` uses this learnable temperature:

```python
uv_attn = softmax(uv_logits / F.softplus(self.uv_heatmap_temp), dim=-1)
```

Initialization: `uv_heatmap_temp = 1.0` → softplus(1.0) ≈ 1.31, giving baseline-like sharpness. The model can learn to sharpen (lower temperature → peaked distribution → precise localization) or diffuse (higher temperature → broad distribution → robust under uncertainty) as needed.

This mirrors idea020's per-query temperature (for body joint cross-attention) but applied here to the UV heatmap. Since the pelvis is a single target, a scalar temperature (not per-query) is sufficient.

Config kwargs: `use_uv_heatmap=True`, `uv_heatmap_loss_weight=0.5`, `uv_heatmap_sigma=2.0`, `uv_heatmap_target='gaussian'`, `uv_heatmap_learnable_temp=True`, `feat_h=40`, `feat_w=24`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. One small helper is added to **`pelvis_utils.py`** for building the soft Gaussian target heatmap over the feature grid.

### `pelvis_utils.py`

Add a helper to project GT UV `[-1, 1]` normalized crop coordinates to feature-grid cell coordinates, and a helper to build a Gaussian target heatmap:

```python
def uv_to_grid_coords(uv_norm: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
    """Convert (u_norm, v_norm) in [-1, 1] to feature grid coordinates (h_frac, w_frac)."""
    u_grid = (uv_norm[..., 0] + 1.0) * 0.5 * feat_w   # (B,) or (J,)
    v_grid = (uv_norm[..., 1] + 1.0) * 0.5 * feat_h
    return torch.stack([v_grid, u_grid], dim=-1)  # (..., 2): (row, col)

def build_gaussian_heatmap_2d(
    center_hw: torch.Tensor,   # (B, 2) float row/col grid coords
    feat_h: int,
    feat_w: int,
    sigma: float,
) -> torch.Tensor:
    """Build a Gaussian target heatmap of shape (B, feat_h * feat_w), normalized to sum=1."""
    B = center_hw.shape[0]
    device = center_hw.device
    h_idx = torch.arange(feat_h, device=device, dtype=torch.float32)
    w_idx = torch.arange(feat_w, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')  # (H, W)
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)       # (H*W, 2)
    mu = center_hw.unsqueeze(1)                                     # (B, 1, 2)
    g = grid.unsqueeze(0)                                           # (1, H*W, 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)                             # (B, H*W)
    hm = torch.exp(-dist2 / (2.0 * sigma ** 2))
    hm = hm / hm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return hm  # (B, H*W)
```

### `pose3d_transformer_head.py`

**1. `__init__` additions**

```python
use_uv_heatmap: bool = False              # replace scalar UV regression with heatmap
uv_heatmap_loss_weight: float = 0.5
uv_heatmap_sigma: float = 2.0
uv_heatmap_target: str = 'gaussian'        # 'gaussian' only for this idea
uv_heatmap_learnable_temp: bool = False   # Design C
feat_h: int = 40
feat_w: int = 24
```

When `use_uv_heatmap=True`, replace `self.uv_out = nn.Linear(hidden_dim, 2)` with:
```python
self.uv_heatmap_proj = nn.Linear(hidden_dim, 1)
nn.init.zeros_(self.uv_heatmap_proj.weight)
nn.init.zeros_(self.uv_heatmap_proj.bias)
if uv_heatmap_learnable_temp:
    self.uv_heatmap_temp = nn.Parameter(torch.tensor(1.0))
```
Keep `self.uv_out = None` (or just unused) when heatmap is active — Designer should gate in forward().

**2. `forward()` — when `use_uv_heatmap=True`**

```python
# spatial is (B, H'*W', hidden_dim) from before decoder
uv_logits = self.uv_heatmap_proj(spatial).squeeze(-1)   # (B, H'*W')
if self.uv_heatmap_learnable_temp:
    temp = F.softplus(self.uv_heatmap_temp)
    uv_attn = F.softmax(uv_logits / temp, dim=-1)       # (B, H'*W')
else:
    uv_attn = F.softmax(uv_logits, dim=-1)

# soft-argmax to continuous (u_norm, v_norm)
H, W = self.feat_h, self.feat_w
h_idx = torch.arange(H, device=spatial.device, dtype=spatial.dtype)   # (H,)
w_idx = torch.arange(W, device=spatial.device, dtype=spatial.dtype)   # (W,)
attn_hw = uv_attn.view(-1, H, W)                                       # (B, H, W)
v_frac = (attn_hw.sum(dim=-1) * h_idx).sum(dim=-1) / (H - 1)           # (B,) in [0,1]
u_frac = (attn_hw.sum(dim=-2) * w_idx).sum(dim=-1) / (W - 1)           # (B,) in [0,1]
pelvis_uv = torch.stack([u_frac * 2.0 - 1.0, v_frac * 2.0 - 1.0], dim=-1)  # (B, 2)

# Store the distribution for loss()
self._uv_attn = uv_attn      # (B, H'*W')
```

**Designer note**: the normalized UV convention in BEDLAM2 is `u_norm = u_pixel / crop_w * 2 - 1`, `v_norm = v_pixel / crop_h * 2 - 1`. The feature grid is `(H', W') = (40, 24)` corresponding to crop (640, 384). The Designer MUST verify the row/col convention by inspecting how the spatial tokens are flattened in the baseline `forward()`: `spatial = feat.flatten(2).transpose(1, 2)` with feat shape `(B, C, H, W)` → row-major order with H outer, W inner. So `spatial[b, h*W + w]` is the token at (row=h, col=w). The `attn_hw.view(-1, H, W)` reshape matches this convention. The mapping from grid-cell index to normalized UV is linear: `u_norm = 2 * (w_idx / (W-1)) - 1`, `v_norm = 2 * (h_idx / (H-1)) - 1`. This is the exact inverse of `uv_to_grid_coords` above. **The Designer MUST test this mapping on a few GT samples before training.**

When `use_uv_heatmap=False` (baseline):
```python
pelvis_uv = self.uv_out(pelvis_token)  # unchanged baseline path
```

**3. `loss()` additions — when `use_uv_heatmap=True`**

After the existing continuous-UV loss (SmoothL1 on `pred['pelvis_uv']` vs `gt_uv`, weight 1.0) which remains active:

```python
if self.use_uv_heatmap and self.uv_heatmap_loss_weight > 0.0:
    # Build GT Gaussian heatmap target
    gt_grid = uv_to_grid_coords(gt_uv, self.feat_h, self.feat_w)   # (B, 2): (row, col)
    gt_hm = build_gaussian_heatmap_2d(gt_grid, self.feat_h, self.feat_w, self.uv_heatmap_sigma)  # (B, H*W)
    # KL divergence: KL(gt || pred) = sum(gt * (log(gt) - log(pred)))
    # Numerically stable via log-softmax of logits
    log_attn = torch.log(self._uv_attn.clamp(min=1e-8))
    heatmap_loss = -(gt_hm * log_attn).sum(dim=-1).mean()   # cross-entropy form
    losses['loss/uv_heatmap/train'] = self.uv_heatmap_loss_weight * heatmap_loss
    self._uv_attn = None  # clear stale reference
```

**4. `predict()` — no changes beyond `forward()` routing**

`predict()` calls `forward()` which already returns the correct `pelvis_uv` tensor when `use_uv_heatmap=True`. No other changes.

### `config.py`

**Design A:**
```python
use_uv_heatmap=True,
uv_heatmap_loss_weight=0.5,
uv_heatmap_sigma=2.0,
uv_heatmap_target='gaussian',
uv_heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

**Design B:**
```python
use_uv_heatmap=True,
uv_heatmap_loss_weight=1.0,
uv_heatmap_sigma=1.0,
uv_heatmap_target='gaussian',
uv_heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

**Design C:**
```python
use_uv_heatmap=True,
uv_heatmap_loss_weight=0.5,
uv_heatmap_sigma=2.0,
uv_heatmap_target='gaussian',
uv_heatmap_learnable_temp=True,
feat_h=40,
feat_w=24,
```

All values are bool/int/float/str literals. No Python `import` statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — mpjpe_pelvis_val**: the 2D heatmap classification provides dense per-cell gradient to the spatial tokens themselves, rather than backpropagating only through the single pelvis-token embedding. This is a much denser supervision signal for "where is the pelvis in the image". Target: `mpjpe_pelvis_val < 580` at stage-1 (vs. baseline 652, best prior 608 from idea023), and `< 300` at stage-2 (vs. baseline 365, best prior 308 from idea023).

- **Secondary gain — composite_val**: via the 0.33 weight on pelvis MPJPE. Target: `composite_val < 320` at stage-1 (improving best prior 323.75).

- **Tertiary gain — mpjpe_abs_val**: the pelvis UV is a direct input to `recover_pelvis_3d`, which is used in the absolute 3D pelvis position. Improving UV accuracy directly improves `mpjpe_abs`. Target: `mpjpe_abs_val < 750` at stage-1 (vs. baseline 833), `< 550` at stage-2.

- **Design A** (σ=2, λ=0.5): diagnostic. Soft Gaussian target with moderate weight; expected to at least match the baseline continuous regression's pelvis MPJPE, with upside from the classification bias.

- **Design B** (σ=1, λ=1.0): aggressive sharpening and strong classification signal. Expected best pelvis MPJPE if the spatial features can produce reliable peaked distributions. Risk: the sharper target may amplify noise early in training.

- **Design C** (learnable temperature): gives the model freedom to adjust distribution sharpness per training stage. Expected most robust; may not be the absolute best if a fixed σ=1.0 is already optimal.

- **Composite target (stage-1)**: `< 320` (best case Design C or B).
- **Composite target (stage-2)**: `< 218`, breaking the 220.23 floor set by idea028/design003.

---

## Risk and Mitigation

- **Output semantics preserved**: `pred['pelvis_uv']` is still `(B, 2)` continuous in `[-1, 1]` — identical interface to baseline. `recover_pelvis_3d`, `compute_mpjpe_abs`, and `bedlam_metric.py` see no change.

- **Zero-init at start**: `uv_heatmap_proj` zero-init → uniform softmax over 960 cells → soft-argmax produces `(u_frac, v_frac) = (0.5, 0.5)` → `pelvis_uv = (0, 0)`. For BEDLAM2 person-centred crops, GT pelvis_uv is concentrated near (0, 0), so the zero-init start is close to the target distribution mean. Safe, gradient-informative starting point.

- **Gradient flow**: the 2D heatmap receives gradient from two sources: (a) the continuous SmoothL1 loss via soft-argmax's differentiable expectation, and (b) the KL classification loss directly on the spatial distribution. Both gradients flow back to the spatial tokens via `uv_heatmap_proj`, which also contains gradient from all other losses (joint, depth, body_reprojection, etc., through shared backbone features). No new parameters for Design A/B beyond the 256-dim linear head replacing a 512-dim regression head (actually fewer params).

- **Parameter count delta**: baseline `self.uv_out = Linear(256, 2)` has 256*2 + 2 = 514 params. This idea's `uv_heatmap_proj = Linear(256, 1)` has 256 + 1 = 257 params. Net: −257 params for Design A/B; +1 param for Design C (scalar temperature). Negligible either way.

- **Softmax numerical stability**: at FP16 under AMP, the 960-way softmax is numerically safe — logits are initialized to 0 (uniform), and even after training rarely produce logits outside [-10, 10]. `log(uv_attn).clamp(min=1e-8)` prevents `log(0)` on any cell pushed to extreme values. Standard practice.

- **Feature-grid convention**: the Designer MUST verify `feat.flatten(2).transpose(1, 2)` row-major ordering. Specifically, `spatial[b, k, :]` with `k = h * W + w` must correspond to grid cell `(h, w)`. This is the standard PyTorch convention and matches the baseline's positional encoding implementation. Verify once by evaluating a debug batch with known-position GT and confirming the peak heatmap cell matches.

- **Interaction with idea014 (depth classification)**: fully composable. idea014 replaces `Linear(hidden_dim, 1)` depth regression with binned classification on the pelvis token. idea031 replaces `Linear(hidden_dim, 2)` UV regression with heatmap classification on spatial tokens. The two target different output heads and different feature sources (pelvis token vs spatial tokens). A future combined idea is recommended.

- **Interaction with idea023 (heatmap-guided query init)**: both use 2D heatmaps over the H'×W' grid. idea023 uses per-joint heatmaps (22 heatmaps) to initialize body joint queries. idea031 uses a single pelvis heatmap for the UV output. The `uv_heatmap_proj` and idea023's `heatmap_proj` are separate Linear layers with non-overlapping outputs. Composition is straightforward: idea023 takes 22 spatial-token pooling weights, idea031 adds a 23rd (for pelvis). A natural future combined design.

- **Interaction with idea028 (decoupled pelvis queries)**: idea028 routes UV/depth prediction through a dedicated pelvis query's token. idea031 replaces the UV head's *output* mechanism (scalar regression → heatmap classification) but reads logits from spatial tokens, not from a specific query. Combining idea028 + idea031 would require choosing whether the UV heatmap reads from spatial tokens (this idea's default) or from the pelvis query via a different projection. Composable but requires a design decision from the Designer for a future combined idea. In isolation (this idea only), the UV head reads from spatial tokens — same source as the cross-attention keys the decoder uses.

- **Interaction with idea002 (dedicated pelvis query)**: similar to idea028, the architectural pelvis decoupling is orthogonal to the output-representation change.

- **Memory**: `uv_logits` is `(B=4, H'W'=960)` ≈ 7.5 KB float16. `uv_attn` same. `gt_hm` same. `gaussian_target` construction: one `meshgrid(40, 24)` tensor computed per loss call, ~4 KB. All negligible.

- **Speed**: one additional `Linear(256, 1)` on 960 tokens = 256k multiplies per sample = ~1M per batch. One softmax over 960 entries per sample = ~3840 exp() calls. Soft-argmax: two reductions over (40, 24) grid. Total per-step overhead < 0.5 ms on 2080 Ti. Negligible.

- **AMP / float16 safety**: `uv_heatmap_proj(spatial)` is a standard linear; `softmax` is standard (both AMP-friendly). `F.log` on clamped probabilities is safe. The soft-argmax reduction produces `(B, 2)` floats bounded in `[-1, 1]`. No overflow/underflow risk.

- **MMEngine config constraint**: all new kwargs are bool/int/float/str literals. No Python `import` statements. Fully compliant.

- **Feature map assumes feat_h=40, feat_w=24**: if the backbone stride or input size ever changes, these literals would need updating. For this project the backbone+input are invariants, so fixed literals are safe. Designer should add an `assert H == feat_h and W == feat_w` check in `forward()` to catch any future inconsistency early.

- **Training dynamics**: the KL heatmap loss produces gradients of order O(1) in the logits regime; at weight 0.5 it is comparable in magnitude to the SmoothL1 UV loss (which also produces O(1) gradients at the beta=0.05 knee). No gradient-scale pathology expected. If training diverges in Design B (λ=1.0, σ=1.0), the Designer should fall back to Design A settings.
