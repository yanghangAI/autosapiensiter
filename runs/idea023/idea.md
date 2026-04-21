**Idea Name:** Heatmap-Guided Spatial Token Pooling for Query Initialization

**Approach:** Predict a per-joint soft 2D attention heatmap over the spatial feature grid using a lightweight linear projection of the backbone features, then use that heatmap as a soft-pooling weight to compute a per-joint content-aware feature vector that is added to the static joint query embedding before the transformer decoder — giving each query a joint-specific image region summary as a warm-start, supervised by a 2D heatmap loss against GT 2D projections of the body joints.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Static Query Bottleneck

The baseline transformer decoder begins with 70 static joint query embeddings (from `nn.Embedding`). These embeddings carry structural/skeletal priors learned during training, but they are **completely image-agnostic**: all 70 queries are identical for every image in the batch. The first decoder layer's cross-attention must solve, in a single pass, both (a) spatial routing — figuring out which region of the 40×24 spatial token grid is relevant for joint `i` — and (b) feature extraction — reading the right appearance signal once it has found the right region.

Prior ideas have addressed this bottleneck in different ways:

| Idea | Strategy | Limitation |
|---|---|---|
| idea001 | More decoder layers (static queries) | More capacity, no spatial prior |
| idea003 | Content-adaptive query init via global average pool | Single global vector per image — no joint-specific routing |
| idea019 | Deformable sampling: predict sparse sampling offsets | No explicit 2D supervision; offsets must be learned purely from 3D loss |
| idea021 | Static learnable cross-attn bias per joint | Learned average prior; same for all images; no content adaptation |
| idea022 | Dynamic Gaussian bias from 3D intermediate predictions | Requires 3D intermediate estimates; builds bias mid-forward-pass; noisy early in training |

**None of these ideas use an intermediate 2D heatmap as an explicit, supervised routing signal.** A 2D heatmap is:
1. Directly computable as a lightweight `Linear(embed_dim, num_body_joints)` applied at each spatial token.
2. Directly supervisable with a strong, accurate signal: the GT 2D joint locations in crop coordinates, computed by projecting GT 3D absolute joints through K (identical to the projection used in idea010's reprojection loss).
3. Fully differentiable: the heatmap is a softmax over spatial tokens, which can be backpropagated through.

### What this idea proposes

**Stage 1 — Heatmap prediction:**
For each image, apply a small linear head to the projected spatial tokens (shape `(B, H'W', hidden_dim)`) to predict per-joint logit maps, producing `(B, H'W', num_body_joints=22)`. Taking softmax over the spatial dimension gives a per-joint soft attention map `A ∈ R^{B × 22 × H'W'}`, where `A[b, j, :]` is a probability distribution over spatial locations indicating where joint `j` is expected in the feature grid.

**Stage 2 — Content-aware query warm-start:**
Use the soft attention map to pool a joint-specific feature vector:
```
z_j = sum_k A[b, j, k] * spatial_tokens[b, k]    (B, 22, hidden_dim)
```
This produces a per-joint summary of the most relevant spatial features for that joint. Add this to the static joint query embedding:
```
queries[:, :22] = queries[:, :22] + z_j     # content-adaptive warm-start for body joints
queries[:, 22:] = queries[:, 22:]            # hand queries unchanged (zero delta)
```

**Stage 3 — Heatmap supervision:**
Supervise the heatmap logits with a GT heatmap computed from 2D projections of GT absolute joint positions. The GT heatmap is a one-hot or Gaussian-spread distribution over the 40×24 spatial grid centred on each joint's projected 2D location. Loss: cross-entropy (hard target, one-hot) or KL divergence (soft Gaussian target with σ=2 grid cells).

The heatmap loss weight is small (λ=0.1 to 0.3) to avoid overriding the primary 3D joint regression signal. The heatmap module `heatmap_proj: Linear(hidden_dim, 22)` is initialized to zero output so the model starts at exactly baseline behaviour (zero spatial routing bias, queries start at static embeddings).

### Why this is different from all prior ideas

This is the **first idea to use an explicit 2D heatmap intermediate representation with direct GT supervision for cross-attention spatial routing**:

- **vs. idea003 (content-adaptive init)**: idea003 uses global average pooling — a single shared vector per image — and adds it to all queries. This idea produces a per-joint soft-pooled feature vector, directly routing each query to its expected image region. The key difference: joint-specific routing vs. global image statistics.
- **vs. idea019 (deformable sampling)**: idea019 predicts sparse 2D offsets for each query (2–4 sample points), unsupervised. This idea uses a dense soft heatmap over all 960 tokens, supervised by 2D projections of GT 3D joints. The supervision signal is much stronger and more direct.
- **vs. idea021 (static cross-attn bias)**: idea021 adds a static per-joint bias to cross-attention logits — no image adaptation, no GT supervision. This idea's heatmap is computed from the actual backbone features and supervised with GT 2D joint positions — fully image-specific and supervised.
- **vs. idea022 (dynamic reprojection bias)**: idea022 computes a dynamic Gaussian bias from 3D *intermediate predictions* in a cascaded decoder. This idea computes the routing signal from backbone features in a single-pass, before the decoder, and supervises it with GT 2D data. No cascaded decoding required.

### Grounding in observed results

- **idea003/design002**: best improvement from query initialization (composite 345.35, body MPJPE 190.9mm at stage-1; stage-2 composite 225.44 — competitive with best). The content-adaptive init direction is validated but limited by using a global average feature. Per-joint spatial pooling is a direct strengthening of this direction.
- **idea008/design002**: the body-only decoder dramatically improved `mpjpe_rel_val` (362mm vs. baseline 438mm) by removing hand queries. This confirmed that query-space pollution is a real bottleneck. Combining body-focused attention routing with accurate per-joint heatmaps should further reduce relative error.
- **idea010/design002**: 2D reprojection loss improved stage-2 body MPJPE to 168.79mm (best among all designs). The 2D GT projections from K are a strong supervision signal. This idea reuses the same 2D projection machinery, but as a routing signal rather than a loss term — complementary, not redundant.
- **mpjpe_rel_val plateau**: the best relative MPJPE at stage-1 is 362mm (idea008/design002), compared to baseline 438mm. The improvement came from removing query contamination, not from better spatial routing. Joint-specific heatmap routing is the next logical step to push relative error further.

---

## Proposed Variations

### Design A — Hard one-hot heatmap target, small weight (minimal supervision)

Add `heatmap_proj = nn.Linear(hidden_dim, 22)` to the head, initialized to all-zeros so that at training start, all spatial attention weights are uniform (equal soft-pooling over the full spatial grid — same as idea003's global average pool). The GT heatmap is a hard one-hot over the nearest grid cell to each GT 2D joint projection. Loss: cross-entropy with weight `λ_hm = 0.1`.

At the softmax temperature for the pooling attention, use `temperature=1.0` (standard softmax), allowing the model to sharpen or diffuse the attention map freely.

Config kwargs: `use_heatmap_init=True`, `heatmap_loss_weight=0.1`, `heatmap_target='onehot'`, `heatmap_temperature=1.0`.

### Design B — Soft Gaussian heatmap target, moderate weight

Same architecture as Design A but the GT heatmap is a Gaussian distribution over the 40×24 grid centred on the projected 2D joint location with σ=2 grid cells (≈ 32 pixels at the 640×384 input scale). Loss: KL divergence (between softmax of predicted logits and the Gaussian GT distribution), weight `λ_hm = 0.2`.

The Gaussian target provides smoother gradients near the correct location and better handles the case where a joint projects near the boundary of two grid cells. It also provides a non-zero target over neighbouring cells, regularizing the heatmap to be spatially coherent.

Config kwargs: `use_heatmap_init=True`, `heatmap_loss_weight=0.2`, `heatmap_target='gaussian'`, `heatmap_sigma=2.0`, `heatmap_temperature=1.0`.

### Design C — Soft Gaussian target + learnable temperature per joint

Same as Design B but add a learnable per-joint softmax temperature scalar `self.heatmap_temp = nn.Parameter(torch.ones(22))`, passed through `F.softplus` to ensure positivity. This allows the model to learn sharp, highly focused heatmaps for easy-to-locate joints (e.g., pelvis, spine) and diffuse heatmaps for harder joints (e.g., wrists). Initialization: `heatmap_temp=1.0` (identical to Design B start). Loss weight: `λ_hm = 0.2`.

This mirrors the motivation from idea020 (per-query temperature for cross-attention sharpness) but applies the temperature to the heatmap pooling attention rather than the main decoder cross-attention — orthogonal and composable.

Config kwargs: `use_heatmap_init=True`, `heatmap_loss_weight=0.2`, `heatmap_target='gaussian'`, `heatmap_sigma=2.0`, `heatmap_learnable_temp=True`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. A small helper function is added to **`pelvis_utils.py`**.

### `pelvis_utils.py`

Add a new helper `project_joints_to_grid_coords(joints_abs, K, crop_h, crop_w, feat_h=40, feat_w=24)`:

```python
def project_joints_to_grid_coords(
    joints_abs: torch.Tensor,  # (J, 3) absolute camera-frame joints
    K: np.ndarray,             # (3, 3) crop intrinsic
    crop_h: int,
    crop_w: int,
    feat_h: int = 40,
    feat_w: int = 24,
) -> torch.Tensor:
    """Project absolute 3D joints to feature grid (h, w) coordinates.

    Returns:
        (J, 2) float tensor: (h_frac, w_frac) in feature grid units.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = joints_abs[:, 0].clamp(min=0.01)
    Y = joints_abs[:, 1]
    Z = joints_abs[:, 2]
    u_px = -Y / X * fx + cx  # pixel u in crop
    v_px = -Z / X * fy + cy  # pixel v in crop
    h_frac = v_px / crop_h * feat_h
    w_frac = u_px / crop_w * feat_w
    return torch.stack([h_frac, w_frac], dim=-1)  # (J, 2)
```

### `pose3d_transformer_head.py`

**1. New module-level helper: `_build_gaussian_heatmap_target`**

```python
def _build_gaussian_heatmap_target(
    joint_grid_coords: torch.Tensor,  # (J, 2) in grid units
    feat_h: int,
    feat_w: int,
    sigma: float,
) -> torch.Tensor:
    """Build soft Gaussian heatmap target over the spatial grid.

    Returns:
        (J, feat_h * feat_w) float tensor, normalised to sum to 1.
    """
    J = joint_grid_coords.shape[0]
    device = joint_grid_coords.device
    gh = torch.arange(feat_h, device=device, dtype=torch.float32)
    gw = torch.arange(feat_w, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(gh, gw, indexing='ij')  # (H', W')
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)  # (H'W', 2)

    mu = joint_grid_coords.unsqueeze(1)  # (J, 1, 2)
    g = grid.unsqueeze(0)                # (1, H'W', 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)  # (J, H'W')
    heatmap = torch.exp(-dist2 / (2.0 * sigma ** 2))
    # Normalise to probability distribution
    heatmap = heatmap / (heatmap.sum(dim=-1, keepdim=True).clamp(min=1e-6))
    return heatmap  # (J, H'W')
```

**2. `Pose3dTransformerHead.__init__` additions**

New kwargs with defaults matching baseline behaviour (all False/None = baseline):
```python
use_heatmap_init: bool = False       # enable heatmap pooling module
heatmap_loss_weight: float = 0.1     # λ for heatmap loss
heatmap_target: str = 'onehot'       # 'onehot' or 'gaussian'
heatmap_sigma: float = 2.0           # Gaussian σ in grid cells (Design B/C)
heatmap_temperature: float = 1.0     # softmax temperature for pooling attention
heatmap_learnable_temp: bool = False  # per-joint learnable temperature (Design C)
feat_h: int = 40
feat_w: int = 24
```

When `use_heatmap_init=True`:
```python
self.heatmap_proj = nn.Linear(hidden_dim, 22)
nn.init.zeros_(self.heatmap_proj.weight)
nn.init.zeros_(self.heatmap_proj.bias)

if heatmap_learnable_temp:
    self.heatmap_temp = nn.Parameter(torch.ones(22))
```

**3. `forward()` additions**

After computing `spatial = spatial + pos_enc` and before the decoder:

```python
if self.use_heatmap_init:
    # heatmap_logits: (B, H'W', 22) — one score per spatial token per body joint
    heatmap_logits = self.heatmap_proj(spatial)  # (B, H'W', 22)

    # Apply per-joint temperature
    if self.heatmap_learnable_temp:
        temp = F.softplus(self.heatmap_temp).view(1, 1, 22)  # (1, 1, 22)
    else:
        temp = self.heatmap_temperature

    # Soft attention over spatial tokens: (B, 22, H'W')
    attn_weights = F.softmax(heatmap_logits.transpose(1, 2) / temp, dim=-1)

    # Soft pooling: (B, 22, hidden_dim)
    pooled_features = torch.bmm(attn_weights, spatial)

    # Add to body joint queries (indices 0-21), leave hand queries unchanged
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1).clone()
    queries[:, :22] = queries[:, :22] + pooled_features
else:
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)

# Store heatmap_logits for loss() to access
self._heatmap_logits = heatmap_logits if self.use_heatmap_init else None
```

**Note on `expand` vs `clone`**: the baseline uses `expand` (no-copy). When adding `pooled_features`, the result must be writeable. The Designer should use `queries = queries + F.pad(pooled_features, (0, 0, 0, 48))` (zero-pad to full 70 joints) instead of modifying in-place, to avoid the expand-copy issue cleanly:

```python
pad = torch.zeros(B, 48, self.hidden_dim, device=spatial.device, dtype=spatial.dtype)
delta = torch.cat([pooled_features, pad], dim=1)  # (B, 70, hidden_dim)
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1) + delta
```

The zero-init on `heatmap_proj` ensures `heatmap_logits ≡ 0` at training start → uniform `attn_weights = 1/960` → `pooled_features = mean(spatial)` = global average pool = exactly idea003/design001's behaviour. This is a stronger warm-start than the zero-delta baseline, but still safe (the model already handles global average pooling via idea003's precedent). After a few epochs, the model sharpens the heatmap toward the correct joint regions.

**4. `loss()` additions**

After assembling `pred = self.forward(feats)` and GT tensors:

```python
if self.use_heatmap_init and self._heatmap_logits is not None:
    # Build GT grid coordinates per sample per body joint
    heatmap_loss = 0.0
    B = len(batch_data_samples)
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h, crop_w = int(img_shape[0]), int(img_shape[1])

        # GT absolute joints for sample i (body joints 0-21)
        gt_pelvis_3d = recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
        gt_abs_joints = gt_joints[i, :22] + gt_pelvis_3d         # (22, 3)

        # Project to feature grid coordinates
        grid_coords = project_joints_to_grid_coords(
            gt_abs_joints, K, crop_h, crop_w, self.feat_h, self.feat_w)  # (22, 2)

        if self.heatmap_target == 'onehot':
            # Hard one-hot: nearest grid cell
            h_idx = grid_coords[:, 0].long().clamp(0, self.feat_h - 1)
            w_idx = grid_coords[:, 1].long().clamp(0, self.feat_w - 1)
            target_idx = h_idx * self.feat_w + w_idx               # (22,)
            # Cross-entropy: logits (H'W', 22) → target (22,) indices
            logits_i = self._heatmap_logits[i].T  # (22, H'W')
            heatmap_loss = heatmap_loss + F.cross_entropy(logits_i, target_idx)
        else:
            # Soft Gaussian target
            gt_hm = _build_gaussian_heatmap_target(
                grid_coords, self.feat_h, self.feat_w, self.heatmap_sigma)  # (22, H'W')
            logits_i = self._heatmap_logits[i].T   # (22, H'W')
            log_probs = F.log_softmax(logits_i, dim=-1)              # (22, H'W')
            heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum()

    losses['loss/heatmap/train'] = self.heatmap_loss_weight * heatmap_loss / B
    self._heatmap_logits = None  # clear to avoid stale reference
```

### `config.py`

**Design A:**
```python
use_heatmap_init=True,
heatmap_loss_weight=0.1,
heatmap_target='onehot',
heatmap_temperature=1.0,
heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

**Design B:**
```python
use_heatmap_init=True,
heatmap_loss_weight=0.2,
heatmap_target='gaussian',
heatmap_sigma=2.0,
heatmap_temperature=1.0,
heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

**Design C:**
```python
use_heatmap_init=True,
heatmap_loss_weight=0.2,
heatmap_target='gaussian',
heatmap_sigma=2.0,
heatmap_temperature=1.0,
heatmap_learnable_temp=True,
feat_h=40,
feat_w=24,
```

All values are bool/int/float/str literals. No Python import statements. MMEngine config constraint fully satisfied.

---

## Expected Outcome

- **Primary gain — mpjpe_rel_val (body relative MPJPE)**: the heatmap routing gives each body query a joint-specific feature summary before entering the decoder, focusing cross-attention to the correct image region from the start. Expected improvement: `mpjpe_rel_val < 400` at stage-1 (vs. baseline 438, best prior 362 from idea008). The per-joint routing is more targeted than idea003's global pooling and more directly supervised than idea019's unsupervised deformable offsets.

- **Secondary gain — mpjpe_body_val**: better spatial routing produces better 3D joint regression quality. Target: `mpjpe_body_val < 188` at stage-1, `< 170` at stage-2 (matching best prior from idea010/design002).

- **Pelvis MPJPE**: the heatmap module operates only on body joint queries (0–21); the pelvis depth/UV are regressed from query token 0 which also benefits from improved pooling. Expected mild positive effect on `mpjpe_pelvis_val`.

- **Design A** (hard one-hot, λ=0.1): diagnostic — does direct 2D supervision of the routing heatmap improve 3D predictions? Conservative weight avoids destabilising 3D regression. Expected composite_val < 345 at stage-1.

- **Design B** (soft Gaussian, λ=0.2): smoother gradients near the correct location; better handles joints at grid-cell boundaries. Expected to outperform Design A by providing more informative gradient signal. Expected composite_val < 340.

- **Design C** (learnable temperature): per-joint temperature allows narrow attention for easy joints (pelvis, spine) and broad for hard joints (wrists, ankles). Highest potential. Expected composite_val < 335.

- **Composite target (stage-1)**: `composite_val < 335`, improving on best prior stage-1 of 328.14 (idea013/design003).
- **Composite target (stage-2)**: `composite_val < 220`, competitive with best prior stage-2 of 224.52 (idea001/design001).

---

## Risk and Mitigation

- **Zero-init warm-start produces global pooling**: at training start, `heatmap_logits ≡ 0` → uniform `attn_weights = 1/960` → `pooled_features = mean(spatial)`. This is exactly equivalent to a global average pool added to each body query — same as idea003/design001's approach, which produced stable training. The model starts from a safe, reasonable initialisation and progressively sharpens the heatmap via the supervision signal.

- **GT heatmap requires `gt_abs_joints`**: the GT absolute joints are assembled by `recover_pelvis_3d(gt_depth, gt_uv, K) + gt_joints`. This requires K (always present) and gt_depth/gt_uv (already extracted in loss()). The per-sample loop mirrors the existing `compute_mpjpe_abs` structure — no new infrastructure.

- **Joints projected outside the feature grid**: joints near image boundaries or behind the camera may project to negative or out-of-bounds grid coordinates. For hard one-hot targets, clamp `h_idx` and `w_idx` to `[0, feat_h-1]` and `[0, feat_w-1]`. For Gaussian targets, the Gaussian naturally decays away from the boundary; the normalisation handles boundary effects.

- **Feature grid orientation consistency**: the spatial tokens are flattened as `feat.flatten(2).transpose(1, 2)` where `feat` is `(B, C, H=40, W=24)` — row-major order (H outer, W inner). The helper `project_joints_to_grid_coords` must output `(h_frac, w_frac)` in the same ordering, and `_build_gaussian_heatmap_target` must use `torch.meshgrid(..., indexing='ij')`. The feat_h=40, feat_w=24 must be confirmed by the Designer (img_h=640, img_w=384, stride=16 → H'=40, W'=24 ✓).

- **Memory**: `heatmap_logits` is `(B, H'W', 22) = (4, 960, 22)` = ~84K float16 values ≈ 168 KB. `attn_weights` is `(B, 22, H'W') = (4, 22, 960)` ≈ 168 KB. `pooled_features` is `(B, 22, 256)` ≈ 44 KB. Total additional memory: < 1 MB. Negligible on the 2080 Ti.

- **Speed**: the `heatmap_proj` linear is `(B*H'W', 22) = (3840, 22)` multiply — < 0.1 ms on 2080 Ti. The `bmm` for soft pooling is `(B=4, 22, 960) × (B=4, 960, 256)` = 4 × 22 × 256 × 960 ≈ 21M multiply-adds ≈ 0.2 ms. The per-sample GT projection loop is identical overhead to `compute_mpjpe_abs`. Net per-step overhead: < 0.5 ms, negligible.

- **`_heatmap_logits` side-channel**: storing heatmap logits on `self` for the loss() to access follows the same pattern as `self._train_mpjpe` in the baseline and `self._reproj_bias` in idea022. It is cleared after loss() reads it. The Designer must ensure that `predict()` does not erroneously read `_heatmap_logits` — this is safe since `predict()` calls `forward()` then never reads loss-side attributes.

- **Interaction with idea003 (content-adaptive init)**: idea003 adds a MLP-projected global pooling delta to all queries. This idea adds a heatmap-pooled per-joint delta to body queries. The mechanisms are additive and composable. If either idea performs well in isolation, combining them is a natural future experiment.

- **Interaction with idea008 (body-only decoder)**: reducing queries to 22 body joints is fully compatible with this idea (heatmap only applies to body joints anyway). Composing the two ideas would require adjusting the query indexing but is straightforward. Left for a future combined idea.

- **AMP / float16 safety**: the Gaussian heatmap computation uses exp() — values will be in (0, 1] so no overflow. The log_softmax for KL divergence is numerically stable (log_softmax does not overflow for typical logit ranges). The Designer should add a small epsilon to the KL divergence denominator for the one-hot case if needed.

- **MMEngine config constraint**: all new kwargs are bool/int/float/str literals. No Python import statements in config. Fully compliant.
