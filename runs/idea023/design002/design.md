# Design 002 — Heatmap-Guided Query Init: Soft Gaussian Target (λ=0.2)

**Design Description:** Same heatmap-pooled query warm-start as design001, but replace hard one-hot cross-entropy with KL divergence against a Gaussian heatmap target (σ=2 grid cells) at weight 0.2, providing smoother gradient signal near joint locations and better handling of grid-cell boundary cases.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pelvis_utils.py` — add `project_joints_to_grid_coords` helper (identical to design001)
2. `pose3d_transformer_head.py` — add heatmap projection, pooling, Gaussian target builder, and KL loss
3. `config.py` — add new head kwargs

---

## Algorithm

The heatmap-guided query warm-start algorithm is identical to design001 for the forward pass, differing only in the loss supervision:

1. **Heatmap prediction**: `heatmap_proj: Linear(256, 22)` applied to each spatial token → `heatmap_logits (B, 960, 22)`.
2. **Soft attention pooling**: softmax over spatial dimension (temperature=1.0) → `attn_weights (B, 22, 960)`. BMM with spatial tokens → `pooled_features (B, 22, 256)`.
3. **Query warm-start**: zero-pad to `(B, 70, 256)`, add to static queries. Body joints (0–21) get per-joint routed features; hand joints (22–69) unchanged.
4. **Gaussian target construction**: for each GT body joint, project 3D absolute position through K to pixel coordinates, convert to feature grid units `(h_frac, w_frac)`, build a Gaussian probability distribution over all 960 tokens with σ=2 grid cells.
5. **KL loss**: cross-entropy between `log_softmax(heatmap_logits)` and the Gaussian target, summed over spatial tokens and averaged over joints and batch. Scaled by λ=0.2.

The algorithmic motivation for Gaussian over one-hot: joints that project near a grid-cell boundary get gradient signal from both neighbouring cells, making the supervision smoother and reducing the sensitivity to quantisation of the grid. The σ=2 grid cells (≈32px at 640×384 input) is wide enough to cover typical quantisation error but narrow enough to be informative.

---

## 1. `pelvis_utils.py` Changes

**Identical to design001.** Add `project_joints_to_grid_coords` after `compute_mpjpe_abs`:

```python
def project_joints_to_grid_coords(
    joints_abs: 'torch.Tensor',
    K: 'np.ndarray',
    crop_h: int,
    crop_w: int,
    feat_h: int = 40,
    feat_w: int = 24,
) -> 'torch.Tensor':
    """Project absolute 3D joints to feature grid (h, w) coordinates.

    BEDLAM2 projection convention: u = fx*(-Y/X) + cx, v = fy*(-Z/X) + cy

    Args:
        joints_abs: (J, 3) absolute camera-frame joints [X, Y, Z] in metres.
        K: (3, 3) crop intrinsic matrix (numpy float32).
        crop_h: Crop height in pixels.
        crop_w: Crop width in pixels.
        feat_h: Feature grid height (default 40).
        feat_w: Feature grid width (default 24).

    Returns:
        (J, 2) float tensor: (h_frac, w_frac) in feature grid units.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    X = joints_abs[:, 0].clamp(min=0.01)
    Y = joints_abs[:, 1]
    Z = joints_abs[:, 2]

    u_px = -Y / X * fx + cx
    v_px = -Z / X * fy + cy

    h_frac = v_px / crop_h * feat_h
    w_frac = u_px / crop_w * feat_w

    return torch.stack([h_frac, w_frac], dim=-1)  # (J, 2)
```

---

## 2. `pose3d_transformer_head.py` Changes

### 2a. Imports

Same as design001 — replace the existing `pelvis_utils` import line with:

```python
import numpy as np
import torch.nn.functional as F
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          project_joints_to_grid_coords as _project_joints_to_grid_coords,
                          recover_pelvis_3d as _recover_pelvis_3d)
```

### 2b. Add module-level helper `_build_gaussian_heatmap_target`

Insert the following function **before** the `Pose3dTransformerHead` class definition (e.g., after `_build_2d_sincos_pos_enc` and before `_DecoderLayer`):

```python
def _build_gaussian_heatmap_target(
    joint_grid_coords: torch.Tensor,
    feat_h: int,
    feat_w: int,
    sigma: float,
) -> torch.Tensor:
    """Build soft Gaussian heatmap target over the spatial grid.

    Args:
        joint_grid_coords: (J, 2) joint positions in feature grid units (h, w).
        feat_h: Feature grid height.
        feat_w: Feature grid width.
        sigma: Gaussian standard deviation in grid cells.

    Returns:
        (J, feat_h * feat_w) float tensor, normalised to sum to 1 per joint.
    """
    device = joint_grid_coords.device
    gh = torch.arange(feat_h, device=device, dtype=torch.float32)
    gw = torch.arange(feat_w, device=device, dtype=torch.float32)
    # indexing='ij': grid_h varies over rows, grid_w varies over columns
    grid_h, grid_w = torch.meshgrid(gh, gw, indexing='ij')  # each (feat_h, feat_w)
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)  # (feat_h*feat_w, 2)

    mu = joint_grid_coords.unsqueeze(1)  # (J, 1, 2)
    g = grid.unsqueeze(0)                # (1, feat_h*feat_w, 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)  # (J, feat_h*feat_w)

    heatmap = torch.exp(-dist2 / (2.0 * sigma ** 2))
    # Normalise each joint's distribution to sum to 1
    heatmap = heatmap / heatmap.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return heatmap  # (J, feat_h*feat_w)
```

### 2c. `__init__` parameters

Add to `Pose3dTransformerHead.__init__` after `loss_weight_uv`:

```python
use_heatmap_init: bool = False,
heatmap_loss_weight: float = 0.1,
heatmap_target: str = 'onehot',
heatmap_sigma: float = 2.0,
heatmap_temperature: float = 1.0,
heatmap_learnable_temp: bool = False,
feat_h: int = 40,
feat_w: int = 24,
```

Store them inside `__init__` after `self.loss_weight_uv = loss_weight_uv`:

```python
self.use_heatmap_init = use_heatmap_init
self.heatmap_loss_weight = heatmap_loss_weight
self.heatmap_target = heatmap_target
self.heatmap_sigma = heatmap_sigma
self.heatmap_temperature = heatmap_temperature
self.heatmap_learnable_temp = heatmap_learnable_temp
self.feat_h = feat_h
self.feat_w = feat_w
self._heatmap_logits = None
```

After `self._init_head_weights()`, add:

```python
if self.use_heatmap_init:
    self.heatmap_proj = nn.Linear(hidden_dim, 22)
    nn.init.zeros_(self.heatmap_proj.weight)
    nn.init.zeros_(self.heatmap_proj.bias)
    # heatmap_learnable_temp=False for this design; no per-joint temperature parameter
```

### 2d. Modify `forward()`

After `spatial = spatial + pos_enc`, before decoder, insert:

```python
# ── Heatmap-guided query warm-start ──────────────────────────────────────────
if self.use_heatmap_init:
    heatmap_logits = self.heatmap_proj(spatial)  # (B, H'W', 22)
    self._heatmap_logits = heatmap_logits

    # Temperature is scalar float (heatmap_learnable_temp=False)
    attn_weights = F.softmax(
        heatmap_logits.transpose(1, 2) / self.heatmap_temperature, dim=-1
    )  # (B, 22, H'W')

    pooled_features = torch.bmm(attn_weights, spatial)  # (B, 22, hidden_dim)

    pad = torch.zeros(B, self.num_joints - 22, self.hidden_dim,
                      device=spatial.device, dtype=spatial.dtype)
    delta = torch.cat([pooled_features, pad], dim=1)  # (B, 70, hidden_dim)
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1) + delta
else:
    self._heatmap_logits = None
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
```

Remove the original standalone query broadcast line.

### 2e. Modify `loss()`

After `losses['loss/uv/train'] = ...` and before `with torch.no_grad():`, insert:

```python
# ── Heatmap routing loss (KL divergence, Gaussian target) ────────────────────
if self.use_heatmap_init and self._heatmap_logits is not None:
    heatmap_loss = 0.0
    B_loss = len(batch_data_samples)
    for i in range(B_loss):
        ds = batch_data_samples[i]
        K_np = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h = int(img_shape[0])
        crop_w = int(img_shape[1])

        # GT absolute joints for body joints 0-21
        gt_pelvis_3d = _recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K_np, crop_h, crop_w)  # (1, 3)
        gt_abs_joints = gt_joints[i, :22] + gt_pelvis_3d  # (22, 3)

        # Project to feature grid coordinates
        grid_coords = _project_joints_to_grid_coords(
            gt_abs_joints, K_np, crop_h, crop_w,
            self.feat_h, self.feat_w)  # (22, 2)

        # Build soft Gaussian target: (22, H'W'), normalised to sum=1
        gt_hm = _build_gaussian_heatmap_target(
            grid_coords, self.feat_h, self.feat_w, self.heatmap_sigma)

        # KL divergence: KL(gt_hm || softmax(logits))
        # = -sum(gt_hm * log_softmax(logits)) + constant
        # We minimise the cross-entropy term: -sum(gt_hm * log_softmax(logits))
        logits_i = self._heatmap_logits[i].T  # (22, H'W')
        log_probs = F.log_softmax(logits_i, dim=-1)  # (22, H'W')
        # Sum over spatial dimension, mean over joints
        heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum(dim=-1).mean()

    losses['loss/heatmap/train'] = (
        self.heatmap_loss_weight * heatmap_loss / B_loss)
    self._heatmap_logits = None
```

**Key difference from design001**: the loss uses KL divergence (cross-entropy with soft Gaussian target) instead of hard one-hot cross-entropy. The per-sample loss is `-(gt_hm * log_probs).sum(dim=-1).mean()`: sum over spatial tokens, mean over 22 joints. Then divide by batch size.

---

## 3. `config.py` Changes

In the `head=dict(...)` block, add after `loss_weight_uv=1.0`:

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

**Complete head dict:**
```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    use_heatmap_init=True,
    heatmap_loss_weight=0.2,
    heatmap_target='gaussian',
    heatmap_sigma=2.0,
    heatmap_temperature=1.0,
    heatmap_learnable_temp=False,
    feat_h=40,
    feat_w=24,
),
```

---

## Invariants the Builder Must Preserve

1. **`persistent_workers=False`** — unchanged.
2. **Body-only joint loss** (indices 0–21) — unchanged.
3. **`_heatmap_logits` cleared after `loss()`** and in the else branch.
4. **`predict()` does not read `_heatmap_logits`**.
5. **Zero-init on `heatmap_proj`** — same safe warm-start as design001.
6. **`indexing='ij'` in `torch.meshgrid`** inside `_build_gaussian_heatmap_target` — required for correct row-major (H outer, W inner) spatial ordering matching the flatten/transpose in `forward()`.
7. **Flat index computation** in `_build_gaussian_heatmap_target`: the grid is built as `(feat_h*feat_w, 2)` with H-major ordering, matching `feat.flatten(2).transpose(1,2)`. The Gaussian is computed over all H'W' cells simultaneously.
8. **KL loss formulation**: use `F.log_softmax` + cross-entropy with soft target (`-(gt_hm * log_probs).sum(dim=-1).mean()`). Do not use `F.kl_div` directly (it requires log-space input and has a different reduction convention that requires extra care).
9. **No Python import statements in `config.py`**.
10. **`heatmap_sigma=2.0`** is a float literal in config; the `_build_gaussian_heatmap_target` function receives it as `sigma` float.

---

## Difference from Design 001

| Aspect | Design 001 | Design 002 |
|---|---|---|
| Heatmap target type | Hard one-hot | Soft Gaussian (σ=2 grid cells) |
| Loss function | Cross-entropy | KL divergence (soft CE) |
| Loss weight (λ) | 0.1 | 0.2 |
| `heatmap_target` config | `'onehot'` | `'gaussian'` |
| `heatmap_sigma` config | N/A (unused) | `2.0` |
| Helper function needed | None beyond projection | `_build_gaussian_heatmap_target` |

---

## Expected Behaviour

- **At training start**: identical to design001 — zero-init logits → uniform attention → global average pool warm-start.
- **Gaussian target**: for each joint, the GT heatmap assigns highest probability to the nearest grid cell and smoothly decays with σ=2 grid cells (~32px at input scale). Joints near grid-cell boundaries receive non-zero gradient signal at both cells, avoiding the hard discontinuity of one-hot targets.
- **Loss magnitude**: `-sum(gt_hm * log_probs)` starts near entropy of uniform distribution over 960 tokens (~6.87 nats). Decreases as heatmap sharpens. With Gaussian target the minimum is the entropy of the Gaussian (~lower bound), not 0.
- **Higher weight λ=0.2** vs design001 (0.1) reflects that Gaussian loss has smaller magnitude (soft targets reduce max CE value) and the smoother gradient justifies a stronger supervision signal.

---

## Expected Metrics (Stage-1, Epoch 20)

- `composite_val < 340` (expected to outperform design001)
- `mpjpe_body_val < 188mm`
- `mpjpe_rel_val < 410mm`
