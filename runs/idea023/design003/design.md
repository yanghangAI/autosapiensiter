# Design 003 — Heatmap-Guided Query Init: Gaussian Target + Learnable Per-Joint Temperature

**Design Description:** Same as design002 (Gaussian KL heatmap loss, λ=0.2), but add a learnable per-joint softmax temperature `nn.Parameter(torch.ones(22))` passed through `F.softplus`, allowing the model to learn sharp narrow heatmaps for easy-to-locate joints (pelvis, spine) and diffuse heatmaps for harder joints (wrists, ankles).

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pelvis_utils.py` — add `project_joints_to_grid_coords` helper (identical to design001/design002)
2. `pose3d_transformer_head.py` — all changes from design002, plus per-joint learnable temperature
3. `config.py` — add new head kwargs with `heatmap_learnable_temp=True`

---

## Algorithm

The algorithm is identical to design002 (Gaussian KL heatmap loss), with one additional mechanism: learnable per-joint softmax temperature.

1. **Heatmap prediction**: `heatmap_proj: Linear(256, 22)` → `heatmap_logits (B, 960, 22)`.
2. **Learnable temperature**: `self.heatmap_temp = nn.Parameter(torch.ones(22))`. At runtime, `temp = F.softplus(self.heatmap_temp).view(1, 22, 1)` (shape `(1, 22, 1)`). `F.softplus` ensures positivity.
3. **Soft attention pooling**: `F.softmax(heatmap_logits.transpose(1,2) / temp, dim=-1)` → `attn_weights (B, 22, 960)`. Each of the 22 joints divides its logits by its own learned temperature. BMM with spatial tokens → `pooled_features (B, 22, 256)`.
4. **Query warm-start**: identical to design001/002.
5. **Loss**: identical to design002 — KL divergence with Gaussian targets on raw logits (before temperature), weight λ=0.2.

The algorithmic intuition: easy-to-locate joints (pelvis, thorax) receive strong, consistent spatial signal and can afford a low temperature (sharp attention to a small region). Hard joints (wrists, ankles) have variable positions and benefit from a higher temperature (broader pooling to avoid missing the joint). The per-joint temperature allows the model to learn this joint-specific sharpness automatically from data. The loss operates on raw logits (temperature-free) so that the calibration target is independent of the attention sharpness.

---

## 1. `pelvis_utils.py` Changes

**Identical to design001/design002.** Add `project_joints_to_grid_coords` after `compute_mpjpe_abs`:

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

Same as design002:

```python
import numpy as np
import torch.nn.functional as F
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          project_joints_to_grid_coords as _project_joints_to_grid_coords,
                          recover_pelvis_3d as _recover_pelvis_3d)
```

### 2b. Module-level helper `_build_gaussian_heatmap_target`

**Identical to design002.** Insert before `_DecoderLayer` class:

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
    grid_h, grid_w = torch.meshgrid(gh, gw, indexing='ij')
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)  # (feat_h*feat_w, 2)

    mu = joint_grid_coords.unsqueeze(1)  # (J, 1, 2)
    g = grid.unsqueeze(0)                # (1, feat_h*feat_w, 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)  # (J, feat_h*feat_w)

    heatmap = torch.exp(-dist2 / (2.0 * sigma ** 2))
    heatmap = heatmap / heatmap.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return heatmap  # (J, feat_h*feat_w)
```

### 2c. `__init__` parameters

Add to `Pose3dTransformerHead.__init__` after `loss_weight_uv` (same set as design001/002):

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
    if self.heatmap_learnable_temp:
        # Per-joint temperature parameter; init to 1.0 (identical to scalar temp=1.0 start)
        # F.softplus(1.0) ≈ 1.313, so we initialise the raw parameter to
        # softplus_inv(1.0) = log(exp(1.0) - 1) ≈ 0.541 to get effective temp ≈ 1.0
        # Simpler: just use torch.ones(22) as raw param; F.softplus(1.0) ≈ 1.31
        # means initial effective temperature ≈ 1.31 (slightly above 1.0, acceptable)
        self.heatmap_temp = nn.Parameter(torch.ones(22))
```

**Temperature initialization detail**: `nn.Parameter(torch.ones(22))` with `F.softplus` gives initial effective temperature ≈ `softplus(1.0) = log(1 + e) ≈ 1.3133`. This is slightly above 1.0 but produces marginally softer attention than design002 at initialization — acceptable and safe. The model will learn to adjust per-joint temperatures from this starting point. If exact temperature=1.0 at init is required, use raw parameter value ≈ 0.5413 (`log(exp(1.0) - 1)`). **The Builder should use `torch.ones(22)` for simplicity** (initial effective temp ≈ 1.31 is safe).

### 2d. Modify `forward()`

After `spatial = spatial + pos_enc`, before decoder, insert:

```python
# ── Heatmap-guided query warm-start ──────────────────────────────────────────
if self.use_heatmap_init:
    heatmap_logits = self.heatmap_proj(spatial)  # (B, H'W', 22)
    self._heatmap_logits = heatmap_logits

    if self.heatmap_learnable_temp:
        # Per-joint learnable temperature: (22,) → (1, 22, 1) for broadcasting
        # F.softplus ensures positivity; raw param init ~1.0 → effective temp ~1.31
        temp = F.softplus(self.heatmap_temp).view(1, 22, 1)  # (1, 22, 1)
    else:
        temp = self.heatmap_temperature  # scalar float

    # heatmap_logits.transpose(1, 2) → (B, 22, H'W')
    # Divide by temperature (broadcasts correctly for both scalar and (1,22,1))
    attn_weights = F.softmax(
        heatmap_logits.transpose(1, 2) / temp, dim=-1
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

**Broadcasting note for learnable temperature**: `heatmap_logits.transpose(1, 2)` is `(B, 22, H'W')`. Dividing by `temp` of shape `(1, 22, 1)` broadcasts correctly — each of the 22 joints gets its own temperature scalar applied across all H'W' spatial tokens and all B batch elements.

Remove the original standalone query broadcast line.

### 2e. Modify `loss()`

**Identical to design002.** After `losses['loss/uv/train'] = ...`, before `with torch.no_grad():`:

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

        gt_pelvis_3d = _recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K_np, crop_h, crop_w)  # (1, 3)
        gt_abs_joints = gt_joints[i, :22] + gt_pelvis_3d  # (22, 3)

        grid_coords = _project_joints_to_grid_coords(
            gt_abs_joints, K_np, crop_h, crop_w,
            self.feat_h, self.feat_w)  # (22, 2)

        gt_hm = _build_gaussian_heatmap_target(
            grid_coords, self.feat_h, self.feat_w, self.heatmap_sigma)

        logits_i = self._heatmap_logits[i].T  # (22, H'W')
        log_probs = F.log_softmax(logits_i, dim=-1)  # (22, H'W')
        heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum(dim=-1).mean()

    losses['loss/heatmap/train'] = (
        self.heatmap_loss_weight * heatmap_loss / B_loss)
    self._heatmap_logits = None
```

**Note on temperature and loss**: the learnable temperature is applied in `forward()` when computing `attn_weights` for the query warm-start (pooling attention). The heatmap `_heatmap_logits` stored for the loss are the **raw logits before temperature** (output of `heatmap_proj`). The loss is computed on these raw logits via `F.log_softmax(logits_i, dim=-1)` — this is correct: the loss supervises the raw logit space so that the distribution before temperature scaling is well-calibrated. The temperature affects how sharply the pooling attention acts, not the loss target.

---

## 3. `config.py` Changes

In the `head=dict(...)` block, add after `loss_weight_uv=1.0`:

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
    heatmap_learnable_temp=True,
    feat_h=40,
    feat_w=24,
),
```

`heatmap_temperature=1.0` is included but unused at runtime when `heatmap_learnable_temp=True` (the learnable `self.heatmap_temp` parameter takes precedence in `forward()`). It is kept in config for consistency with the `__init__` signature. The Builder must ensure the `forward()` code checks `self.heatmap_learnable_temp` first.

---

## Invariants the Builder Must Preserve

1. **`persistent_workers=False`** — unchanged.
2. **Body-only joint loss** (indices 0–21) — unchanged.
3. **`_heatmap_logits` cleared after `loss()`** and in the else branch.
4. **`predict()` does not read `_heatmap_logits`**.
5. **Zero-init on `heatmap_proj`** — same safe warm-start as design001/002.
6. **`indexing='ij'` in `torch.meshgrid`** — required for correct H-major spatial ordering.
7. **`F.softplus` for temperature positivity** — raw `heatmap_temp` parameter is unconstrained; `F.softplus` maps it to (0, +∞). Never apply temperature before zero-crossing.
8. **Broadcasting shape for learnable temp**: `(1, 22, 1)` divides into `(B, 22, H'W')` correctly. Use `.view(1, 22, 1)` not `.view(22, 1)` to ensure batch dimension broadcast.
9. **Heatmap loss operates on raw logits** (before temperature): `self._heatmap_logits[i]` is the output of `heatmap_proj` before any temperature division. The loss uses `F.log_softmax(logits_i, dim=-1)` with temperature=1. This is correct: the temperature is a rescaling of the same distribution, not a different distribution. Both designs train to the same target distribution; temperature controls sharpness of the query pooling, not the calibration target.
10. **No Python import statements in `config.py`**.

---

## Difference from Design 002

| Aspect | Design 002 | Design 003 |
|---|---|---|
| Temperature type | Scalar float `1.0` | `nn.Parameter(torch.ones(22))` via `F.softplus` |
| Temperature scope | Global (all joints) | Per-joint (22 separate scalars) |
| `heatmap_learnable_temp` | `False` | `True` |
| New parameters | None | 22 floats (negligible) |
| Loss function | KL / soft CE (identical) | KL / soft CE (identical) |
| Loss weight λ | 0.2 (identical) | 0.2 (identical) |

---

## Expected Behaviour

- **At training start**: similar to design002. Zero-init logits → uniform attention (initial temperature ≈ 1.31 via softplus gives only a mild softening vs. 1.0, negligible at uniform distribution). Safe warm-start.
- **After convergence**: easy joints (pelvis/spine — large, central, high-contrast in depth) should learn low temperature values → sharp, focused attention. Hard joints (wrists/ankles — small, variable, near boundaries) should retain higher temperatures → more diffuse pooling that averages over a neighbourhood. This joint-specific adaptation is the main added value over design002.
- **Temperature monitoring**: the 22 temperature values can be read as `F.softplus(model.head.heatmap_temp).detach().cpu().numpy()` at any checkpoint. Expected range: [0.3, 3.0] after convergence.
- **AMP safety**: `F.softplus` is safe under float16; the output is always positive and bounded. The temperature values after softplus are in a reasonable range for dividing logits.

---

## Expected Metrics (Stage-1, Epoch 20)

- `composite_val < 335` (expected highest potential of the three designs)
- `mpjpe_body_val < 185mm`
- `mpjpe_rel_val < 400mm`
