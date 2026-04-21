# Design 001 — Heatmap-Guided Query Init: Hard One-Hot Target (λ=0.1)

**Design Description:** Add a zero-init `Linear(256, 22)` heatmap projector that produces per-joint soft attention weights over the 40×24 spatial token grid; use these weights to pool a joint-specific feature vector added to body query embeddings (0–21) before the decoder; supervise with hard one-hot cross-entropy at weight 0.1.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pelvis_utils.py` — add `project_joints_to_grid_coords` helper
2. `pose3d_transformer_head.py` — add heatmap projection, pooling, and loss
3. `config.py` — add new head kwargs

---

## Algorithm

The heatmap-guided query warm-start algorithm proceeds in three steps for each forward pass:

1. **Heatmap prediction**: apply `heatmap_proj: Linear(hidden_dim=256, 22)` to each of the H'W'=960 spatial tokens → `heatmap_logits` of shape `(B, 960, 22)`.
2. **Soft attention pooling**: transpose to `(B, 22, 960)`, divide by temperature (scalar 1.0), apply softmax over the 960 spatial dimension → `attn_weights (B, 22, 960)`. Matrix-multiply by spatial tokens `(B, 960, 256)` → `pooled_features (B, 22, 256)`: a per-joint content-aware feature vector.
3. **Query warm-start**: zero-pad `pooled_features` to `(B, 70, 256)` (zeros for hand joints 22–69), add to static joint query embeddings before the decoder. Body queries (0–21) get image-specific spatial routing; hand queries (22–69) are unchanged.

The heatmap loss (supervision step): for each sample in the batch, project GT absolute joints through K to get 2D pixel positions, convert to feature grid coordinates, snap to nearest grid cell (one-hot), compute cross-entropy between predicted logits `(22, 960)` and the one-hot target `(22,)` index. Scale by `λ=0.1` and average over batch.

The algorithmic invariant: `heatmap_proj` is initialized to all-zeros, so at training start `pooled_features = mean(spatial)` for all joints — equivalent to global average pool (same as idea003/design001). No instability at initialization.

---

## 1. `pelvis_utils.py` Changes

Add the following function **after** `compute_mpjpe_abs` (append to end of file):

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
        feat_h: Feature grid height (default 40, from img_h=640 / stride=16).
        feat_w: Feature grid width (default 24, from img_w=384 / stride=16).

    Returns:
        (J, 2) float tensor: (h_frac, w_frac) in feature grid units.
        Values may be outside [0, feat_h-1] x [0, feat_w-1] for out-of-frame joints.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    X = joints_abs[:, 0].clamp(min=0.01)  # forward depth; clamp to avoid div-by-zero
    Y = joints_abs[:, 1]
    Z = joints_abs[:, 2]

    # Project to pixel coordinates (BEDLAM2 convention)
    u_px = -Y / X * fx + cx  # (J,)
    v_px = -Z / X * fy + cy  # (J,)

    # Convert pixel coordinates to feature grid coordinates
    h_frac = v_px / crop_h * feat_h  # (J,)
    w_frac = u_px / crop_w * feat_w  # (J,)

    return torch.stack([h_frac, w_frac], dim=-1)  # (J, 2)
```

**Import note:** `project_joints_to_grid_coords` uses `torch` and `np` which are already imported at the top of `pelvis_utils.py`. No new imports needed.

---

## 2. `pose3d_transformer_head.py` Changes

### 2a. Add import for `torch.nn.functional` and `project_joints_to_grid_coords`

At the top of the file, after the existing imports, add:

```python
import numpy as np
import torch.nn.functional as F
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          project_joints_to_grid_coords as _project_joints_to_grid_coords)
```

**Note:** The existing file imports `compute_mpjpe_abs` only; replace that import line with the combined import above. Also add `import numpy as np` if not already present. Check the baseline: the baseline imports `numpy` indirectly via `pelvis_utils`; add `import numpy as np` explicitly to this file since `loss()` uses `np.asarray`.

**Exact change to the existing import line** (line 36 in baseline):
```python
# Replace:
from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs

# With:
import numpy as np
import torch.nn.functional as F
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          project_joints_to_grid_coords as _project_joints_to_grid_coords)
```

### 2b. Add `__init__` parameters

Add the following keyword arguments to `Pose3dTransformerHead.__init__` after `loss_weight_uv`:

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

Inside `__init__`, after `self.loss_weight_uv = loss_weight_uv`, store the new parameters:

```python
self.use_heatmap_init = use_heatmap_init
self.heatmap_loss_weight = heatmap_loss_weight
self.heatmap_target = heatmap_target
self.heatmap_sigma = heatmap_sigma
self.heatmap_temperature = heatmap_temperature
self.heatmap_learnable_temp = heatmap_learnable_temp
self.feat_h = feat_h
self.feat_w = feat_w
self._heatmap_logits = None  # side-channel for loss(); cleared after use
```

After `self._init_head_weights()`, add the conditional heatmap projection:

```python
if self.use_heatmap_init:
    self.heatmap_proj = nn.Linear(hidden_dim, 22)
    nn.init.zeros_(self.heatmap_proj.weight)
    nn.init.zeros_(self.heatmap_proj.bias)
    # No learnable temperature for Design 001 (heatmap_learnable_temp=False)
```

### 2c. Modify `forward()`

After the line `spatial = spatial + pos_enc` and before the query broadcast and decoder call, insert:

```python
# ── Heatmap-guided query warm-start ──────────────────────────────────────────
if self.use_heatmap_init:
    # heatmap_logits: (B, H'W', 22) — one logit per spatial token per body joint
    heatmap_logits = self.heatmap_proj(spatial)  # (B, H'W', 22)
    self._heatmap_logits = heatmap_logits         # stored for loss()

    # Soft attention over spatial dimension: (B, 22, H'W')
    # Temperature is scalar float (heatmap_learnable_temp=False for this design)
    attn_weights = F.softmax(
        heatmap_logits.transpose(1, 2) / self.heatmap_temperature, dim=-1
    )  # (B, 22, H'W')

    # Soft pooling: weighted sum of spatial tokens per joint → (B, 22, hidden_dim)
    pooled_features = torch.bmm(attn_weights, spatial)  # (B, 22, hidden_dim)

    # Build delta: pad pooled_features with zeros for hand joints (22-69)
    # to get full (B, 70, hidden_dim) delta tensor
    pad = torch.zeros(B, self.num_joints - 22, self.hidden_dim,
                      device=spatial.device, dtype=spatial.dtype)
    delta = torch.cat([pooled_features, pad], dim=1)  # (B, 70, hidden_dim)

    # Add delta to static queries (expand is read-only; + creates new tensor)
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1) + delta
else:
    self._heatmap_logits = None
    queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
```

**Remove the original query broadcast line** that follows (since it is now inside the else branch):
```python
# Delete this line (it already existed in baseline):
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
```

The decoder call remains unchanged:
```python
decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)
```

### 2d. Modify `loss()`

After `losses['loss/uv/train'] = ...` and before the `with torch.no_grad():` block, insert the heatmap loss computation:

```python
# ── Heatmap routing loss ──────────────────────────────────────────────────────
if self.use_heatmap_init and self._heatmap_logits is not None:
    heatmap_loss = 0.0
    B_loss = len(batch_data_samples)
    for i in range(B_loss):
        ds = batch_data_samples[i]
        K_np = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h = int(img_shape[0])
        crop_w = int(img_shape[1])

        # Reconstruct GT absolute joints for body joints 0-21
        gt_pelvis_3d = recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K_np, crop_h, crop_w)  # (1, 3)
        gt_abs_joints = gt_joints[i, :22] + gt_pelvis_3d  # (22, 3)

        # Project to feature grid coordinates
        grid_coords = _project_joints_to_grid_coords(
            gt_abs_joints, K_np, crop_h, crop_w,
            self.feat_h, self.feat_w)  # (22, 2)

        # Hard one-hot target: nearest grid cell, clamped to valid range
        h_idx = grid_coords[:, 0].long().clamp(0, self.feat_h - 1)  # (22,)
        w_idx = grid_coords[:, 1].long().clamp(0, self.feat_w - 1)  # (22,)
        target_idx = h_idx * self.feat_w + w_idx  # (22,) flat spatial index

        # Cross-entropy over spatial tokens: logits (22, H'W'), target (22,)
        logits_i = self._heatmap_logits[i].T  # (22, H'W')
        heatmap_loss = heatmap_loss + F.cross_entropy(logits_i, target_idx)

    losses['loss/heatmap/train'] = (
        self.heatmap_loss_weight * heatmap_loss / B_loss)
    self._heatmap_logits = None  # clear side-channel
```

**Note:** `recover_pelvis_3d` is already used in `_compute_mpjpe_abs` inside `pelvis_utils.py`. In the head file, call it directly via `from pelvis_utils import recover_pelvis_3d`. Add this to the import line in step 2a:

```python
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          project_joints_to_grid_coords as _project_joints_to_grid_coords,
                          recover_pelvis_3d as _recover_pelvis_3d)
```

Then use `_recover_pelvis_3d(...)` in the loss function above.

---

## 3. `config.py` Changes

In the `head=dict(...)` block inside `model=dict(...)`, add the following kwargs after `loss_weight_uv=1.0`:

```python
use_heatmap_init=True,
heatmap_loss_weight=0.1,
heatmap_target='onehot',
heatmap_temperature=1.0,
heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

All values are bool/int/float/str literals. No Python import statements. MMEngine config constraint fully satisfied.

**Complete head dict after change:**
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
    heatmap_loss_weight=0.1,
    heatmap_target='onehot',
    heatmap_temperature=1.0,
    heatmap_learnable_temp=False,
    feat_h=40,
    feat_w=24,
),
```

---

## Invariants the Builder Must Preserve

1. **`persistent_workers=False`** in both dataloaders — do not change.
2. **Loss is restricted to body joints (indices 0–21)** — `loss_joints_module` call uses `pred['joints'][:, _BODY]` and `gt_joints[:, _BODY]`. The heatmap loss is *separate* from this and only applies to body joints via the 22-column heatmap projection.
3. **`_heatmap_logits` side-channel** must be set to `None` after `loss()` reads it, and also initialized to `None` when `use_heatmap_init=False` (or in the else branch).
4. **`predict()` must not access `_heatmap_logits`** — `predict()` calls `forward()` (which sets `self._heatmap_logits`) then never calls `loss()`. The side-channel is harmless but must not interfere with prediction output.
5. **Zero-init on `heatmap_proj`** — at training start, logits are all zero → uniform softmax → `pooled_features = mean(spatial)` = global average pool. This is a safe, stable start identical to idea003/design001's approach.
6. **Feature grid dimensions**: img_h=640, img_w=384, backbone stride=16 → feat_h=40, feat_w=24. These are hardcoded in config.py and confirmed correct.
7. **Spatial token flattening order**: `feat.flatten(2).transpose(1, 2)` where feat is `(B, C, H=40, W=24)` → row-major (H outer, W inner). `project_joints_to_grid_coords` returns `(h_frac, w_frac)` matching this ordering. Flat index = `h_idx * feat_w + w_idx`.
8. **No Python import statements in `config.py`** — all new kwargs are literals only.
9. **Absolute imports in `pose3d_transformer_head.py`** — the file lives outside the mmpose package; all imports use full paths (e.g., `from mmpose.models.heads.base_head import BaseHead`).
10. **`num_joints - 22 = 48`** padding slots for hand queries — Builder should use `self.num_joints - 22` (not hardcode 48) to remain correct if `num_joints` ever changes.

---

## Expected Behaviour

- **At training start**: `heatmap_proj` weights are zero → `heatmap_logits = 0` → uniform attention → `pooled_features = mean(spatial tokens)` per joint → queries start as `static_embed + global_avg_pool_feature`. Identical to idea003/design001 warm-start. Training is stable.
- **After convergence**: `heatmap_proj` learns to route each of the 22 body joint queries to their expected image region. Cross-attention in the decoder starts with spatially pre-routed queries, reducing the routing burden on the first decoder layer.
- **Heatmap loss** (cross-entropy, weight 0.1): penalizes the heatmap for assigning low probability to the grid cell containing the GT 2D joint projection. The small weight (0.1) keeps the primary 3D regression loss dominant.
- **New loss key**: `loss/heatmap/train` appears in training logs. Magnitude should decrease from ~log(960) ≈ 6.87 nats initially toward ~0 as the heatmap sharpens.
- **No change to validation/test**: the heatmap module is active during `forward()` but `_heatmap_logits` is only consumed in `loss()`. `predict()` calls `forward()` but ignores the side-channel attribute.

---

## Expected Metrics (Stage-1, Epoch 20)

- `composite_val < 345` (baseline ~360)
- `mpjpe_body_val < 190mm`
- `mpjpe_rel_val < 420mm` (baseline 438mm)
