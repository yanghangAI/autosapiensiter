# Design 001 — Auxiliary L1 Depth Reconstruction, λ=0.1 (diagnostic)

**Design Description:** Add a zero-init `Linear(hidden_dim=256, 1)` auxiliary head on the projected spatial tokens that regresses the bilinearly-downsampled input depth map at the 40x24 feature grid, supervised by `SmoothL1(beta=0.1)` with a valid-foreground mask (0.1 m < d < 30 m) and total weight λ=0.1.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pelvis_utils.py` — add `downsample_depth_map` helper.
2. `pose3d_transformer_head.py` — add a global module forward pre-hook that captures depth from 4-channel inputs; add `aux_depth_head`; compute aux-depth loss in `loss()`.
3. `config.py` — add aux-depth kwargs to the `head=dict(...)` block.

No other files are modified. All invariant files (`bedlam2_transforms.py`, `bedlam2_dataset.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `train.py`, `tools/train.py`, `infra/*`) are untouched.

---

## Algorithm

At every training step:

1. The RGBD input tensor `inputs` of shape `(B, 4, H=640, W=384)` is passed by the estimator to the backbone. A **module-global `forward_pre_hook`** registered once at import time of `pose3d_transformer_head.py` intercepts calls whose first positional argument is a 4-D float tensor with exactly 4 channels; it writes that tensor (detached) to a module-level slot keyed by the current default device so that the head can retrieve the depth channel downstream.
2. The head's `forward()` runs unchanged and additionally emits a per-cell metric-depth prediction: `aux_depth_pred = self.aux_depth_head(spatial).squeeze(-1).view(B, feat_h, feat_w)`.
3. In `loss()`:
   - Retrieve the captured RGBD input from the module-level slot. Extract the depth channel `inputs[:, 3:4]` (shape `(B, 1, H, W)`).
   - Undo the data-pipeline normalisation: multiply by `aux_depth_denorm_scale=20.0` (the `_DEPTH_MAX_METERS` constant used in `PackBedlamInputs`). This yields depth in metres in `[0, 20]`.
   - Bilinearly downsample to `(B, feat_h=40, feat_w=24)` via the helper.
   - Build a valid mask: `(depth_gt > 0.1) & (depth_gt < 30.0)`.
   - `recon_loss = F.smooth_l1_loss(aux_depth_pred[valid], depth_gt[valid], beta=0.1)`.
   - Guard against empty-mask batches: if `valid.sum() == 0`, use `recon_loss = aux_depth_pred.sum() * 0.0` (preserves gradient graph with zero value).
   - `losses['loss/aux_depth/train'] = aux_depth_loss_weight * recon_loss` with `aux_depth_loss_weight=0.1`.

The main joint/depth/uv losses are unchanged. The body-only restriction on joint loss (indices 0–21) is preserved. The `predict()` path is unchanged (aux head output is unused at inference; no side-channel reads are required).

### Zero-init property

`aux_depth_head.weight` and `aux_depth_head.bias` are zero-initialised so that at step 0 `aux_depth_pred = 0`. The gradient of the aux loss w.r.t. the main pathway flows only through `spatial` (shared tokens) via the Linear's weight gradient; the main losses are numerically unchanged at step 0.

---

## 1. `pelvis_utils.py` Changes

Append the following function at the end of the file, after `compute_mpjpe_abs`:

```python
import torch.nn.functional as F  # add this import near top if not present


def downsample_depth_map(
    depth_map: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """Bilinearly downsample a (B, 1, H, W) depth map to (B, feat_h, feat_w).

    Args:
        depth_map: (B, 1, H, W) float tensor.
        feat_h: Target height (default 40 for 640 / stride 16).
        feat_w: Target width (default 24 for 384 / stride 16).

    Returns:
        (B, feat_h, feat_w) float tensor.
    """
    return F.interpolate(
        depth_map, size=(feat_h, feat_w),
        mode='bilinear', align_corners=False,
    ).squeeze(1)
```

Imports: `torch` is already imported at the top of `pelvis_utils.py`. Add `import torch.nn.functional as F` at the top if not already present (baseline does not import it).

---

## 2. `pose3d_transformer_head.py` Changes

### 2a. New module-level depth capture hook (added at the top of the file)

Immediately after the existing imports and **before** any class definitions, add the following module-level block:

```python
import torch.nn.functional as F  # add to imports if not already present

# ── Depth-channel capture via a global module forward pre-hook ───────────────
# The head's loss() needs access to the raw RGBD input tensor (specifically its
# depth channel). The estimator calls backbone.forward(inputs) before head.loss,
# but does not forward `inputs` to the head. To avoid modifying the invariant
# estimator / preprocessor / transforms, we register a *global* module forward
# pre-hook. It fires on every nn.Module forward and captures any first-arg
# tensor that is 4-D with exactly 4 channels — i.e. the RGBD batch entering
# the backbone. The captured tensor is stored in a module-level dict keyed by
# the tensor's device. The head reads it back inside `loss()`.
#
# Safety / scope:
#   - The hook fires on every module forward. The filter (4-D tensor, C==4) is
#     strict enough to only match the RGBD input. Even if it were to match
#     spuriously, the stored tensor is merely overwritten; nothing downstream
#     depends on staleness because `loss()` always reads the *latest* value.
#   - `.detach()` prevents the aux-depth supervision target from creating a
#     graph edge back through the input tensor.
#   - Multi-GPU: each device gets its own entry. On single-GPU (2080 Ti per
#     the runtime constraints) there is exactly one entry.
#   - `inputs` is normalised in [0, 1] by `PackBedlamInputs`
#     (depth / _DEPTH_MAX_METERS with _DEPTH_MAX_METERS = 20.0). We multiply
#     by 20.0 inside `loss()` to recover metric depth.

_LAST_RGBD_INPUT: Dict[torch.device, torch.Tensor] = {}


def _rgbd_capture_pre_hook(module, args):
    if not args:
        return None
    x = args[0]
    if not isinstance(x, torch.Tensor):
        return None
    if x.dim() == 4 and x.shape[1] == 4 and x.is_floating_point():
        _LAST_RGBD_INPUT[x.device] = x.detach()
    return None


# Register once per import. MMEngine re-imports the head module when building
# the runner; we guard to avoid duplicate hooks.
if not globals().get('_RGBD_CAPTURE_HOOK_REGISTERED', False):
    import torch.nn.modules.module as _tmm
    _tmm.register_module_forward_pre_hook(_rgbd_capture_pre_hook)
    _RGBD_CAPTURE_HOOK_REGISTERED = True
```

Also update the `pelvis_utils` import to include the new helper:

```python
from pelvis_utils import (compute_mpjpe_abs as _compute_mpjpe_abs,
                          downsample_depth_map as _downsample_depth_map)
```

### 2b. Add `__init__` keyword arguments

Add the following keyword arguments to `Pose3dTransformerHead.__init__` after `loss_weight_uv` and before `init_cfg`:

```python
use_aux_depth: bool = False,
aux_depth_loss_weight: float = 0.1,
aux_depth_log_space: bool = False,
aux_depth_grad_weight: float = 0.0,
aux_depth_valid_min: float = 0.1,
aux_depth_valid_max: float = 30.0,
aux_depth_denorm_scale: float = 20.0,
feat_h: int = 40,
feat_w: int = 24,
```

Inside `__init__`, immediately after the existing `self.loss_weight_uv = loss_weight_uv`, add:

```python
self.use_aux_depth = use_aux_depth
self.aux_depth_loss_weight = aux_depth_loss_weight
self.aux_depth_log_space = aux_depth_log_space
self.aux_depth_grad_weight = aux_depth_grad_weight
self.aux_depth_valid_min = aux_depth_valid_min
self.aux_depth_valid_max = aux_depth_valid_max
self.aux_depth_denorm_scale = aux_depth_denorm_scale
self.feat_h = feat_h
self.feat_w = feat_w
```

Immediately after `self._init_head_weights()` add:

```python
if self.use_aux_depth:
    self.aux_depth_head = nn.Linear(hidden_dim, 1)
    nn.init.zeros_(self.aux_depth_head.weight)
    nn.init.zeros_(self.aux_depth_head.bias)
```

### 2c. Modify `forward()`

Inside `forward()`, **after** the line `spatial = spatial + pos_enc` and **before** the `queries = ...` broadcast line, insert:

```python
# ── Auxiliary depth reconstruction head ──────────────────────────────────────
if self.use_aux_depth:
    aux_depth_pred = self.aux_depth_head(spatial).squeeze(-1)    # (B, H*W)
    self._aux_depth_pred = aux_depth_pred.view(B, self.feat_h, self.feat_w)
else:
    self._aux_depth_pred = None
```

`B`, `H`, `W` are already defined earlier in `forward()`. The existing decoder + output-head pipeline is unchanged; the aux prediction is stored as a side-channel on `self._aux_depth_pred` for `loss()` to read.

Also initialise `self._aux_depth_pred = None` once at the end of `__init__` so `predict()` never observes a stale attribute.

### 2d. Modify `loss()`

After `losses['loss/uv/train'] = ...` and **before** the `with torch.no_grad():` block that computes `_train_mpjpe`, insert:

```python
# ── Auxiliary depth reconstruction loss ─────────────────────────────────────
if self.use_aux_depth and self._aux_depth_pred is not None:
    device = self._aux_depth_pred.device
    rgbd = _LAST_RGBD_INPUT.get(device, None)
    if rgbd is None:
        # Defensive fallback: no RGBD capture — emit a zero-graph loss so
        # the key still appears in logs and gradient flow is safe.
        losses['loss/aux_depth/train'] = self._aux_depth_pred.sum() * 0.0
    else:
        depth_norm = rgbd[:, 3:4]                              # (B, 1, H, W)
        depth_m = depth_norm * self.aux_depth_denorm_scale     # metres
        depth_gt = _downsample_depth_map(
            depth_m, self.feat_h, self.feat_w)                 # (B, fh, fw)

        if self.aux_depth_log_space:
            target = torch.log1p(depth_gt)
        else:
            target = depth_gt
        pred = self._aux_depth_pred

        valid = (depth_gt > self.aux_depth_valid_min) & (
            depth_gt < self.aux_depth_valid_max)
        if valid.any():
            recon_loss = F.smooth_l1_loss(
                pred[valid], target[valid], beta=0.1)
        else:
            recon_loss = pred.sum() * 0.0

        if self.aux_depth_grad_weight > 0:
            dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
            dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
            dx_tgt = target[:, :, 1:] - target[:, :, :-1]
            dy_tgt = target[:, 1:, :] - target[:, :-1, :]
            grad_loss = (dx_pred - dx_tgt).abs().mean() + (
                dy_pred - dy_tgt).abs().mean()
            recon_loss = recon_loss + self.aux_depth_grad_weight * grad_loss

        losses['loss/aux_depth/train'] = (
            self.aux_depth_loss_weight * recon_loss)
    # Clear side-channel after consumption.
    self._aux_depth_pred = None
```

For Design 001 the ctrl flags above reduce to: `aux_depth_log_space=False`, `aux_depth_grad_weight=0.0`, `aux_depth_loss_weight=0.1`.

Finally, ensure `predict()` is unchanged. It calls `forward()` which sets `self._aux_depth_pred`, but `predict()` never reads it and the attribute is harmlessly retained until the next `loss()` call (which does not occur during evaluation).

---

## 3. `config.py` Changes

In the `head=dict(...)` block inside `model=dict(...)`, append the following kwargs after `loss_weight_uv=1.0,`:

```python
use_aux_depth=True,
aux_depth_loss_weight=0.1,
aux_depth_log_space=False,
aux_depth_grad_weight=0.0,
aux_depth_valid_min=0.1,
aux_depth_valid_max=30.0,
aux_depth_denorm_scale=20.0,
feat_h=40,
feat_w=24,
```

All values are bool/int/float literals. No Python `import` statements are introduced. MMEngine config constraint is satisfied.

**Resulting head block (for reference):**

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
    use_aux_depth=True,
    aux_depth_loss_weight=0.1,
    aux_depth_log_space=False,
    aux_depth_grad_weight=0.0,
    aux_depth_valid_min=0.1,
    aux_depth_valid_max=30.0,
    aux_depth_denorm_scale=20.0,
    feat_h=40,
    feat_w=24,
),
```

All other config blocks (optimizer, LR schedule, dataloader, hooks, env) are **unchanged**.

---

## Invariants the Builder Must Preserve

1. **`persistent_workers=False`** in both dataloaders.
2. **Body-only joint loss (indices 0–21)** — unchanged.
3. **`predict()` path** must not depend on aux-depth; it must produce identical output shapes/keys as baseline.
4. **Zero-init** on `aux_depth_head` weight and bias.
5. **Absolute imports** in `pose3d_transformer_head.py` (file is outside `mmpose` package).
6. **No `import` statements in `config.py`** — only bool/int/float/str literals and the existing `__import__()` pattern used by the baseline (if any).
7. **Feature grid** `feat_h=40`, `feat_w=24` (from `img_h=640, img_w=384, stride=16`).
8. **Depth denormalisation scale `20.0`** must match `_DEPTH_MAX_METERS` in `bedlam2_transforms.py`. If that constant changes, the Builder must update `aux_depth_denorm_scale` accordingly; as of the current invariant version, it is `20.0`.
9. **Hook registration guard** (`_RGBD_CAPTURE_HOOK_REGISTERED`) must be in place to prevent duplicate hook registration when the module is re-imported.
10. **Mask lower bound `0.1 m`** filters zero-fill / missing-depth pixels (BEDLAM2 fills missing depth as 0, which after division by 20 becomes 0, which stays 0 here). **Mask upper bound `30 m`** is above `_DEPTH_MAX_METERS=20`; after denorm the range is `[0, 20]`, so the upper bound is effectively trivial but kept for parity with the idea spec.
11. **Empty-mask safety**: if no pixel is valid in a batch, the loss falls back to `pred.sum() * 0.0` to keep a live gradient graph.
12. **AMP safety**: `F.interpolate(bilinear)` + `F.smooth_l1_loss` are FP16-safe under `FixedAmpOptimWrapper`.

---

## Edge Cases and Risks

- **Depth-channel clipping at 20 m**: `PackBedlamInputs` clips raw depth to `[0, 20]` before normalising. Thus any true depth above 20 m is capped. The valid range `[0.1, 30]` (post-denorm) therefore includes all *valid foreground* pixels up to the cap and excludes the zero-fill floor. This is the intended regime for human-subject foreground depth.
- **Depth zero-fill pixels**: `PackBedlamInputs` substitutes `torch.zeros(1, H, W)` when depth is missing. These pixels are masked out by `depth_gt > 0.1`.
- **Preemption / resume**: the hook is registered at module import time and survives checkpoint reload (it's attached to the global module-class registry, not saved state). `CheckpointHook` does not persist nor restore global hooks; they are re-registered on import.
- **Validation-time forward**: the global hook will also fire during validation, re-capturing a val RGBD tensor. Because `predict()` never reads `_aux_depth_pred`, this is harmless.

---

## Expected Behaviour

- **Step 0**: `aux_depth_pred = 0` → `recon_loss = SmoothL1(0, depth_gt)` on valid pixels; gradient on `aux_depth_head.weight` only (zero gradient into `spatial` at step 0 because `aux_depth_head.weight=0` makes d loss / d spatial = 0). Main losses identical to baseline at step 0.
- **Steady state**: aux-depth loss decreases as spatial tokens learn to carry dense metric-depth. Magnitude typically O(0.5–5) in metres before λ; with λ=0.1 the aux term contributes O(0.05–0.5) to total loss, which is small compared to the main losses (O(1–10)).
- **New CSV scalar key**: `loss/aux_depth/train`. The existing metric keys (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`) are unchanged.

---

## Expected Metrics (Stage-1, Epoch 20)

- `composite_val <= baseline` (auxiliary head with zero-init can only help or be neutral).
- Mild improvement expected on `mpjpe_pelvis_val` and `mpjpe_abs_val` from richer spatial feature depth grounding.
- Mild improvement or neutral on `mpjpe_body_val`.
