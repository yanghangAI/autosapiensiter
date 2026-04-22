**Design Description:** 2D spatial heatmap classification head for pelvis UV; soft Gaussian target (sigma=2 cells); KL heatmap loss weight 0.5; no learnable temperature.

**Starting Point:** `baseline/`

---

## Algorithm

Replace the baseline scalar `uv_out = Linear(hidden_dim, 2)` with a 2D spatial heatmap classification over the `H'xW' = 40x24` feature grid. Project spatial tokens (pre-decoder `spatial = feat.flatten(2).transpose(1, 2) + pos_enc`) to a per-cell logit via `uv_heatmap_proj = Linear(hidden_dim, 1)`, softmax over the 960 cells to produce an attention distribution, and recover continuous `(u_norm, v_norm) in [-1, 1]` via soft-argmax. The existing SmoothL1 UV loss (weight 1.0) remains active on this soft-argmax output; additionally, add a KL/cross-entropy heatmap loss (weight 0.5) against a Gaussian target heatmap (sigma=2 grid cells) centred on the GT pelvis grid location. `uv_heatmap_proj` is zero-initialized so the attention distribution is uniform at step 0 and the soft-argmax returns grid centre `(0, 0)`.

## Overview

Design A from idea031 — the minimal diagnostic variant. Converts the scalar UV regression into a spatial distribution over the 40x24 feature grid, supervised by both the continuous SmoothL1 loss (via differentiable soft-argmax) and a KL classification loss against a soft Gaussian heatmap. Output interface `pred['pelvis_uv']` is unchanged — still `(B, 2)` float in `[-1, 1]`; `recover_pelvis_3d`, `compute_mpjpe_abs`, `bedlam_metric.py` see no change.

---

## Files to Change

1. `pose3d_transformer_head.py` — add new kwargs to `Pose3dTransformerHead.__init__`; add `uv_heatmap_proj` module; branch in `forward()` to compute heatmap-based pelvis UV; branch in `loss()` to add heatmap loss.
2. `pelvis_utils.py` — add `uv_to_grid_coords(...)` and `build_gaussian_heatmap_2d(...)` helpers.
3. `config.py` — add new kwargs to the `head` dict.

---

## `pelvis_utils.py` Changes

Add the following two helper functions at the end of the module (keep all existing functions unchanged):

```python
def uv_to_grid_coords(uv_norm: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
    """Convert (u_norm, v_norm) in [-1, 1] to (row, col) feature-grid coordinates.

    Args:
        uv_norm: (..., 2) tensor, last dim is (u, v) in [-1, 1].
        feat_h: feature map height (e.g., 40).
        feat_w: feature map width  (e.g., 24).

    Returns:
        (..., 2) tensor, last dim is (row, col) in float grid units.
        row in [0, feat_h-1], col in [0, feat_w-1] when uv_norm in [-1, 1].
    """
    u_grid = (uv_norm[..., 0] + 1.0) * 0.5 * (feat_w - 1)   # col in [0, W-1]
    v_grid = (uv_norm[..., 1] + 1.0) * 0.5 * (feat_h - 1)   # row in [0, H-1]
    return torch.stack([v_grid, u_grid], dim=-1)            # (..., 2): (row, col)


def build_gaussian_heatmap_2d(
    center_hw: torch.Tensor,   # (B, 2) float (row, col) grid coords
    feat_h: int,
    feat_w: int,
    sigma: float,
) -> torch.Tensor:
    """Build an L1-normalized (sum=1) Gaussian target heatmap flattened to (B, feat_h*feat_w).

    Args:
        center_hw: (B, 2) float tensor, (row, col) grid coords.
        feat_h, feat_w: feature-grid dimensions.
        sigma: Gaussian std in grid-cell units.
    """
    B = center_hw.shape[0]
    device = center_hw.device
    dtype = center_hw.dtype
    h_idx = torch.arange(feat_h, device=device, dtype=dtype)
    w_idx = torch.arange(feat_w, device=device, dtype=dtype)
    grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')   # (H, W)
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)       # (H*W, 2)
    mu = center_hw.unsqueeze(1)                                     # (B, 1, 2)
    g = grid.unsqueeze(0)                                           # (1, H*W, 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)                             # (B, H*W)
    hm = torch.exp(-dist2 / (2.0 * sigma ** 2))
    hm = hm / hm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return hm                                                       # (B, H*W)
```

Both helpers operate on the same dtype/device as their inputs; no autocast surprises.

---

## `pose3d_transformer_head.py` Changes

### 1. Imports at top of file

Ensure `torch.nn.functional as F` is imported (baseline already imports it). Import the two new helpers:

```python
from pelvis_utils import (
    # existing imports unchanged
    uv_to_grid_coords,
    build_gaussian_heatmap_2d,
)
```

If the baseline's existing `from pelvis_utils import ...` line does not exist (baseline imports by module or by specific names), add a new import line for the two helpers alongside existing ones. Do not change any other import in the file.

### 2. Add new kwargs to `Pose3dTransformerHead.__init__`

Add the following parameters to the signature (all defaulting to baseline-equivalent values so the non-heatmap path is preserved):

```python
use_uv_heatmap: bool = False,
uv_heatmap_loss_weight: float = 0.5,
uv_heatmap_sigma: float = 2.0,
uv_heatmap_target: str = 'gaussian',
uv_heatmap_learnable_temp: bool = False,
feat_h: int = 40,
feat_w: int = 24,
```

Store them as attributes in `__init__`:

```python
self.use_uv_heatmap = use_uv_heatmap
self.uv_heatmap_loss_weight = float(uv_heatmap_loss_weight)
self.uv_heatmap_sigma = float(uv_heatmap_sigma)
self.uv_heatmap_target = str(uv_heatmap_target)
self.uv_heatmap_learnable_temp = bool(uv_heatmap_learnable_temp)
self.feat_h = int(feat_h)
self.feat_w = int(feat_w)
```

### 3. Replace UV head module construction when `use_uv_heatmap=True`

Where the baseline constructs `self.uv_out = nn.Linear(hidden_dim, 2)`, change to:

```python
if self.use_uv_heatmap:
    self.uv_heatmap_proj = nn.Linear(hidden_dim, 1)
    nn.init.zeros_(self.uv_heatmap_proj.weight)
    nn.init.zeros_(self.uv_heatmap_proj.bias)
    if self.uv_heatmap_learnable_temp:
        self.uv_heatmap_temp = nn.Parameter(torch.tensor(1.0))
    self.uv_out = None  # explicitly disabled; guarded in forward()
else:
    self.uv_out = nn.Linear(hidden_dim, 2)
```

For Design A, `uv_heatmap_learnable_temp=False`, so `self.uv_heatmap_temp` is not created.

### 4. Modify `forward()` to branch on `use_uv_heatmap`

After the `spatial = spatial + pos_enc` line (and before the decoder is invoked), do NOT modify the spatial tokens fed into the decoder. The heatmap branch reads from the same `spatial` tensor independently of the decoder path.

At the point where the baseline computes `pelvis_uv = self.uv_out(pelvis_token)`, replace with:

```python
if self.use_uv_heatmap:
    H, W = self.feat_h, self.feat_w
    # spatial: (B, H*W, hidden_dim); baseline layout is row-major (row=h outer, col=w inner)
    assert spatial.shape[1] == H * W, (
        f"spatial token count {spatial.shape[1]} != feat_h*feat_w={H*W}; "
        "update feat_h/feat_w kwargs."
    )
    uv_logits = self.uv_heatmap_proj(spatial).squeeze(-1)    # (B, H*W)
    if self.uv_heatmap_learnable_temp:
        temp = F.softplus(self.uv_heatmap_temp).clamp(min=1e-3)
        uv_attn = F.softmax(uv_logits / temp, dim=-1)
    else:
        uv_attn = F.softmax(uv_logits, dim=-1)               # (B, H*W)

    attn_hw = uv_attn.view(-1, H, W)                         # (B, H, W)
    h_idx = torch.arange(H, device=spatial.device, dtype=attn_hw.dtype)
    w_idx = torch.arange(W, device=spatial.device, dtype=attn_hw.dtype)
    # Marginal over W gives row distribution; marginal over H gives col distribution.
    v_frac = (attn_hw.sum(dim=-1) * h_idx).sum(dim=-1) / max(H - 1, 1)   # (B,) in [0, 1]
    u_frac = (attn_hw.sum(dim=-2) * w_idx).sum(dim=-1) / max(W - 1, 1)   # (B,) in [0, 1]
    pelvis_uv = torch.stack(
        [u_frac * 2.0 - 1.0, v_frac * 2.0 - 1.0], dim=-1
    )                                                         # (B, 2) in [-1, 1]

    # Stash the distribution for loss() to consume.
    self._uv_attn = uv_attn                                   # (B, H*W)
else:
    pelvis_uv = self.uv_out(pelvis_token)                     # baseline unchanged
    self._uv_attn = None
```

`pelvis_token` is the same token the baseline uses — no change to how it is obtained or consumed for other outputs (pelvis depth regression remains unchanged).

The return value of `forward()` (the pred dict) is unchanged in shape and keys: `{'joints': ..., 'pelvis_depth': ..., 'pelvis_uv': pelvis_uv}`.

### 5. Modify `loss()` to add the heatmap loss

After the existing UV SmoothL1 loss is computed (keep that term as-is with its existing weight 1.0), add:

```python
if self.use_uv_heatmap and self.uv_heatmap_loss_weight > 0.0 and self._uv_attn is not None:
    # gt_uv: (B, 2) in [-1, 1] — same tensor used by the SmoothL1 UV loss.
    gt_grid = uv_to_grid_coords(gt_uv, self.feat_h, self.feat_w)   # (B, 2) (row, col)
    gt_hm = build_gaussian_heatmap_2d(
        gt_grid, self.feat_h, self.feat_w, self.uv_heatmap_sigma,
    )                                                              # (B, H*W)
    log_attn = torch.log(self._uv_attn.clamp(min=1e-8))            # (B, H*W)
    heatmap_loss = -(gt_hm * log_attn).sum(dim=-1).mean()
    losses['loss/uv_heatmap/train'] = (
        self.uv_heatmap_loss_weight * heatmap_loss
    )
self._uv_attn = None
```

Notes:
- `gt_uv` is the same ground-truth tensor already referenced by the baseline SmoothL1 UV loss; reuse the existing variable.
- `losses` is the loss dict the baseline `loss()` returns; the new key `loss/uv_heatmap/train` follows the existing `loss/...` naming convention in the file.
- The target `gt_hm` is detached (no gradient through the target); `build_gaussian_heatmap_2d` does not track gradients on `center_hw` for training stability, so the Designer expects Builder to call it with `gt_grid.detach()` if `gt_uv` ever carries grad. For BEDLAM2 GT loaded from NPZ, `gt_uv` has no grad and no detach is needed, but the Builder should add a `.detach()` on `gt_grid` defensively.

### 6. `predict()` requires no changes

`predict()` calls `self.forward(feats)` and reads `pred['pelvis_uv']` — shape and semantics unchanged.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add these kwargs after `loss_weight_uv=1.0` (keep all existing fields unchanged):

```python
use_uv_heatmap=True,
uv_heatmap_loss_weight=0.5,
uv_heatmap_sigma=2.0,
uv_heatmap_target='gaussian',
uv_heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

All values are bool/int/float/str literals. No Python `import` statements. MMEngine config constraint fully satisfied.

Everything else in `config.py` (optimizer, LR schedule, data pipeline, hooks) is unchanged.

---

## Invariants to Preserve

- `persistent_workers=False` — unchanged.
- Loss restricted to body joints (indices 0–21) — unchanged; this design does not touch the joint loss.
- `resume=True` and `CheckpointHook` with `max_keep_ckpts=1` — unchanged.
- `accumulative_counts=8`, `batch_size=4` — unchanged.
- `seed=2026` — unchanged.
- AMP via `FixedAmpOptimWrapper` — unchanged.
- No Python `import` statements in `config.py` — satisfied (all new values are literals).
- Absolute imports in `pose3d_transformer_head.py` — unchanged.
- `pred['pelvis_uv']` shape `(B, 2)` in `[-1, 1]` — preserved.
- `recover_pelvis_3d`, `compute_mpjpe_abs`, `bedlam_metric.py` — not touched, and receive identical input shapes/ranges.
- Pelvis depth head (scalar regression) — unchanged in this design.

---

## Expected Behaviour After Change

- At init: `uv_heatmap_proj` is zero → `uv_logits = 0` → `uv_attn` uniform over 960 cells → `u_frac = v_frac = 0.5` → `pelvis_uv = (0, 0)`. For BEDLAM2 person-centred crops, GT pelvis UV concentrates near `(0, 0)`, so the zero-init start is close in expectation.
- Heatmap loss gradient: cross-entropy on a uniform distribution vs a sharp Gaussian target produces well-defined, non-pathological gradients; logits begin to peak toward the GT cell within the first tens of iterations.
- Soft-argmax gradient: the continuous SmoothL1 loss also pushes `u_frac, v_frac` toward GT, reinforcing the classification signal.
- Parameter count delta: `uv_out = Linear(256, 2)` (514 params) replaced by `uv_heatmap_proj = Linear(256, 1)` (257 params). Net: −257 params. Negligible.
- Memory: `uv_logits`, `uv_attn`, `gt_hm` each `(B=4, 960)` ≈ 15 KB in fp16. Negligible.
- Speed: +1 linear over 960 tokens + 1 softmax + 2 reductions per step. <0.5 ms overhead on 2080 Ti.
- Shape/interface of `pred['pelvis_uv']` unchanged: `(B, 2)` float in `[-1, 1]`.

---

## Edge Cases and Constraints

- **Row/col convention**: baseline flattens spatial features as `feat.flatten(2).transpose(1, 2)` with `feat` shape `(B, C, H, W)`. This gives row-major order: `spatial[b, h*W + w]` is the token at grid `(row=h, col=w)`. The `attn_hw.view(-1, H, W)` reshape matches this. `v_frac` (vertical, row) comes from the H-axis marginal; `u_frac` (horizontal, col) comes from the W-axis marginal. The Builder MUST NOT swap H and W in either the reshape or the soft-argmax reductions.
- **Normalized UV convention**: `u_norm = u_pixel / crop_w * 2 - 1`, `v_norm = v_pixel / crop_h * 2 - 1` (same as baseline). `uv_to_grid_coords` maps `u_norm → col ∈ [0, W-1]` and `v_norm → row ∈ [0, H-1]`.
- **GT UV outside `[-1, 1]`**: `uv_to_grid_coords` may produce out-of-range row/col for occluded/out-of-frame pelvis. `build_gaussian_heatmap_2d` still returns a valid (possibly near-zero) normalized heatmap because `hm.sum().clamp(min=1e-6)` prevents division by zero. The Builder should not add any clipping beyond this; the Gaussian-over-grid naturally handles out-of-range centres with small probability mass on the boundary cells.
- **Single-pelvis assumption**: BEDLAM2 has one subject per crop; `gt_uv` is `(B, 2)`. The heatmap loss operates over the single GT centre per sample. No multi-peak handling needed.
- **AMP dtype**: `uv_logits` and `uv_attn` are fp16 under AMP. Log is taken on clamped probabilities ≥ 1e-8; this is safe in fp16 (log(1e-8) ≈ -18.4, well within fp16 range).
- **`self._uv_attn` lifetime**: set in `forward()`, consumed in `loss()`, cleared after use. Because `predict()` also calls `forward()` but never calls `loss()`, ensure `self._uv_attn` does not leak across iterations — initialize it to `None` in `__init__` and clear it at the end of `loss()` (as above).
- **Baseline path untouched**: when `use_uv_heatmap=False` (the default), every added code path is gated by `if self.use_uv_heatmap`, and `self.uv_out` is constructed and used exactly as in baseline. Parameter count and numerical behaviour in the `False` branch are bit-exact with baseline.

---

## Target Metrics (Stage 1)

- `composite_val < 325` (vs. baseline ~335; best prior 323.75)
- `mpjpe_pelvis_val < 600` (vs. baseline 652; best prior 608)
- `mpjpe_abs_val < 780` (vs. baseline 833)
- `mpjpe_body_val` not expected to regress (target: ≤ baseline).
