# Design 001 — Bilateral Symmetry Consistency Loss, λ=0.3 (Uniform Pairs)

**Design Description:** Add a uniform-weight bilateral symmetry loss (λ=0.3) over 8 verified symmetric body joint pairs to couple L/R gradient flow.

**Starting Point:** `baseline/`

---

## Coordinate System and Joint Index Verification

BEDLAM2 convention (confirmed from `pelvis_utils.py`): **X=forward (depth), Y=left, Z=up**.

Left-right mirror axis is **Y (index 1)**: negate Y to flip left↔right.

The 22 body joints in active-index space (indices 0–21 of the 70-joint set, after `SubtractRootJoint`) follow the SMPL-X ordering verified from `claude_code/data/constants.py`:

| Index | Joint Name     |
|-------|----------------|
| 0     | pelvis (root)  |
| 1     | left_hip       |
| 2     | right_hip      |
| 3     | spine1         |
| 4     | left_knee      |
| 5     | right_knee     |
| 6     | spine2         |
| 7     | left_ankle     |
| 8     | right_ankle    |
| 9     | spine3         |
| 10    | left_foot      |
| 11    | right_foot     |
| 12    | neck           |
| 13    | left_collar    |
| 14    | right_collar   |
| 15    | head           |
| 16    | left_shoulder  |
| 17    | right_shoulder |
| 18    | left_elbow     |
| 19    | right_elbow    |
| 20    | left_wrist     |
| 21    | right_wrist    |

Singleton joints (no symmetric partner in body range): pelvis (0), spine1 (3), spine2 (6), spine3 (9), neck (12), head (15).

**Verified symmetric pairs (8 pairs):**
```
[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]
```
Ordering: [L-hip/R-hip, L-knee/R-knee, L-ankle/R-ankle, L-foot/R-foot, L-collar/R-collar, L-shoulder/R-shoulder, L-elbow/R-elbow, L-wrist/R-wrist].

**Correction from idea.md:** The idea.md listed 10 pairs based on an incorrect assumption that eyes, ears, and heels appear in the 22-joint body range. In the actual BEDLAM2/SMPL-X 70-joint active set, eyes are at indices 22–23 (active space), heels are at indices 65–70 (surface landmarks), and the body range (0–21) contains only 8 symmetric pairs. This design uses the 8 correct verified pairs.

---

## Algorithm

The bilateral symmetry consistency algorithm operates as follows in `loss()`:

1. For each symmetric pair `(i, j)`, compute the predicted asymmetry vector: `asym_pred = pred_joints[:, i] - mirror(pred_joints[:, j])`.
2. Compute the GT asymmetry vector: `asym_gt = gt_joints[:, i] - mirror(gt_joints[:, j])`.
3. Mirror = negate the Y-axis (index 1) of the right joint's coordinates, converting the raw L–R difference into a signed bilateral asymmetry in BEDLAM2's X=forward, Y=left, Z=up frame.
4. Apply element-wise smooth-L1 (beta=0.05) to `asym_pred - asym_gt`, then take the mean over batch, pairs, and coordinates.
5. Scale by `bilateral_sym_loss_weight` and add to the losses dict as `'loss/sym/train'`.

Design 001 uses uniform weights across all 8 pairs (no per-pair scaling, no adaptive weighting).

---

## Changes Required

### Files to Modify

1. `pose3d_transformer_head.py` — add bilateral symmetry loss to `__init__` and `loss()`
2. `config.py` — pass new kwargs to the head config

### File: `pose3d_transformer_head.py`

#### `__init__` signature change

Add the following keyword arguments to `Pose3dTransformerHead.__init__` **after** `loss_weight_uv`:

```python
bilateral_sym_loss_weight: float = 0.0,
sym_pairs: list = None,
sym_mirror_axis: int = 1,
sym_pair_weights: list = None,
sym_adaptive_weight: bool = False,
sym_tau: float = 0.1,
```

All defaults reproduce baseline behaviour when `bilateral_sym_loss_weight=0.0`.

#### `__init__` body additions

After the existing `self.loss_weight_uv = loss_weight_uv` assignment, add:

```python
self.bilateral_sym_loss_weight = bilateral_sym_loss_weight
self.sym_mirror_axis = sym_mirror_axis
self.sym_adaptive_weight = sym_adaptive_weight
self.sym_tau = sym_tau

if sym_pairs is not None:
    pairs_tensor = torch.tensor(sym_pairs, dtype=torch.long)  # (P, 2)
    self.register_buffer('sym_pairs_buf', pairs_tensor)
    if sym_pair_weights is not None:
        w_tensor = torch.tensor(sym_pair_weights, dtype=torch.float32)  # (P,)
        self.register_buffer('sym_pair_weights_buf', w_tensor)
    else:
        self.sym_pair_weights_buf = None
else:
    self.sym_pairs_buf = None
    self.sym_pair_weights_buf = None
```

#### `loss()` body additions

Append the following block **after** `losses['loss/uv/train'] = ...` and **before** the `with torch.no_grad():` MPJPE block:

```python
# ── Bilateral Symmetry Consistency Loss ──────────────────────────────────
if self.bilateral_sym_loss_weight > 0.0 and self.sym_pairs_buf is not None:
    left_idx  = self.sym_pairs_buf[:, 0]   # (P,) — left joint indices
    right_idx = self.sym_pairs_buf[:, 1]   # (P,) — right joint indices

    # Build mirror scale: negate the sym_mirror_axis (Y=1 for BEDLAM2)
    mirror = torch.ones(3, device=pred['joints'].device)
    mirror[self.sym_mirror_axis] = -1.0

    # Predicted asymmetry: pred_left - mirror(pred_right),  shape (B, P, 3)
    pred_left  = pred['joints'][:, left_idx]                    # (B, P, 3)
    pred_right = pred['joints'][:, right_idx] * mirror          # (B, P, 3)
    asym_pred  = pred_left - pred_right

    # GT asymmetry: same operation on GT joints, shape (B, P, 3)
    gt_left    = gt_joints[:, left_idx]                         # (B, P, 3)
    gt_right   = gt_joints[:, right_idx] * mirror               # (B, P, 3)
    asym_gt    = gt_left - gt_right

    # Smooth-L1 on asymmetry error, beta=0.05 (matching joint loss)
    asym_diff  = asym_pred - asym_gt                            # (B, P, 3)
    beta_sym   = 0.05
    abs_diff   = asym_diff.abs()
    sym_loss   = torch.where(
        abs_diff < beta_sym,
        0.5 * abs_diff ** 2 / beta_sym,
        abs_diff - 0.5 * beta_sym,
    )   # (B, P, 3)

    # Per-pair weights (None for Design 001 — uniform)
    if self.sym_pair_weights_buf is not None:
        sym_loss = sym_loss * self.sym_pair_weights_buf.view(1, -1, 1)

    # Adaptive GT-magnitude weighting (False for Design 001)
    if self.sym_adaptive_weight:
        with torch.no_grad():
            asym_gt_mag = asym_gt.detach().norm(dim=-1, keepdim=True)  # (B,P,1)
            asym_w = 1.0 / (1.0 + asym_gt_mag / self.sym_tau)
        sym_loss = sym_loss * asym_w

    losses['loss/sym/train'] = self.bilateral_sym_loss_weight * sym_loss.mean()
```

**Placement constraint:** This block must appear after the three existing loss lines and before the `with torch.no_grad():` MPJPE block. The `gt_joints` variable is already in scope at that point (extracted earlier in the method).

**Key invariants to preserve:**
- Loss is keyed `'loss/sym/train'` — separate from `'loss/joints/train'`. MMEngine will sum all `losses` values automatically.
- `gt_joints` used here is the **full** (B, 70, 3) tensor (before the `_BODY` slice used for the joint loss). Indexing `[:, left_idx]` with left_idx values 1–21 is valid.
- The `mirror` tensor is constructed on the same device as `pred['joints']` — no device mismatch.
- AMP (float16) is active: `torch.where` is safe with float16 tensors. No explicit dtype cast needed.

### File: `config.py`

In the `model` dict, under `head=dict(...)`, add the following kwargs after `loss_weight_uv=1.0`:

```python
bilateral_sym_loss_weight=0.3,
sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]],
sym_mirror_axis=1,
```

Do **not** add `sym_pair_weights`, `sym_adaptive_weight`, or `sym_tau` — their defaults are correct for Design 001.

All values are int/float/list-of-list-of-int literals. No Python import statements. MMEngine-compliant.

---

## Expected Behaviour

- Stage-1 (20 epochs): bilateral symmetry loss couples gradients between the 8 L/R joint pairs. The loss key `loss/sym/train` appears in training logs. Expected `mpjpe_body_val < 190 mm` at stage-1 (diagnostic; any improvement over baseline 195 mm indicates the coupling mechanism works).
- `mpjpe_pelvis_val` unchanged (symmetry loss does not touch pelvis predictions).
- At `bilateral_sym_loss_weight=0.0` (baseline), no symmetry loss is computed and behaviour is identical to baseline.
- The effective symmetry loss magnitude at λ=0.3 with 8 pairs × 3 coords = 24 terms is approximately 24/66 × 0.3 ≈ 11% of the joint loss magnitude — safe scale, no instability expected.

---

## Constraints and Edge Cases

1. **`gt_joints` shape**: the baseline extracts `gt_joints` as `(B, 70, 3)` after `squeeze(1)`. Confirm this before indexing with `left_idx` (values 1–21). If `gt_joints` is somehow `(B, 22, 3)` (sliced), update indices accordingly — but the baseline code does not slice before the loss block.
2. **Buffer registration order**: `register_buffer` calls in `__init__` must not conflict with existing buffers (`pos_enc` is registered lazily in `_get_pos_enc`, not in `__init__`). No conflict.
3. **`sym_pair_weights_buf` None guard**: when `sym_pair_weights` is not passed (Design 001), `self.sym_pair_weights_buf` is set to `None` (not a buffer). The `if self.sym_pair_weights_buf is not None:` check is safe.
4. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.
