# Design 002 — Bilateral Symmetry Loss, λ=0.5, Distal-Limb Focused Pair Weights

**Design Description:** Bilateral symmetry loss (λ=0.5) with per-pair weights upweighting distal joints (wrists, ankles, feet) 4× over proximal joints (hips, collars).

**Starting Point:** `baseline/`

---

## Coordinate System and Joint Index Verification

Identical to Design 001. BEDLAM2 convention: X=forward, Y=left, Z=up. Mirror axis = Y (index 1).

Verified 8 symmetric pairs in body range (0–21):
```
[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]
```
Pair indices correspond to: [L-hip/R-hip, L-knee/R-knee, L-ankle/R-ankle, L-foot/R-foot, L-collar/R-collar, L-shoulder/R-shoulder, L-elbow/R-elbow, L-wrist/R-wrist].

**Correction from idea.md:** idea.md assumed 10 pairs including eyes, ears, heels in the body range — incorrect for the actual BEDLAM2 SMPL-X 22-joint body set. The correct count is 8 pairs (verified from `claude_code/data/constants.py` and `bedlam2_transforms.py::_FLIP_PAIRS`).

---

## Algorithm

The bilateral symmetry consistency algorithm is identical to Design 001 with the addition of static per-pair weights:

1. For each symmetric pair `(i, j)`, compute `asym_pred = pred_joints[:, i] - mirror(pred_joints[:, j])` and `asym_gt = gt_joints[:, i] - mirror(gt_joints[:, j])`. Mirror negates Y (index 1) for BEDLAM2's Y=left convention.
2. Compute element-wise smooth-L1 (beta=0.05) on `asym_pred - asym_gt`, yielding `sym_loss` of shape `(B, P, 3)`.
3. Multiply `sym_loss` by the per-pair weight tensor broadcast as `(1, P, 1)`: distal pairs (ankle, foot, wrist) get weight 2.0; proximal pairs (hip, collar) get weight 0.5; mid-limb pairs get intermediate weights.
4. Take the mean over all elements (B, P, 3) after weighting, then scale by `bilateral_sym_loss_weight=0.5`. Add to `losses['loss/sym/train']`.

The per-pair weighting is a static tensor registered as a buffer — it does not change during training.

---

## Per-Pair Weight Specification

Design 002 applies per-pair scalar weights to focus the symmetry loss on distal joints where bilateral errors are most costly:

| Pair index | Joint pair             | Weight | Rationale                          |
|------------|------------------------|--------|------------------------------------|
| 0          | L-hip / R-hip          | 0.5    | Proximal; well-predicted           |
| 1          | L-knee / R-knee        | 1.0    | Mid-limb; moderate weight          |
| 2          | L-ankle / R-ankle      | 2.0    | Distal; high error; upweighted     |
| 3          | L-foot / R-foot        | 2.0    | Distal (ball of foot); upweighted  |
| 4          | L-collar / R-collar    | 0.5    | Proximal; well-predicted           |
| 5          | L-shoulder / R-shoulder| 1.0    | Mid-upper; moderate weight         |
| 6          | L-elbow / R-elbow      | 1.5    | Distal arm; moderate-high weight   |
| 7          | L-wrist / R-wrist      | 2.0    | Most distal; highest error; upweighted |

Config literal: `sym_pair_weights=[0.5, 1.0, 2.0, 2.0, 0.5, 1.0, 1.5, 2.0]`

**Deviation from idea.md:** idea.md listed 10 pairs with weights `[0.5,1.0,2.0,0.5,1.0,2.0,2.0,0.5,0.5,2.0]` (shoulder, elbow, wrist, hip, knee, ankle, ball, eye, ear, heel). With 8 corrected pairs, the weights are re-mapped: hip=0.5, knee=1.0, ankle=2.0, foot=2.0, collar=0.5, shoulder=1.0, elbow=1.5, wrist=2.0. The overall design intent (upweight distal joints 4× over proximal) is preserved.

---

## Changes Required

### Files to Modify

1. `pose3d_transformer_head.py` — same additions as Design 001 (all six new kwargs)
2. `config.py` — pass Design 002 specific values including `sym_pair_weights`

### File: `pose3d_transformer_head.py`

**Identical changes to Design 001.** Add the same six kwargs to `__init__`, same buffer registration block, and same loss block. The loss block handles both `sym_pair_weights_buf is not None` (Design 002) and `None` (Design 001) via the existing conditional check. No additional code changes beyond what Design 001 specifies.

Complete `__init__` additions (after `self.loss_weight_uv = loss_weight_uv`):

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

Complete loss block (appended after `losses['loss/uv/train']`, before `with torch.no_grad():`):

```python
# ── Bilateral Symmetry Consistency Loss ──────────────────────────────────
if self.bilateral_sym_loss_weight > 0.0 and self.sym_pairs_buf is not None:
    left_idx  = self.sym_pairs_buf[:, 0]   # (P,)
    right_idx = self.sym_pairs_buf[:, 1]   # (P,)

    mirror = torch.ones(3, device=pred['joints'].device)
    mirror[self.sym_mirror_axis] = -1.0

    pred_left  = pred['joints'][:, left_idx]
    pred_right = pred['joints'][:, right_idx] * mirror
    asym_pred  = pred_left - pred_right             # (B, P, 3)

    gt_left    = gt_joints[:, left_idx]
    gt_right   = gt_joints[:, right_idx] * mirror
    asym_gt    = gt_left - gt_right                 # (B, P, 3)

    asym_diff  = asym_pred - asym_gt
    beta_sym   = 0.05
    abs_diff   = asym_diff.abs()
    sym_loss   = torch.where(
        abs_diff < beta_sym,
        0.5 * abs_diff ** 2 / beta_sym,
        abs_diff - 0.5 * beta_sym,
    )   # (B, P, 3)

    # Per-pair weights: (1, P, 1) broadcast over (B, P, 3)
    if self.sym_pair_weights_buf is not None:
        sym_loss = sym_loss * self.sym_pair_weights_buf.view(1, -1, 1)

    if self.sym_adaptive_weight:
        with torch.no_grad():
            asym_gt_mag = asym_gt.detach().norm(dim=-1, keepdim=True)
            asym_w = 1.0 / (1.0 + asym_gt_mag / self.sym_tau)
        sym_loss = sym_loss * asym_w

    losses['loss/sym/train'] = self.bilateral_sym_loss_weight * sym_loss.mean()
```

### File: `config.py`

In `model.head` dict, after `loss_weight_uv=1.0`:

```python
bilateral_sym_loss_weight=0.5,
sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]],
sym_mirror_axis=1,
sym_pair_weights=[0.5, 1.0, 2.0, 2.0, 0.5, 1.0, 1.5, 2.0],
```

Do **not** add `sym_adaptive_weight` or `sym_tau` — their defaults (`False`, `0.1`) are correct for Design 002.

All values are int/float/list literals. No Python import statements. MMEngine-compliant.

---

## Expected Behaviour

- The per-pair weight tensor is registered as a buffer `sym_pair_weights_buf` of shape `(8,)` and broadcast as `(1, 8, 1)` against the `(B, 8, 3)` sym_loss tensor.
- Effective symmetry loss contribution for wrist pair at λ=0.5, weight=2.0: `0.5 * 2.0 * (3 terms) / total_terms`. The mean is computed over all elements after weighting, so higher-weight pairs contribute proportionally more gradient.
- Stage-1 expected: `mpjpe_body_val < 188 mm`, primary gain on wrist/ankle MPJPE (relative joints). `mpjpe_pelvis_val` unchanged.
- At λ=0.5 with mean pair weight ≈ 1.31 (average of [0.5,1.0,2.0,2.0,0.5,1.0,1.5,2.0]), effective scale ≈ 0.5 × 1.31 × 24/66 ≈ 24% of joint loss — within safe range.

---

## Constraints and Edge Cases

Identical to Design 001:
1. `gt_joints` is full `(B, 70, 3)` at the point of the symmetry loss block.
2. `sym_pair_weights_buf` shape `(8,)` — must match `sym_pairs` length exactly. Builder must verify `len(sym_pairs) == len(sym_pair_weights)`.
3. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.
4. `register_buffer('sym_pair_weights_buf', w_tensor)` — registered as persistent buffer so it moves to the correct device automatically. Do not store as plain attribute.
