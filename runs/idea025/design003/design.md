# Design 003 — Bilateral Symmetry Loss, λ=0.5, GT-Magnitude Adaptive Weighting

**Design Description:** Bilateral symmetry loss (λ=0.5) with per-sample GT-asymmetry-magnitude adaptive weights: near-symmetric poses get full weight, highly asymmetric poses get reduced weight via `w = 1 / (1 + |asym_gt| / tau)`, tau=0.1 m.

**Starting Point:** `baseline/`

---

## Coordinate System and Joint Index Verification

Identical to Design 001. BEDLAM2 convention: X=forward, Y=left, Z=up. Mirror axis = Y (index 1).

Verified 8 symmetric pairs in body range (0–21):
```
[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]
```
[L-hip/R-hip, L-knee/R-knee, L-ankle/R-ankle, L-foot/R-foot, L-collar/R-collar, L-shoulder/R-shoulder, L-elbow/R-elbow, L-wrist/R-wrist].

**Correction from idea.md:** Same as Design 001 — 8 verified pairs, not 10. Eyes/ears/heels are outside the body range (0–21).

---

## Algorithm

The bilateral symmetry consistency algorithm with GT-magnitude adaptive weighting:

1. For each symmetric pair `(i, j)`, compute `asym_pred = pred_joints[:, i] - mirror(pred_joints[:, j])` and `asym_gt = gt_joints[:, i] - mirror(gt_joints[:, j])`. Mirror negates Y (index 1) for BEDLAM2's Y=left convention.
2. Compute element-wise smooth-L1 (beta=0.05) on `asym_pred - asym_gt`, yielding `sym_loss` of shape `(B, P, 3)`.
3. Under `torch.no_grad()`, compute the GT asymmetry magnitude per (sample, pair): `asym_gt_mag = ||asym_gt||_2`, shape `(B, P, 1)`.
4. Compute adaptive weight: `asym_w = 1 / (1 + asym_gt_mag / tau)` with `tau=0.1 m`. This gives weight ∈ (0, 1]: 1.0 for symmetric poses, approaching 0 for highly asymmetric poses.
5. Multiply `sym_loss` by `asym_w` (broadcast `(B, P, 1)` over `(B, P, 3)`).
6. Take the mean, scale by `bilateral_sym_loss_weight=0.5`, add to `losses['loss/sym/train']`.

The adaptive weight is a per-sample, per-pair scalar derived from GT data under no-grad — not a learnable parameter.

---

## Adaptive Weighting Mechanism

For each sample in the batch and each symmetric pair, compute a soft weight that is inversely proportional to the GT asymmetry magnitude:

```
asym_gt_mag = ||asym_gt||_2  per (sample, pair)     shape: (B, P)
asym_w = 1.0 / (1.0 + asym_gt_mag / tau)            shape: (B, P)
```

With `tau = 0.1` (metres = 100 mm):
- When GT asymmetry is 0 mm (perfectly symmetric pose): `asym_w = 1.0` (full weight).
- When GT asymmetry is 100 mm (1× tau): `asym_w = 0.5` (half weight).
- When GT asymmetry is 300 mm (3× tau): `asym_w ≈ 0.25` (quarter weight).

This prevents penalising correct predictions of genuinely large asymmetries (e.g., one arm raised) while maximising the coupling signal for symmetric-posture frames (where bilateral confusion is the primary failure mode).

The adaptive weight is computed inside `torch.no_grad()` using `.detach()` to prevent gradient flow through the weighting itself — the weight is a scalar modifier, not a learnable parameter.

---

## Changes Required

### Files to Modify

1. `pose3d_transformer_head.py` — same additions as Designs 001/002
2. `config.py` — pass Design 003 specific values

### File: `pose3d_transformer_head.py`

**Identical structure to Designs 001/002.** All six kwargs, same buffer registration, same loss block. The `sym_adaptive_weight=True` flag activates the adaptive path.

New kwargs for `Pose3dTransformerHead.__init__` (after `loss_weight_uv`):

```python
bilateral_sym_loss_weight: float = 0.0,
sym_pairs: list = None,
sym_mirror_axis: int = 1,
sym_pair_weights: list = None,
sym_adaptive_weight: bool = False,
sym_tau: float = 0.1,
```

`__init__` body additions (after `self.loss_weight_uv = loss_weight_uv`):

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

Loss block (append after `losses['loss/uv/train']`, before `with torch.no_grad():`):

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

    # Per-pair weights: not used in Design 003 (sym_pair_weights_buf is None)
    if self.sym_pair_weights_buf is not None:
        sym_loss = sym_loss * self.sym_pair_weights_buf.view(1, -1, 1)

    # Adaptive GT-magnitude weighting (active in Design 003)
    if self.sym_adaptive_weight:
        with torch.no_grad():
            asym_gt_mag = asym_gt.detach().norm(dim=-1, keepdim=True)  # (B, P, 1)
            asym_w = 1.0 / (1.0 + asym_gt_mag / self.sym_tau)         # (B, P, 1)
        sym_loss = sym_loss * asym_w                                    # (B, P, 3)

    losses['loss/sym/train'] = self.bilateral_sym_loss_weight * sym_loss.mean()
```

**Critical implementation note for adaptive weighting:**
- `asym_gt` is a float tensor (dtype matches `gt_joints`, typically float32 under AMP for GT data). `.norm(dim=-1, keepdim=True)` produces shape `(B, P, 1)`.
- `self.sym_tau` is a Python float scalar. Division `asym_gt_mag / self.sym_tau` is safe (scalar broadcast).
- `asym_w` range: strictly in `(0.0, 1.0]` — no division-by-zero possible since denominator ≥ 1.0.
- The `with torch.no_grad()` block contains both the magnitude computation and the weight computation — no gradients flow through `asym_w`.
- `sym_loss * asym_w` broadcasts `(B, P, 3) * (B, P, 1)` — correct.

### File: `config.py`

In `model.head` dict, after `loss_weight_uv=1.0`:

```python
bilateral_sym_loss_weight=0.5,
sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]],
sym_mirror_axis=1,
sym_adaptive_weight=True,
sym_tau=0.1,
```

Do **not** add `sym_pair_weights` — its default (`None`) is correct for Design 003 (no static per-pair weights; the adaptive weighting is sample-level, not pair-level).

All values are int/float/bool/list literals. No Python import statements. MMEngine-compliant.

---

## Expected Behaviour

- `asym_w` is computed per (sample, pair) and applied element-wise before the `.mean()`. For a batch where all poses are nearly symmetric, `asym_w ≈ 1.0` for all pairs (full coupling). For a batch with arms raised asymmetrically, wrist/elbow pairs get `asym_w ≈ 0.2–0.5` (reduced penalty for correctly predicted large asymmetries).
- Stage-1 expected: `mpjpe_body_val < 185 mm`, `composite_val < 330`. The adaptive weighting makes the loss most informative for symmetric-posture frames (which dominate BEDLAM2 standing/walking clips) while not penalising accurate asymmetric predictions.
- `mpjpe_pelvis_val` unchanged.
- Loss key `'loss/sym/train'` appears in training logs; its magnitude should decrease over training as the network learns to correctly predict both symmetric and asymmetric poses.

---

## Constraints and Edge Cases

1. **`gt_joints` shape**: full `(B, 70, 3)` at the loss block — same as Designs 001/002.
2. **`sym_tau` units**: must be in metres, matching the root-relative joint coordinate space (post-`SubtractRootJoint`). `0.1 m = 100 mm` is appropriate — joint displacements in root-relative space are on the order of 0.1–1.5 m.
3. **AMP compatibility**: `asym_gt.detach().norm(...)` — if `asym_gt` is float16 (possible under AMP for GT tensors), `.norm()` is safe in float16. However, `gt_joints` is loaded from numpy as float32 and cast to device — typically remains float32. If float16, the division `asym_gt_mag / self.sym_tau` is float16 / float (Python scalar) → float16, which may lose precision. To be safe, Builder may cast: `asym_gt_mag = asym_gt.detach().float().norm(dim=-1, keepdim=True)` and cast `asym_w` back if needed — but this is optional since the weight is used as a multiplier only.
4. No `sym_pair_weights` buffer is registered in Design 003 — `self.sym_pair_weights_buf = None`. The `if self.sym_pair_weights_buf is not None:` branch is skipped.
5. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.
