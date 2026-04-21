**Idea Name:** Bilateral Symmetry Consistency Loss for Body Joint Pairs

**Approach:** For each symmetric left-right body joint pair, supervise the predicted asymmetry vector (difference between left and right joint after mirroring the X-axis) against the GT asymmetry vector, coupling gradient flow between symmetric joints so the network is penalized specifically for getting L/R relative structure wrong.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

### The Symmetric Joint Bottleneck

The baseline applies a uniform `SoftWeightSmoothL1Loss` independently across all 22 body joints with no coupling between symmetric pairs. The left wrist, right wrist, left ankle, and right ankle are predicted from entirely independent information pathways — each joint query attends to the spatial token grid on its own, with no gradient-level linkage to its bilateral partner.

In practice, symmetric joints are the hardest to regress precisely. Looking at the results:

- **Stage-1 `mpjpe_body_val` plateau**: consistently 183–196 mm across 24 prior ideas. The floor was first broken only by idea023 (heatmap routing, 183.4 mm) and idea008/design002 (body-only decoder, 195.8 mm → not a gain). The plateau is extremely consistent.
- **idea024** (per-joint difficulty weighting) achieved 194.9 mm with design001, 294.9 mm with design002 (instability), and 196.8 mm with design003 — no improvement on the plateau from EMA reweighting alone.
- **idea012** (pairwise 3D distance matrix): improved body MPJPE to 216.9/216.6/226.9 mm — actually **worse** than baseline, because enforcing arbitrary pairwise distances added conflicting gradient signals.
- The persistent plateau at ~183 mm suggests that after ~5 epochs, the loss signal from the joint coordinate loss is insufficient to further separate left and right distal joints that are predicted near each other but on the wrong side.

No prior idea has used the specific structural property of **bilateral symmetry** as a loss-level coupling constraint:

| Idea | Mechanism | Difference from idea025 |
|---|---|---|
| idea006 | Learnable self-attention bias between joint queries | Attention structure; no bilateral pairing; operates pre-decoder |
| idea012 | Pairwise 3D distance matrix loss (all 22×22 pairs) | All pairs equally; no bilateral pairing; no asymmetry direction |
| idea013 | Bone-vector output parameterization | Output rep change; no bilateral pairing |
| idea024 | Per-joint EMA difficulty reweighting | Reweights existing loss; no coupling between joints |
| idea010 | 2D reprojection consistency loss | 2D image-space; no bilateral pairing |

### The Bilateral Symmetry Loss

The human body skeleton has 10 symmetric joint pairs in the 22-joint body set:

```
Pair  Left (i)   Right (j)   Name
  0      1           2       L-shoulder / R-shoulder
  1      3           4       L-elbow    / R-elbow
  2      5           6       L-wrist    / R-wrist
  3      7           8       L-hip      / R-hip
  4      9          10       L-knee     / R-knee
  5     11          12       L-ankle    / R-ankle
  6     13          14       L-ball     / R-ball
  7     15          16       L-eye      / R-eye
  8     17          18       L-ear      / R-ear
  9     19          20       L-heel     / R-heel
```

(Indices 0 = pelvis root, 21 = head-top — bilateral singletons, excluded from pairing.)

Note: the exact joint ordering must be verified by the Designer from the BEDLAM2 dataset constants (`infra/constants.py` or data pipeline). The pairing list above follows the standard 22-joint SMPL body topology. The Designer should hardcode the verified list as a literal `sym_pairs` in `config.py`.

For each symmetric pair `(i, j)` with left joint index `i` and right joint index `j`, the **asymmetry vector** in GT is:

```
asym_gt[pair] = gt_joints[:, i] - mirror(gt_joints[:, j])
             = gt_joints[:, i] - gt_joints[:, j] * mirror_scale
```

where `mirror_scale = tensor([-1., 1., 1.])` flips the Y-axis (left-right in BEDLAM2's X=forward, Y=left, Z=up convention). Mirroring Y reverses the sign of the lateral offset between left and right joints, turning the raw L-R difference into a signed asymmetry that should be zero for a perfectly symmetric pose and nonzero only for genuine pose asymmetry.

The **bilateral symmetry loss** is the mean smooth-L1 between predicted and GT asymmetry vectors:

```
asym_pred[pair] = pred_joints[:, i] - pred_joints[:, j] * mirror_scale
asym_gt[pair]   = gt_joints[:, i]   - gt_joints[:, j]   * mirror_scale

L_sym = SmoothL1(asym_pred, asym_gt)   # averaged over B, num_pairs, 3
```

This is equivalent to penalizing the prediction error on the **relative** left-right structure, above and beyond what the per-joint coordinate loss already penalizes. Critically:

1. **Gradient coupling**: the loss L_sym introduces gradient from joint `i`'s error into the prediction of joint `j` and vice versa — the first bilateral coupling in any prior idea.
2. **Asymmetry focus**: the GT asymmetry vector `asym_gt` is typically small (humans are approximately symmetric). When predicted asymmetry is large and GT is small, the network is penalized for breaking symmetry. This directly targets the common failure mode where the network predicts left-wrist and right-wrist as a symmetric pair reflected about the wrong axis.
3. **Scale-invariant by construction**: the asymmetry vector is a difference of two joints at similar distances from the camera, so it is approximately depth-independent. The loss operates in the root-relative coordinate frame (after SubtractRootJoint), where left-right offsets are on the order of 0.2–0.5 m.
4. **No new architecture, no new parameters**: the loss is purely a training signal computed from predictions and GT already available in `loss()`.
5. **Zero initialization equivalent**: when `sym_loss_weight=0.0` (or when the weight is 0) the behaviour is exactly baseline.

### Why This Will Help

**Grounding in results:**

- **idea008/design002** (body-only decoder): the best `mpjpe_rel_val` at stage-1 is 362 mm (vs. baseline 438mm), achieved by removing 48 hand queries and letting 22 body queries dominate the decoder capacity. However, its composite_val of 333.6 is worse than idea023/design001 (323.8), suggesting that body-joint attention quality is still the limiting factor even without hand query contamination.
- **stage-1 body MPJPE floor at 183 mm**: ideas targeting architecture (001, 003, 011), attention (006, 007, 019, 020, 021, 023), and loss (005, 010, 012, 024) have all converged to the same floor. A new type of structural constraint — bilateral coupling — is the logical next step.
- **idea012 failure (pairwise distance matrix)**: idea012's pairwise loss degraded body MPJPE (216–226 mm vs. baseline 195 mm) because the unconstrained distance matrix added conflicting signals for non-anatomically-paired joints. The bilateral symmetry loss is *selective* (only 10 pairs, chosen for genuine biological symmetry) and *directional* (the mirror convention captures the physical meaning of left-right symmetry), unlike idea012's exhaustive all-pairs approach.
- **idea002/design003 stage-2 body MPJPE = 156.6 mm** (best stage-2 result): the dedicated pelvis query helped decouple pathways. Per-joint bilateral coupling should provide a complementary improvement to the body joint pathway.

---

## Proposed Variations

### Design A — Pure bilateral symmetry loss, λ=0.3 (minimal coupling)

Add a `bilateral_sym_loss_weight: float` kwarg to the head. When nonzero, compute the asymmetry loss over the 10 hardcoded symmetric pairs with weight `λ=0.3`.

The symmetric pairs list is hardcoded as a flat literal list in `config.py`:
```python
sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]],
```

The mirror scale `[-1., 1., 1.]` (flip Y for BEDLAM2's coordinate system) is hardcoded as a float list in config:
```python
sym_mirror_axis=1,  # index of the axis to mirror (Y=1 in BEDLAM2 convention)
```

SmoothL1 beta matching the joint loss: `beta=0.05` (same as baseline joint loss). The loss is averaged over pairs and batch.

Config kwargs: `bilateral_sym_loss_weight=0.3`, `sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]]`, `sym_mirror_axis=1`.

Design A is the minimal diagnostic: does any bilateral coupling help?

### Design B — Distance-weighted bilateral symmetry loss, λ=0.5 (distal-limb focus)

Same as Design A but apply per-pair weights that up-weight distal limb joints (wrists, ankles, heels) and down-weight proximal joints (shoulders, hips) that are already well-predicted:

```python
# Pair weights — hardcoded as float list in config
sym_pair_weights=[0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 2.0, 0.5, 0.5, 2.0]
# pairs: [shoulder, elbow, wrist, hip, knee, ankle, ball, eye, ear, heel]
```

Wrists (weight 2.0), ankles (weight 2.0), balls (weight 2.0), and heels (weight 2.0) are upweighted 4× relative to shoulders and hips (weight 0.5). This focuses the bilateral coupling gradient where it is most needed (distal joints with high error), matching the motivation from idea024 but applied to the structural symmetry constraint rather than absolute per-joint error.

Loss weight: `λ=0.5` (larger because the per-pair weighting reduces effective scale for easy pairs).

Config kwargs: `bilateral_sym_loss_weight=0.5`, `sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]]`, `sym_mirror_axis=1`, `sym_pair_weights=[0.5,1.0,2.0,0.5,1.0,2.0,2.0,0.5,0.5,2.0]`.

### Design C — Asymmetry-magnitude adaptive weighting, λ=0.5 (curriculum-style)

Same as Design A architecture but apply an **adaptive per-pair weight** based on the current GT asymmetry magnitude: pairs where the GT pose is highly asymmetric (large `|asym_gt|`) get lower weight (the asymmetry is genuinely large, so the symmetry constraint is less informative), while pairs where GT is near-symmetric (small `|asym_gt|`) get higher weight (the symmetry constraint is most informative when the body is nearly symmetric).

Concretely:
```python
asym_gt_mag = asym_gt.detach().norm(dim=-1)   # (B, 10) — magnitude per pair per sample
# Soft weight: inversely proportional to GT asymmetry magnitude
# w = 1 / (1 + asym_gt_mag / tau)    tau = 0.1 m (100 mm)
sym_tau = 0.1
asym_weight = 1.0 / (1.0 + asym_gt_mag / sym_tau)  # (B, 10), in [0.5, 1.0] range
```

This is a **curriculum-by-difficulty** approach: symmetric poses (where the constraint should be tight) get full weight; highly asymmetric poses (e.g., one arm raised, one lowered) get reduced weight proportionally. This prevents the loss from penalizing the network for correctly predicting genuine large asymmetries.

Loss weight: `λ=0.5`.

Config kwargs: `bilateral_sym_loss_weight=0.5`, `sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]]`, `sym_mirror_axis=1`, `sym_adaptive_weight=True`, `sym_tau=0.1`.

---

## Implementation Scope

All changes are confined to **`pose3d_transformer_head.py`** and **`config.py`**. No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

### `pose3d_transformer_head.py`

**`__init__` additions:**

```python
# New constructor kwargs (all with defaults matching baseline behaviour):
bilateral_sym_loss_weight: float = 0.0    # 0.0 = baseline (no symmetry loss)
sym_pairs: list = None                     # [[i, j], ...] left-right joint index pairs
sym_mirror_axis: int = 1                   # axis to negate for mirror (Y=1 in BEDLAM2)
sym_pair_weights: list = None              # per-pair weights (Design B); None = uniform
sym_adaptive_weight: bool = False          # GT-magnitude adaptive weighting (Design C)
sym_tau: float = 0.1                       # adaptive weighting scale in metres (Design C)

# Store as buffers/attributes:
self.bilateral_sym_loss_weight = bilateral_sym_loss_weight
self.sym_mirror_axis = sym_mirror_axis
self.sym_adaptive_weight = sym_adaptive_weight
self.sym_tau = sym_tau

if sym_pairs is not None:
    # Register as buffer so it moves to correct device automatically
    pairs_tensor = torch.tensor(sym_pairs, dtype=torch.long)  # (num_pairs, 2)
    self.register_buffer('sym_pairs_buf', pairs_tensor)
    if sym_pair_weights is not None:
        w_tensor = torch.tensor(sym_pair_weights, dtype=torch.float32)
        self.register_buffer('sym_pair_weights_buf', w_tensor)
    else:
        self.sym_pair_weights_buf = None
else:
    self.sym_pairs_buf = None
    self.sym_pair_weights_buf = None
```

**`loss()` additions** (appended after the existing joint/depth/UV losses):

```python
# ── Bilateral Symmetry Consistency Loss ─────────────────────────────────
if self.bilateral_sym_loss_weight > 0.0 and self.sym_pairs_buf is not None:
    left_idx  = self.sym_pairs_buf[:, 0]  # (P,) left joint indices
    right_idx = self.sym_pairs_buf[:, 1]  # (P,) right joint indices

    # Mirror scale: negate the sym_mirror_axis to flip left<->right
    mirror = torch.ones(3, device=pred['joints'].device)
    mirror[self.sym_mirror_axis] = -1.0   # Y-axis flip for BEDLAM2 convention

    # Predicted and GT asymmetry vectors: (B, P, 3)
    pred_left  = pred['joints'][:, left_idx]               # (B, P, 3)
    pred_right = pred['joints'][:, right_idx] * mirror     # (B, P, 3) mirrored
    asym_pred  = pred_left - pred_right                     # (B, P, 3)

    gt_left    = gt_joints[:, left_idx]                    # (B, P, 3)
    gt_right   = gt_joints[:, right_idx] * mirror          # (B, P, 3) mirrored
    asym_gt    = gt_left - gt_right                         # (B, P, 3)

    # Compute smooth-L1 on asymmetry error (same beta as joint loss)
    asym_diff  = asym_pred - asym_gt                        # (B, P, 3)
    beta_sym   = 0.05
    abs_diff   = asym_diff.abs()
    sym_loss   = torch.where(
        abs_diff < beta_sym,
        0.5 * abs_diff ** 2 / beta_sym,
        abs_diff - 0.5 * beta_sym
    )  # (B, P, 3)

    # Per-pair weights (Design B): shape (1, P, 1)
    if self.sym_pair_weights_buf is not None:
        sym_loss = sym_loss * self.sym_pair_weights_buf.view(1, -1, 1)

    # Adaptive GT-magnitude weighting (Design C): shape (B, P, 1)
    if self.sym_adaptive_weight:
        with torch.no_grad():
            asym_gt_mag = asym_gt.detach().norm(dim=-1, keepdim=True)  # (B,P,1)
            asym_w = 1.0 / (1.0 + asym_gt_mag / self.sym_tau)
        sym_loss = sym_loss * asym_w

    losses['loss/sym/train'] = self.bilateral_sym_loss_weight * sym_loss.mean()
```

### `config.py`

**Design A:**
```python
bilateral_sym_loss_weight=0.3,
sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]],
sym_mirror_axis=1,
```

**Design B:**
```python
bilateral_sym_loss_weight=0.5,
sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]],
sym_mirror_axis=1,
sym_pair_weights=[0.5,1.0,2.0,0.5,1.0,2.0,2.0,0.5,0.5,2.0],
```

**Design C:**
```python
bilateral_sym_loss_weight=0.5,
sym_pairs=[[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]],
sym_mirror_axis=1,
sym_adaptive_weight=True,
sym_tau=0.1,
```

All values are bool/int/float/str/list literals. No Python import statements in `config.py`. Fully compliant with MMEngine no-Python-imports restriction.

**Designer Note:** Verify the exact symmetric joint index pairs from `infra/constants.py` or the data pipeline before hardcoding `sym_pairs`. The indices listed above follow standard SMPL 22-joint body topology (the ordering used in BEDLAM2's `lifting_target` output after `SubtractRootJoint`). If the ordering differs, the Designer must update `sym_pairs` accordingly. The mirror axis `sym_mirror_axis=1` assumes BEDLAM2's Y=left convention — verify this against the BEDLAM2 convention comment at the top of `pelvis_utils.py` (`X=forward, Y=left, Z=up`).

---

## Expected Outcome

- **Primary gain — `mpjpe_body_val`**: bilateral coupling targets the stage-1 plateau at 183–196 mm by providing additional gradient pressure on symmetric distal joints. Expected improvement: `mpjpe_body_val < 185` at stage-1 for Designs A/B, potentially `< 182` for Design C. Stage-2 target: `< 155 mm` (approaching best prior of 156.6 mm from idea002/design003).

- **Secondary gain — `mpjpe_rel_val`**: root-relative MPJPE is dominated by distal limb positions. By directly supervising L-R asymmetry of wrists/ankles/heels, `mpjpe_rel_val` should improve from the 414–440 mm range seen in non-body-focused designs toward the 333–362 mm range seen in idea008.

- **Pelvis MPJPE**: unaffected (symmetry loss operates only on body joint predictions; pelvis depth/UV loss is unchanged).

- **Design A** (λ=0.3, uniform pairs): minimal diagnostic. Expected composite_val < 340 at stage-1; primary test of the bilateral coupling mechanism.

- **Design B** (λ=0.5, distal-focused weights): up-weights wrist/ankle/heel pairs where bilateral confusion is most costly. Expected composite_val < 335 at stage-1, with primary gain on `mpjpe_body_val < 185`.

- **Design C** (adaptive GT-magnitude weighting, λ=0.5): the most targeted variant — reduces penalty for genuinely asymmetric poses, maximizes coupling for symmetric-posture frames. Expected composite_val < 328 at stage-1 (competitive with best prior of 323.8 from idea023/design001).

- **Composite target (stage-2)**: aim for `composite_val < 222` across all designs; best case `< 218` for Design C if the bilateral coupling compounds with improved body MPJPE.

---

## Risk and Mitigation

- **Incorrect symmetric joint pair mapping**: if the `sym_pairs` list uses wrong indices, the loss would couple non-symmetric joints (e.g., left wrist with left hip), adding conflicting gradient and degrading performance. **Mitigation**: Designer must verify indices from `infra/constants.py` before hardcoding. A sanity check can be done by printing `gt_joints[:, left_idx].mean()` vs `gt_joints[:, right_idx].mean()` across a batch — symmetric joint pairs should have similar mean absolute values for Y and Z coordinates and similar-magnitude-but-opposite-sign Y offsets.

- **Mirror axis convention**: BEDLAM2 uses X=forward, Y=left, Z=up. The mirror for left-right symmetry should negate Y (`sym_mirror_axis=1`). If the convention is different (e.g., Y=up in some intermediate coordinate frame), the mirror axis must be adjusted. **Mitigation**: cross-check with `pelvis_utils.py` (line: `u = -Y/X * fx + cx`) confirming Y is the lateral axis.

- **Asymmetry loss conflicting with joint loss for genuinely asymmetric poses**: when the person has a highly asymmetric pose (one arm up, one arm down), the GT asymmetry vector is large. The symmetry loss in Design A/B would assign a high weight to this case, penalizing a correct prediction of large asymmetry. **Mitigation**: Design C explicitly addresses this via adaptive weighting. Designs A and B are still safe because the loss targets `asym_pred ≈ asym_gt` — not `asym_pred ≈ 0` — so a correctly predicted large asymmetry contributes zero loss.

- **Gradient scale**: with 10 pairs and 3 coordinates, the symmetry loss has 30 terms vs. 66 terms (22 joints × 3 coords) for the joint loss. At λ=0.3 (Design A), the symmetry loss contributes ~30/66 × 0.3 ≈ 14% of the joint loss magnitude. At λ=0.5 with pair weights 2.0 on 4 distal pairs (Design B), the effective contribution is ~22% for distal pairs. These scales are within safe range and comparable to the auxiliary losses used in ideas 010, 012, and 013.

- **Interaction with idea005 (uncertainty weighting)**: if composed, the per-task uncertainty weight would scale the entire joint loss (including the sym loss if included in the joint task). To keep them orthogonal, the symmetry loss is logged as a separate key (`loss/sym/train`), not folded into `loss/joints/train`. This means idea005's uncertainty weight would not interfere.

- **Interaction with idea008 (body-only decoder)**: idea008 removes hand queries (indices 22–69) and all 10 symmetric pairs are in the body set (indices 0–21), so this idea is fully compatible with idea008. A combined design is a natural future experiment.

- **Interaction with idea024 (per-joint difficulty weighting)**: the EMA difficulty weighting (idea024) redistributes per-joint loss weights among the 22 joints. The bilateral symmetry loss adds a *separate* coupling term that is orthogonal to per-joint difficulty weighting. If combined, the joint loss gradient (from difficulty weighting) and the symmetry gradient (from this idea) are additive at each joint. No interference.

- **MMEngine config constraint**: `bilateral_sym_loss_weight` is float, `sym_pairs` is a list-of-lists of int, `sym_mirror_axis` is int, `sym_pair_weights` is list of float, `sym_adaptive_weight` is bool, `sym_tau` is float. All are literals. No Python import statements required. Fully compliant.

- **Memory**: the asymmetry tensors `asym_pred` and `asym_gt` are `(B=4, P=10, 3)` = 120 float16 values ≈ 240 bytes. The pair index buffers are `(10, 2)` long tensors ≈ 160 bytes. Total additional memory: negligible.

- **Speed**: the symmetry loss computation involves 3 index selections and elementwise operations on `(B, 10, 3)` tensors — < 0.01 ms. Negligible overhead.
