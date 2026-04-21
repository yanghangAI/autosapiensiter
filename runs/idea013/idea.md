**Idea Name:** Kinematic Chain Bone-Vector Output Parameterization

**Approach:** Reparameterize the 22 body joint outputs as 21 bone vectors (parent→child 3D offsets along the BEDLAM2 skeleton tree) plus a zero root, reconstructing absolute joint positions by cumulative summation along the kinematic chain so that the network predicts *relative bone displacements* rather than independent per-joint coordinates — making every joint's prediction structurally dependent on its ancestors.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline head regresses each body joint's 3D coordinate independently:
```
joints = joints_out(decoded)   # (B, 22, 3) — one Linear head, 22 independent 3-vectors
```
There is no structural relationship between predictions; a change at the elbow prediction does not propagate to the wrist prediction. The network must *learn* the full kinematic tree structure purely from gradient signal, which is a hard task given 20 epochs and the fact that per-joint coordinate losses don't penalize structural inconsistency (a long-arm bias at the shoulder + a short-forearm bias at the elbow can cancel at their respective joint losses but produce a wildly non-anatomical pose).

### The kinematic-tree reparameterization

The human body is a **tree**: the pelvis is the root, and every other body joint has a unique parent whose position determines the child's origin. Under a bone-vector parameterization, the network predicts:
- `bone_vec[i] ∈ R^3` for each non-root joint `i`, representing the 3D offset from joint `parent[i]` to joint `i`.
- The root (pelvis) is zero after `SubtractRootJoint` in the data pipeline.

Absolute root-relative joint positions are then recovered by **forward kinematics** (cumulative sum along the chain):
```
joints_rr[0]  = 0                          # pelvis (root)
joints_rr[i]  = joints_rr[parent[i]] + bone_vec[i]     for i = 1..21
```

This reparameterization is a **bijection** between 21 bone vectors and 21 non-root joint positions, so expressiveness is preserved. But the gradient flow changes dramatically:

1. **Every child joint's loss flows back through all its ancestors.** If the wrist is predicted too far left, the gradient decomposes into corrections to *shoulder→elbow* and *elbow→wrist* bone vectors. The shoulder→elbow prediction then receives gradient signal from errors at the elbow, wrist, *and* hand-root (via chain). This is analogous to how skip connections in ResNet concentrate gradient, but along an anatomical tree.

2. **Bone-length prior is implicit.** Each bone_vec's magnitude is exactly the bone length. The network can learn a stable bone-length distribution (nearly subject-independent in the BEDLAM2 dataset, which uses consistent rigged skeletons) more easily than it can learn 22 coordinate distributions.

3. **Translation of any subtree is free.** Shifting an entire limb is a single bone_vec change at the root of that limb. Under the independent-coordinate parameterization, the same shift requires consistent updates to every joint in the subtree — a coordinated multi-output update that the network has to learn.

4. **Structurally impossible poses have higher loss.** Under independent regression, nothing prevents predicting the elbow on the opposite side of the body from the shoulder. Under bone-vector regression, the elbow is *always* the shoulder + a predicted offset — self-consistent by construction.

### Why this is different from every prior idea

| Prior Idea | What it changes | How this differs |
|---|---|---|
| idea001 | Multi-layer decoder | Changes *how features are processed*; output head still independent per-joint. |
| idea002 | Decoupled pelvis query | Changes *which query* produces which output; still per-joint independent regression for body. |
| idea003 | Content-adaptive query init | Changes query initialisation; output head unchanged. |
| idea004 | Depth-aware positional encoding | Changes spatial token features; output head unchanged. |
| idea005 | Uncertainty loss weighting | Rebalances loss terms; output parameterization unchanged. |
| idea006 | Skeleton self-attention bias | Changes query-query attention via a learnable bias; output head unchanged. |
| idea007 | Cross-attention spatial gating | Changes which spatial tokens each query attends to; output unchanged. |
| idea008 | Body-only decoder + hand MLP | Changes which joints are decoded; body outputs still per-joint independent. |
| idea009 | Spatial token dropout | Regularises cross-attention; output unchanged. |
| idea010 | 2D reprojection loss | Adds a loss term; output parameterization unchanged. |
| idea011 | Iterative coordinate refinement | Adds a second forward pass; *output remains per-joint coordinates*, predicted as a sum of two per-joint heads (pass-1 + residual). Both passes use the independent-coordinate parameterization — residuals are still per-joint, not bone-structured. |
| idea012 | Pairwise distance matrix loss | Adds a loss term over *all* pairs of joints; output parameterization unchanged. Supervises inter-joint *distances* but the network still produces 22 independent 3-vectors. |

This idea is **the first output-parameterization change**. All prior ideas kept the head's output as 22 independent 3D coordinates — variations were in the decoder body, the queries, the loss, or the refinement logic. None restructure *what the 22 outputs physically represent*. Bone-vector parameterization is orthogonal to every above change and can compose with idea002's decoupled pelvis, idea008's body-focused decoder, idea011's refinement passes, idea012's distance loss, etc.

### Grounding in observed results

- **idea012's rationale** was that inter-joint *distances* carry structural information missing from per-joint losses. Idea012 adds this as a *loss-level* signal on unchanged outputs. This idea takes the logical next step: rather than adding a loss that *encourages* structural consistency, **embed the structure in the prediction itself**, so structural consistency is a property of the output by construction and the network's capacity is spent on *content* rather than *topology*.
- **idea008/design003** (composite 157.83) showed that reducing to 22 body-only queries and using a structured body→hand recovery (MLP) yields strong gains. This is evidence that *structural priors* help on BEDLAM2. Bone-vector parameterization is another structural prior, applied at the output end of the body pathway.
- Body MPJPE has a floor around ~140 mm (idea002/design002) across all 9 completed ideas + 3 in-progress. Breaking this floor requires a new *prior*, not more decoder capacity. Kinematic parameterization introduces a structural prior the baseline lacks.
- Pelvis, depth, and UV pathways remain untouched — so there is no risk of the pelvis regression seen in idea001.

### Kinematic Chain Prior Art

Bone-vector (kinematic-chain) parameterization is a widely-used, effective approach in 3D human pose estimation:
- **SMPL / SMPL-X** body models define joints via a kinematic tree of rotations over rest-pose bone lengths.
- **Li et al. (2019, CVPR)** "3D Human Pose Estimation from Monocular Images with Deep Convolutional Network" and related work explicitly parameterize human pose via joint rotations or bone vectors for structural robustness.
- **Integral Human Pose Regression** (Sun et al., ECCV 2018) and follow-ups have observed that structural priors at the output head improve both absolute and relative accuracy.
- **Simple Baselines 3D** (Martinez et al., ICCV 2017) showed that even direct coordinate regression benefits from carefully parameterizing the skeleton.

For this project, we use the simplest variant: predict *bone-translation vectors* (not rotations) and recover positions by cumulative sum. This is fully differentiable with no numerical risk (no quaternion normalization, no matrix exponentials, no ill-defined rotations).

## Analysis of Baseline Weak Point

Consider the baseline's elbow prediction. The joint loss term for the elbow is:
```
L_elbow = smooth_l1(pred_elbow, gt_elbow)
```
The gradient at the elbow output is `d L_elbow / d pred_elbow`, which flows through the elbow's own linear head (one of 22 output heads). This gradient does **not** affect the shoulder or wrist predictions directly; it affects them only via backbone/decoder-level shared representations.

Under bone-vector parameterization, the forward kinematics is:
```
pred_elbow = pred_shoulder + bone_vec_shoulder_to_elbow
            = pred_chest    + bone_vec_chest_to_shoulder + bone_vec_shoulder_to_elbow
            = pred_pelvis(root, =0) + sum of bone_vecs along chain
```
So the elbow loss's gradient propagates back through **all bone vectors on the chain from the root to the elbow** (chest→shoulder, shoulder→elbow). Errors at end-effectors (wrists, ankles) influence all ancestor bones. The network can no longer minimise the elbow loss by nudging a single output head — it must adjust the bone chain coherently.

This is the structural-inductive-bias analog of 2D convolutions vs. fully-connected layers: the *representation* itself encodes the structural assumption, rather than the loss penalising violations of it.

## Proposed Variations

**Design A — Bone-vector head on body joints (minimal)**

Replace the body portion of `joints_out` with a bone-vector head. The output head produces `(B, 22, 3)` where entry `i=0` is the root (forced to zero by construction or by design) and entries `i=1..21` are bone vectors. A forward-kinematics function in `pose3d_transformer_head.py` recovers root-relative joint positions:
```python
parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
def forward_kinematics(bone_vecs):  # (B, 22, 3) → (B, 22, 3)
    joints = bone_vecs.clone()
    joints[:, 0] = 0.0
    for i in range(1, 22):
        joints[:, i] = joints[:, parents[i]] + bone_vecs[:, i]
    return joints
```
Hands (indices 22–69) remain in direct-coordinate regression (unchanged). Loss unchanged (applied on recovered joint positions). This is the minimal-change variant — tests whether the bone-vector reparameterization alone produces gains.

Key implementation detail for **zero-starting-state equivalence**: the baseline output head `joints_out: Linear(hidden_dim, 3)` is trunc-normal initialised with bias=0. Under bone-vector parameterization, the output at init is `bone_vecs ≈ small random vectors`. Cumulative sum of 21 small random vectors gives `joints_rr ≈ O(sqrt(21) * small) ≈ larger drift than baseline`. To approximate baseline initialization, we scale the output head's weight init by `1/sqrt(21) ≈ 0.218` so the cumulative-sum variance matches the baseline's per-joint variance. This avoids destabilising the first few iterations.

**Design B — Bone-vector head with auxiliary bone-length loss (explicit bone prior)**

Same as Design A, but add a small auxiliary loss on the *magnitude* of each predicted bone vector vs. the GT bone vector magnitude:
```
L_bone_len = mean over i=1..21 of smooth_l1(||bone_vec_pred_i||, ||bone_vec_gt_i||)
```
with weight `λ_bone = 0.3`. This adds a *direction-independent* prior on bone lengths. Rationale: bone lengths in BEDLAM2 are tightly concentrated (synthetic skeletons, consistent rigging), so an explicit magnitude prior gives a low-variance training signal that the full-3D loss has to extract indirectly from data. The direction of each bone is still free to depend on pose.

**Design C — Per-limb bone-vector heads (decoupled output projections)**

Instead of a single `Linear(hidden_dim, 3)` applied to all 22 body queries, give each kinematic subtree its own output head: one for spine (pelvis→spine→chest→neck→head, 4 bones), one for left arm, one for right arm, one for left leg, one for right leg (4 bones each). Five separate `Linear(hidden_dim, 3)` heads total. This allows each limb to specialise its output mapping while sharing the decoder trunk. The parent–child summation logic is identical to Design A.

Parameter cost: 5 separate heads vs 1 shared head → negligible (5× 256×3 = 3.8k params vs 1× 256×3 = 0.77k params; ~3 KB extra). Still below 1% of head size.

## Implementation Scope

Changes confined to **two** allowed files.

### `pose3d_transformer_head.py`
1. `__init__`: accept `kinematic_parametrization: bool = False`, `bone_parents: list = None`, `bone_length_loss_weight: float = 0.0`, `per_limb_heads: bool = False` as kwargs.
2. `__init__`: when `kinematic_parametrization=True`, keep `joints_out: Linear(hidden_dim, 3)` but scale-init by `1/sqrt(num_body_bones)`. When `per_limb_heads=True`, allocate 5 separate `Linear(hidden_dim, 3)` heads and a fixed `limb_index` mapping (22-long list of which limb each body joint belongs to) passed from config.
3. `forward()`: after decoder, produce `bone_vecs` for body joints. Apply `forward_kinematics(bone_vecs)` → root-relative joint positions. Replace body slots in `joints` tensor with recovered positions. Hand joints (indices 22–69) remain unchanged (direct regression).
4. `loss()`: unchanged except if `bone_length_loss_weight > 0`, compute GT bone vectors (`gt_joints[:, child] - gt_joints[:, parent]`), compute predicted bone-vector magnitudes, add `bone_length_loss_weight * smooth_l1(||pred_bone||, ||gt_bone||).mean()`.
5. `predict()`: unchanged — reads `pred['joints']` which is already the recovered joint tensor.

### `config.py`
- `kinematic_parametrization=True` (bool literal)
- `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]` (list of int literals — identical to idea012 for consistency)
- `bone_length_loss_weight=0.0 | 0.3` (float literal)
- `per_limb_heads=False | True` (bool literal)
- For Design C: `limb_index=[0, 0, 1, 2, 0, 0, 1, 2, 0, 3, 4, 3, 4, 0, 1, 2, 1, 2, 1, 2, 1, 2]` (22 ints — 0=spine, 1=left_arm, 2=right_arm, 3=left_leg, 4=right_leg; Designer will confirm exact mapping)

No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or `train.py` wrapper.

## Expected Outcome

- **Primary gain — body MPJPE**: the kinematic-chain structural prior is expected to produce 5–15 mm improvement by eliminating structurally impossible predictions and concentrating gradient flow along anatomical chains. Target: `mpjpe_body_val < 140` (breaking the 140.96 mm floor from idea002/design002).
- **mpjpe_rel_val and mpjpe_abs**: expected positive. Root-relative and absolute pose benefit from self-consistent body structure.
- **Pelvis MPJPE**: expected neutral (pelvis pathway unchanged — depth/UV still from token 0's separate heads).
- **Hand MPJPE**: expected neutral (hand joints unchanged, still direct regression).
- **Composite target**: `composite_val < 153`, improving on the best prior (154.85 — idea002/design003).

## Risk and Mitigation

- **Initialization drift from cumulative sum**: with vanilla trunc-normal init, summing 21 random bone vectors produces larger root-relative drift than direct regression. Mitigation: scale weight init of the body bone-vec head by `1/sqrt(21) ≈ 0.218`. Analysed in Design A notes. Designer can additionally verify by a small forward pass that `pred_joints` standard deviation at init matches baseline's.
- **Gradient scale imbalance between body (kinematic) and hand (direct) outputs**: body joints share a head that is now a kinematic sum; gradient magnitude per output-head weight is different from the hand head. Mitigation: the body and hand heads already use the same loss scale; the initialization scaling above aligns initial gradient magnitudes. If Designer observes early-training divergence between body and hand losses, a small warmup on the body loss or a separate optimiser lr_mult on the bone-vec head can be applied via MMEngine's `paramwise_cfg` (but this is a Designer tuning choice, not a core idea requirement).
- **Loss / metric semantics preserved**: the loss is still `smooth_l1(pred_joints[:, 0:22], gt_joints[:, 0:22])`. The pred_joints is now the output of forward_kinematics, which is a *differentiable, bijective* transform of bone_vecs. The metric `BedlamMPJPEMetric` sees the same tensor shape and convention — no change.
- **Parent list correctness**: bone_parents must correspond to the actual BEDLAM2 joint indexing. This is shared with idea012 and has been validated there (same 22-long parents list). Designer should cross-reference with the Reviewer's memory entry for idea012 which has already verified this list.
- **Fixed-point stability of forward_kinematics**: the summation is a finite sequence of 21 adds (O(21) compute). It is not an iterative fixed-point — no numerical convergence concerns. Memory: negligible; one (B, 22, 3) tensor in forward.
- **Hands cause indexing coupling**: body and hand joints share the same `Linear(hidden_dim, 3)` in the baseline. If we modify the output semantics for body, we must not accidentally modify hand outputs. Mitigation: the head produces `(B, 22+48, 3)` — body slots go through `forward_kinematics`, hand slots are the direct linear output. Clean slicing: `body_bone_vecs = joints_raw[:, 0:22]; hand_coords = joints_raw[:, 22:70]; body_coords = forward_kinematics(body_bone_vecs); joints_final = cat([body_coords, hand_coords], dim=1)`.
- **Interaction with prior ideas**: orthogonal to all prior ideas (they change decoder internals, queries, losses, or refinement passes; this changes output parameterization). Can compose with idea002 (decoupled pelvis — pelvis query is separate, body query still produces bone vecs), idea008 (body-focused decoder — bone-vec head replaces body output head), idea011 (iterative refinement — residual pass predicts bone-vec residuals), idea012 (distance matrix loss — supervises distances on the kinematically-recovered joints).
- **MMEngine config constraint**: all new kwargs are bool/float/int-list literals. No imports required.
- **Eval/inference compatibility**: `predict()` is unchanged; `bedlam_metric.py` and `TrainMPJPEAveragingHook` see identical tensor shapes. Invariants preserved.
- **Memory / speed**: one additional (B, 22, 3) tensor allocation; 21 adds per forward. Negligible on 1080 Ti (<0.1 ms per batch).
