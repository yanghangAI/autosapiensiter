# Design 003 — Per-Limb Bone-Vector Heads (decoupled output projections)

**Design Description:** Same kinematic-chain output reparameterization as Design 001 (22 body joints recovered via cumulative sum along the SMPL-X parent chain), but the body portion of the output projection is replaced by FIVE decoupled per-limb `Linear(hidden_dim, 3)` heads — one each for spine, left_arm, right_arm, left_leg, right_leg. Each body token (index 0..21) is routed to the head corresponding to its kinematic subtree via a fixed 22-long `limb_index` list. Hand tokens (indices 22..69) continue to pass through the original shared `joints_out` head. Each per-limb head is trunc-normal initialised with `std=0.02 / sqrt(21)` to preserve cumulative-sum init variance.

**Starting Point:** `baseline/`

---

## Overview

Design 001 uses a single shared `Linear(hidden_dim, 3)` head whose output is reinterpreted as bone vectors for the 22 body tokens and as direct coordinates for the 48 hand tokens. Design 003 introduces a minor architectural refinement: five decoupled output projections, one per anatomical limb group, so that each limb can specialise its mapping from decoded hidden feature → bone vector. The rest of the decoder (backbone, self-attention, cross-attention, FFN, layer norms, queries, pelvis pathway) is shared.

The five limb groups and their membership (by body joint index):

| Limb group | Index value | Body joints (idx ∈ 0..21) | # joints | # bones under kinematic recovery |
|------------|-------------|---------------------------|----------|-----------------------------------|
| spine      | 0           | 0, 3, 6, 9, 12, 15        | 6        | 5 (0→3, 3→6, 6→9, 9→12, 12→15)    |
| left_leg   | 1           | 1, 4, 7, 10               | 4        | 4 (0→1, 1→4, 4→7, 7→10)           |
| right_leg  | 2           | 2, 5, 8, 11               | 4        | 4 (0→2, 2→5, 5→8, 8→11)           |
| left_arm   | 3           | 13, 16, 18, 20            | 4        | 4 (9→13, 13→16, 16→18, 18→20)     |
| right_arm  | 4           | 14, 17, 19, 21            | 4        | 4 (9→14, 14→17, 17→19, 19→21)     |

Total: 22 joints, 21 bones (checks). The pelvis (index 0) is assigned to the `spine` group by convention; since its bone-vec is overwritten to zero by `_forward_kinematics`, the choice of group for index 0 doesn't affect the recovered coordinate.

Limb index list (22 ints, one per body joint):

```python
LIMB_INDEX = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4]
```

Check: count per limb = (0: 6), (1: 4), (2: 4), (3: 4), (4: 4) → 22 total. ✓

---

## BEDLAM2 / SMPL-X 22-Joint Body Skeleton (hardcoded)

Parent list is identical to Designs 001 and 002:

```python
BONE_PARENTS_SMPLX_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
```

See design001.md for the full joint-name table. No changes to the kinematic tree; only the *how* of the per-token output projection differs.

---

## Files to Change

1. `pose3d_transformer_head.py` — add five per-limb `Linear(hidden_dim, 3)` heads (or equivalently, a single `Linear(hidden_dim, 3 * num_limbs)` reshaped and selected), accept and register `limb_index` as a buffer, and in `forward()` route each body token through the head of its assigned limb before concatenating with the hand path. Keep the original `self.joints_out` head for the hand tokens.
2. `config.py` — add the five new head kwargs with `per_limb_heads=True` and the `limb_index` list.
3. `pelvis_utils.py` — **no change**.

No new imports are introduced beyond those in Designs 001/002 (`torch`, `torch.nn`, `math` — all already present).

---

## Algorithm Changes

### `pose3d_transformer_head.py`

#### 1. `Pose3dTransformerHead.__init__` — new parameters (shared with Designs 001/002)

Same five kwargs as Designs 001 and 002 appear in the signature (already specified in design001.md §Algorithm Changes §1):

```python
kinematic_parametrization: bool = False,
bone_parents: list = None,
bone_length_loss_weight: float = 0.0,
per_limb_heads: bool = False,
limb_index: list = None,
```

All storage / assertion logic from Design 001 applies (topological ordering of `bone_parents`, registering as a non-persistent long-tensor buffer, caching `self._bone_parents_list`).

**New for Design 003 — per-limb heads construction.** After the `bone_parents` handling block, add a new block gated on `per_limb_heads`:

```python
if per_limb_heads:
    assert kinematic_parametrization, (
        "per_limb_heads=True requires kinematic_parametrization=True.")
    assert limb_index is not None and len(limb_index) == 22, (
        f"per_limb_heads=True requires limb_index (len-22 list of int), "
        f"got {limb_index!r}")
    num_limbs = int(max(limb_index)) + 1  # expect 5 for the default mapping
    assert num_limbs == 5, (
        f"Expected 5 limb groups (spine/left_leg/right_leg/left_arm/right_arm), "
        f"got {num_limbs} distinct limb indices.")
    for val in limb_index:
        assert 0 <= val < num_limbs, (
            f"limb_index values must be in [0, {num_limbs - 1}], got {val}")

    # Five decoupled body-bone-vec heads. Named `body_limb_heads` to make
    # their purpose unambiguous (they do NOT replace `joints_out`; the
    # original `joints_out` is still used for hand tokens).
    self.body_limb_heads = nn.ModuleList([
        nn.Linear(self.hidden_dim, 3) for _ in range(num_limbs)
    ])

    # Register limb_index as a non-persistent long-tensor buffer for
    # consistent device placement.
    self.register_buffer(
        'limb_index',
        torch.tensor(limb_index, dtype=torch.long),
        persistent=False)

    # Pre-compute per-limb body-token index lists (Python lists of int,
    # stored on host side). For efficient advanced-indexing in forward().
    self._limb_token_lists = [
        [i for i in range(22) if limb_index[i] == limb_id]
        for limb_id in range(num_limbs)
    ]
else:
    self.body_limb_heads = None
    self.limb_index = None
    self._limb_token_lists = None
```

Constraints:
- `per_limb_heads=True` requires `kinematic_parametrization=True`. An attempt to enable per-limb heads without the kinematic parametrization MUST raise AssertionError (behaviourally meaningless: if bone-vec recovery is off, the five heads produce direct coordinates for the body — no structural advantage over the baseline single-head).
- `num_limbs` is derived from `max(limb_index) + 1`. For the hardcoded default list, this equals 5. The assert validates it equals 5 to catch accidental mis-entries.
- The five per-limb heads are stored as an `nn.ModuleList` named `self.body_limb_heads`. Do NOT name the attribute `self.joints_out` (that name is already taken by the hand/baseline head).
- `self._limb_token_lists` is a Python nested list of int — NOT a tensor — so that advanced indexing in `forward()` (`joints[:, token_list, :]`) uses Python-side bucketing without GPU syncs.
- Parameter count increase: 5 × (256 × 3 + 3) = 5 × 771 = 3855 floats = ~15 KB. Net change vs. baseline: (5 × 771) - (1 × 771) = 4 × 771 = 3084 extra trainable parameters. This is **~0.001%** of the total model parameter count. Negligible.

#### 2. `_init_head_weights` — scale-init for per-limb heads

Under `per_limb_heads=True`, each of the five heads is trunc-normal initialised with `std=0.02`, then scaled by `1/sqrt(21)` in-place (identical rule to Design 001 for `self.joints_out.weight`). Also scale the original `self.joints_out.weight` by `1/sqrt(21)` so the hand direct-regression init is unchanged relative to Design 001 (same minor-scale-down for shared-code consistency).

Full updated `_init_head_weights`:

```python
def _init_head_weights(self) -> None:
    # Query embeddings
    nn.init.trunc_normal_(self.joint_queries.weight, std=0.02)
    # Original (shared) output projections
    for m in [self.joints_out, self.depth_out, self.uv_out]:
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # Per-limb heads (Design 003 only)
    if self.per_limb_heads:
        for m in self.body_limb_heads:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    # Scale-init body bone-vec head(s) to keep recovered-joint variance
    # comparable to baseline's direct-regression variance after the
    # cumulative-sum forward-kinematics transform.
    if self.kinematic_parametrization:
        num_body_bones = 21  # 22 body joints - 1 root
        scale = 1.0 / math.sqrt(num_body_bones)
        with torch.no_grad():
            self.joints_out.weight.mul_(scale)
            if self.per_limb_heads:
                for m in self.body_limb_heads:
                    m.weight.mul_(scale)
```

Constraints:
- All weight-in-place multiplications MUST be inside a single `with torch.no_grad():` context.
- The original `self.joints_out.weight` scaling is retained (as in Design 001). This affects the hand direct-regression init but not the body (the body's `joints_out` output is now ignored under `per_limb_heads=True`; see §3 below).
- Do NOT scale biases (they are already zero).

#### 3. `forward()` — per-limb routing for body tokens

Under `per_limb_heads=True`, the body tokens are routed to their respective limb head instead of the shared `joints_out`. The simplest implementation applies each of the five heads to its subset of body tokens:

```python
# Decoder
decoded = self.decoder_layer(queries, spatial)  # (B, num_joints, hidden_dim)

if self.per_limb_heads:
    # Per-limb body-bone-vec heads.
    B = decoded.size(0)
    # Pre-allocate body output tensor.
    body_bone_vecs = decoded.new_zeros(B, 22, 3)
    for limb_id, token_list in enumerate(self._limb_token_lists):
        # token_list is a Python list[int] (e.g., [0, 3, 6, 9, 12, 15] for spine).
        if len(token_list) == 0:
            continue
        idx = torch.tensor(token_list, device=decoded.device, dtype=torch.long)
        # Gather the decoded features for these body tokens.
        sel = decoded.index_select(1, idx)          # (B, k, hidden_dim)
        bone_vecs_limb = self.body_limb_heads[limb_id](sel)  # (B, k, 3)
        body_bone_vecs.index_copy_(1, idx, bone_vecs_limb)

    # Hand tokens go through the original shared head.
    hand_decoded = decoded[:, 22:self.num_joints, :]           # (B, 48, hidden_dim)
    hand_coords = self.joints_out(hand_decoded)                # (B, 48, 3)

    # Forward kinematics on the body portion.
    body_rr = self._forward_kinematics(body_bone_vecs)          # (B, 22, 3)
    joints = torch.cat([body_rr, hand_coords], dim=1)           # (B, num_joints, 3)
else:
    # Designs 001 / 002 path: single shared head.
    joints = self.joints_out(decoded)  # (B, num_joints, 3)
    if self.kinematic_parametrization:
        body_bone_vecs = joints[:, 0:22, :]
        hand_coords = joints[:, 22:self.num_joints, :]
        body_rr = self._forward_kinematics(body_bone_vecs)
        joints = torch.cat([body_rr, hand_coords], dim=1)

pelvis_token = decoded[:, 0, :]  # unchanged
# ... depth_out and uv_out unchanged ...
```

Constraints:
- `body_bone_vecs = decoded.new_zeros(B, 22, 3)` creates a fresh tensor with matching dtype/device; this is the canvas into which each limb head writes its slot.
- `decoded.index_select(1, idx)` gathers the `k` body tokens for a limb. Using `index_select` (not `decoded[:, idx, :]`) is numerically equivalent but uses a dedicated kernel with well-defined autograd behaviour.
- `body_bone_vecs.index_copy_(1, idx, bone_vecs_limb)` scatters the per-limb outputs into the canonical 22-slot tensor. `index_copy_` is autograd-safe (the scattered values are new tensors with a proper backward edge).
- The `idx = torch.tensor(token_list, device=decoded.device, dtype=torch.long)` call allocates a tiny tensor (~5–6 ints = 48 bytes) per limb per forward. Total: 5 small tensors allocated per forward, negligible (`<1 μs` overhead). Alternative: pre-register each of the five `idx` tensors as a buffer in `__init__` (`self.register_buffer(f'_limb_idx_{i}', ...)`). **Use the alternative**: register 5 non-persistent long buffers `self._limb_idx_0, ..., self._limb_idx_4` in `__init__` (immediately after `self._limb_token_lists` is built). This eliminates per-forward allocation:
  ```python
  for i, token_list in enumerate(self._limb_token_lists):
      self.register_buffer(
          f'_limb_idx_{i}',
          torch.tensor(token_list, dtype=torch.long),
          persistent=False)
  ```
  Then in `forward()`: `idx = getattr(self, f'_limb_idx_{limb_id}')` (already on the correct device after `model.to(device)`).
- `hand_decoded` is the 48 hand tokens, passed through the unchanged `self.joints_out` head. No kinematic recovery for hands.
- The final `joints` tensor is a concatenation along `dim=1` of the recovered body tensor (22 slots) and the hand-coord tensor (48 slots). Shape: `(B, 70, 3)` — identical to baseline.
- The `else` branch (non-`per_limb_heads`) path is identical to Designs 001/002.
- Pelvis depth/UV pathway is UNCHANGED — `pelvis_token = decoded[:, 0, :]` reads the decoder's token-0 embedding (pre-head), independent of the per-limb routing.

#### 4. `_forward_kinematics` — unchanged

Identical to Design 001. The method takes `(B, 22, 3)` bone-vec tensor and returns `(B, 22, 3)` recovered root-relative coordinates. Design 003 simply feeds it `body_bone_vecs` assembled from five per-limb heads instead of a single shared head.

#### 5. `loss()` — unchanged in shape; no bone-length loss in Design 003

Design 003 sets `bone_length_loss_weight=0.0`, so the bone-length auxiliary block (described in design001.md §Algorithm Changes §4 and design002.md §Algorithm Changes) is SKIPPED by the `> 0.0` guard. The main `loss/joints/train`, `loss/depth/train`, `loss/uv/train` terms are emitted exactly as in Designs 001 and 002.

(Design 003 deliberately isolates the effect of per-limb heads by not co-enabling the bone-length auxiliary. A follow-up design could combine per-limb heads + bone-length loss if Design 003 succeeds.)

#### 6. `predict()` — unchanged

No change. `forward()` returns the recovered coordinate tensor; `predict()` wraps it in `InstanceData`.

---

## Config Changes

### `config.py`

In the `head=dict(...)` block inside `model=dict(...)`:

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
    kinematic_parametrization=True,
    bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    bone_length_loss_weight=0.0,
    per_limb_heads=True,
    limb_index=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4],
),
```

`limb_index` is a plain Python list of 22 `int` literals (values in `{0, 1, 2, 3, 4}`) — fully MMEngine-config compliant. All other config values are identical to the baseline.

---

## Exact Config Values (unchanged from baseline except five head kwargs)

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.03 |
| backbone lr_mult | 0.1 |
| clip_grad max_norm | 1.0 |
| accumulative_counts | 8 (effective batch 32) |
| LR schedule | LinearLR (epoch 0-3, start_factor=0.333) + CosineAnnealingLR (epoch 3-20, eta_min=0), both convert_to_iter_based=True |
| seed | 2026 |
| batch_size | 4 |
| num_workers | 4 |
| persistent_workers | False |
| hidden_dim | 256 |
| num_heads | 8 |
| dropout | 0.1 |
| loss_joints loss_weight | 1.0 |
| loss_depth loss_weight | 1.0 (× loss_weight_depth=1.0) |
| loss_uv loss_weight | 1.0 (× loss_weight_uv=1.0) |
| **kinematic_parametrization** | **True (new)** |
| **bone_parents** | **[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] (new)** |
| **bone_length_loss_weight** | **0.0 (new; disabled in Design 003)** |
| **per_limb_heads** | **True (new)** |
| **limb_index** | **[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4] (new)** |
| num_epochs | 20 |
| warmup_epochs | 3 |

---

## Constraints and Invariants the Builder Must Preserve

All constraints 1–24 from Design 001 apply verbatim (kinematic parametrization correctness, parent list validation, forward-kinematics clone-and-overwrite, pelvis depth/UV pathway untouched, config-literal-only rule, etc.). Additional constraints specific to Design 003:

32. `per_limb_heads=True` MUST require `kinematic_parametrization=True` (assertion in `__init__`). Incompatible combinations MUST fail fast.
33. `limb_index` MUST be a length-22 list of ints in `[0, num_limbs)` where `num_limbs = max(limb_index) + 1 = 5` for the default list.
34. `self.body_limb_heads` MUST be an `nn.ModuleList` of five `nn.Linear(hidden_dim, 3)` instances. Do NOT use a single `nn.Linear(hidden_dim, 5 * 3)` with slicing; the separate heads are intentional for parameter decoupling across limbs.
35. The original `self.joints_out` head MUST remain in place and MUST continue to serve the 48 hand tokens (`decoded[:, 22:num_joints]`). Do NOT delete or replace `self.joints_out` — it is still used for the hand direct-regression path.
36. Each per-limb head's weight MUST be initialised with `nn.init.trunc_normal_(std=0.02)` AND then in-place multiplied by `1/sqrt(21)` inside `_init_head_weights` (same rule as the single shared head in Design 001).
37. The per-limb routing inside `forward()` MUST use `index_select` for the gather and `index_copy_` for the scatter — these are autograd-safe and O(1) in Python-loop overhead (five iterations total).
38. The pre-registered limb index buffers (`self._limb_idx_0`, ..., `self._limb_idx_4`) MUST be non-persistent (`persistent=False`) — they are derived from the config list.
39. Before concatenating with hand outputs, the body portion MUST pass through `_forward_kinematics` to recover root-relative coordinates from the bone vectors. Skipping this step would emit raw bone vectors into `pred['joints'][:, 0:22]`, which the loss and metric would interpret as coordinates (producing a training failure: the loss would supervise bone-vec-like quantities against coordinate GTs).
40. `bone_length_loss_weight=0.0` in Design 003 — the auxiliary bone-length block is NOT active.
41. Extra learnable parameters: `4 × (256 × 3 + 3) = 3084` floats (the four *additional* limb heads vs. the baseline's single shared head) — still <0.001% of the total model parameter count. Config `hidden_dim=256` is unchanged.
42. `pred['joints']` shape is `(B, 70, 3)` — identical to baseline.

---

## Expected Behaviour After Change

- `forward()` produces the same tensor shape `(B, 70, 3)` as the baseline. Extra compute per forward: five small `(B, k, 256)→(B, k, 3)` matmuls (instead of one `(B, 70, 256)→(B, 70, 3)` matmul). Total FLOPs are identical (each body token's output still costs 1 matmul of the same dimension); the overhead is five Python-side iterations and five `index_select`/`index_copy_` kernel calls = <0.5 ms on 1080 Ti.
- Training emits THREE loss scalars per step: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`. No new loss keys (bone-length auxiliary is off).
- At init (epoch 0 step 0):
  - Each per-limb head has trunc-normal-initialised weights scaled by `1/sqrt(21)`. The recovered-joint variance at init matches Design 001's (each limb's bone-vec outputs have the same init scale as Design 001's shared head).
  - `loss/joints/train` is finite positive, same order of magnitude as Design 001's step-0 value.
- During training, the five limb heads specialise their mappings to the specific anatomical patterns of each limb (e.g., leg-specific bone-vec distributions differ from arm-specific).
- Validation metrics are computed by the unchanged `BedlamMPJPEMetric`.
- Extra parameter count: **3084 new trainable parameters** (four additional `Linear(256, 3)` heads; see §Constraint 41). Effectively zero vs. the ~300 M total backbone + head params.
- Expected result vs. Design 001: small positive delta on `mpjpe_body_val`. The decoupled heads give the model 5× more capacity at the output, at a trivial param cost. If limbs have meaningfully different output-distribution statistics, this should help. If they don't, Design 003 performs comparably to Design 001 — an acceptable null outcome.
- At inference the tensor shapes and dtypes are identical to baseline.

---

## Rationale Summary

- **Why five limbs?** The natural anatomical decomposition of the SMPL-X body skeleton is spine + 2 arms + 2 legs = 5 chains. More finely split (e.g., separate head-neck from spine, or hand-side from wrist) would create very small subgroups with few training examples per head per batch; less finely (e.g., just "body-top" vs "body-bottom") would under-exploit the specialisation benefit. Five is the canonical middle ground.
- **Why not one head per bone (21 heads)?** 21 separate heads would add 20 × 771 = 15.4 K params, still small in absolute terms, but each head would see only 21 tokens per batch of 32 (1 bone × 32 batch = 32 tokens), which is too few for specialisation to kick in before overfitting. Five heads see ~4–6 bones × 32 batch = ~160 tokens per batch — a healthy sample size.
- **Why is the pelvis assigned to spine?** The pelvis is the kinematic root and its bone-vec is overwritten to zero in `_forward_kinematics`. The choice of limb group for index 0 is therefore a formality. Placing it with spine matches the SMPL-X convention (the pelvis is logically on the spine chain).
- **Why not share a head across left/right symmetric limbs?** Left and right arms (and similarly legs) in BEDLAM2 have symmetric anatomy but can appear in very different poses within a frame (e.g., left arm reaching up, right arm at the side). A shared head would have to encode the pose asymmetry in its input (the hidden feature) alone. Separate heads give the network one more dimension of flexibility at minimal cost.
- **Why keep the pelvis-depth and pelvis-UV heads unchanged?** The pelvis depth/UV predictions come from the *token-0 embedding* (not from its bone-vec). The decoupled bone-vec heads affect only the *joint-coordinate* pathway. Splitting depth/UV off per-limb would be meaningless (they're single scalars, not per-joint).

---

## Risk and Mitigation Specific to Design 003

- **Per-limb head init imbalance**: each limb has a different number of bones (spine: 5, legs: 4 each, arms: 4 each), so the cumulative-sum variance for the wrist (4 bones from pelvis+spine+arm-chain → actually 5+0? let me recount: wrist 21 ← elbow 19 ← shoulder 17 ← collar 14 ← spine3 9 ← spine2 6 ← spine1 3 ← pelvis 0, so right_wrist is 7 bones deep, crossing **spine** and **right_arm** limb groups). A wrist's recovered variance is `Var(spine bones 1–3) + Var(right_arm bones 1–4)`. Mitigation: the uniform `1/sqrt(21)` scale is conservative; the worst-case depth is bounded by 7 (not 21), so the init is slightly overdamped for deep joints. This is acceptable — the LR warmup absorbs the minor init scale mismatch within the first 3 epochs.
- **Parameter count**: 3084 extra floats is <0.001% of total model params. No meaningful effect on training speed or memory.
- **Kernel launch overhead**: five `index_select` + `Linear` + `index_copy_` loops replace a single large matmul. On 1080 Ti this adds ~0.3–0.5 ms per forward (five small kernel launches). Negligible vs. the ~250 ms backbone forward.
- **Correctness of `index_copy_`**: it is an out-of-place *gradient* operation (the input `body_bone_vecs` is fresh-allocated via `new_zeros` in the forward; its contents are overwritten by `index_copy_`). Autograd treats this correctly — the gradient at the final `body_bone_vecs` flows back to each per-limb head's output via the scatter-gather.
- **Zero rows in the canvas**: `body_bone_vecs` is initialised with `new_zeros` and then **all 22 rows are written** by the five `index_copy_` calls (the five `_limb_token_lists` partition the 22 body-token indices exactly). No row remains zero. Verify by asserting `set().union(*_limb_token_lists) == set(range(22))` in `__init__` (optional sanity check).
- **Device placement of limb-index buffers**: `register_buffer(..., persistent=False)` ensures they move with `model.to(device)`. All index gather/scatter ops happen on-device.
- **Eval/inference compatibility**: `predict()` calls `forward()`, which routes body tokens through the per-limb heads and applies forward kinematics. The final `(B, 70, 3)` tensor shape is identical to baseline. `bedlam_metric.py` sees the same interface.
- **MMEngine config constraint**: all new kwargs are plain literals. `limb_index` is a 22-int list.
- **Interaction with Designs 001/002**: the head file supports all three via the `kinematic_parametrization` / `per_limb_heads` / `bone_length_loss_weight` flags. Default values (all `False` / `0.0` / `None`) reproduce the baseline exactly, bit-for-bit.
- **Memory**: five `Linear(256, 3)` weight tensors + biases total ~15 KB of extra weights; one `(B, 22, 3)` canvas tensor = ~264 B × batch. All negligible.
