# Design 002 — Shared Attention Bias with Skeleton-Graph Initialisation (Design B)

**Design Description:** Add a `(70, 70)` learnable additive attention bias, initialized from the SMPL-X/BEDLAM2 kinematic graph (`+0.5` for adjacent joints, `0.0` otherwise, `-0.5` on pelvis token 0 diagonal), passed as `attn_mask` to query self-attention.

**Starting Point:** `baseline/`

---

## Overview

Same mechanism as Design 001 but with a skeleton-graph-informed warm-start instead of zero initialization.

## Algorithm

The algorithm modification extends the skeleton-graph-initialized attention bias into the self-attention of `_DecoderLayer`:

1. Build a `(70, 70)` float32 tensor `B` using `_build_skeleton_attn_bias`: set `B[i, j] = B[j, i] = +0.5` for every kinematic edge `(i, j)` in the SMPL-X/BEDLAM2 skeleton graph; set `B[0, 0] = -0.5`; all other entries remain `0.0`.
2. Register `self.attn_bias = nn.Parameter(B.clone())` in `_DecoderLayer.__init__`.
3. At every forward pass, pass `attn_mask=self.attn_bias` to `nn.MultiheadAttention.forward`. The modified attention is: `Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + attn_bias) V`.
4. The warm-start prior provides immediate kinematic structure; gradients continue to refine the bias from this non-zero starting point.
5. No other algorithmic changes — loss function, cross-attention, FFN, backbone, and data pipeline are unchanged. Adjacent joint pairs in the SMPL-X kinematic tree are initialized to `+0.5` (bidirectionally), non-adjacent pairs to `0.0`, and the pelvis token (index 0) diagonal entry to `-0.5`. This tests whether a kinematically-grounded prior accelerates convergence within the 20-epoch budget.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

#### 1a. `_DecoderLayer.__init__` — add `attn_bias` parameter with optional init tensor

**New signature:**
```python
def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
             num_joints: int = 70, attn_bias_init: 'Optional[torch.Tensor]' = None):
```

After `self.dropout2 = nn.Dropout(dropout)` and before the method ends, add:

```python
if attn_bias_init is not None:
    self.attn_bias = nn.Parameter(attn_bias_init.float().clone())
else:
    self.attn_bias = nn.Parameter(torch.zeros(num_joints, num_joints))
```

#### 1b. `_DecoderLayer.forward` — pass `attn_mask`

Replace the self-attention call:
```python
# Current:
q2 = self.self_attn(q, q, q)[0]
# Replace with:
q2 = self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]
```

#### 1c. Module-level helper — `_build_skeleton_attn_bias`

Add the following function at module level (after the imports, before `_DecoderLayer`). This function is pure Python with no imports beyond `torch` (already imported):

```python
def _build_skeleton_attn_bias(num_joints: int = 70,
                               adjacent_val: float = 0.5,
                               pelvis_diag_val: float = -0.5) -> torch.Tensor:
    """Build skeleton-graph attention bias for BEDLAM2 70-joint set.

    Body joints 0-21 follow the SMPL-X kinematic tree.
    Hand joints 22-69 follow per-finger chains (left hand 22-44, right hand 45-67,
    with jaw=68, head_top=69 attached to head joint 15).

    Adjacent pairs receive +adjacent_val (bidirectional).
    Pelvis token (0) diagonal receives pelvis_diag_val.
    All other entries are 0.0.
    """
    bias = torch.zeros(num_joints, num_joints)

    # ── SMPL-X body kinematic edges (0-21) ──────────────────────────────────
    # Standard SMPL-X 22-joint tree: pelvis(0) is root
    body_edges = [
        (0, 1),   # pelvis → left_hip
        (0, 2),   # pelvis → right_hip
        (0, 3),   # pelvis → spine1
        (1, 4),   # left_hip → left_knee
        (2, 5),   # right_hip → right_knee
        (3, 6),   # spine1 → spine2
        (4, 7),   # left_knee → left_ankle
        (5, 8),   # right_knee → right_ankle
        (6, 9),   # spine2 → spine3
        (7, 10),  # left_ankle → left_foot
        (8, 11),  # right_ankle → right_foot
        (9, 12),  # spine3 → neck
        (9, 13),  # spine3 → left_collar
        (9, 14),  # spine3 → right_collar
        (12, 15), # neck → head
        (13, 16), # left_collar → left_shoulder
        (14, 17), # right_collar → right_shoulder
        (16, 18), # left_shoulder → left_elbow
        (17, 19), # right_shoulder → right_elbow
        (18, 20), # left_elbow → left_wrist
        (19, 21), # right_elbow → right_wrist
    ]

    # ── Left hand kinematic edges (joints 22-44, wrist=20 is root) ──────────
    # SMPL-X left hand: 5 fingers x 4 joints + 3 metacarpals = 15+5=20 joints
    # Layout: index(22-25), middle(26-29), pinky(30-33), ring(34-37),
    #         thumb(38-41), plus metacarpal bases (42-44) attached to wrist(20)
    # Simplified as linear chains per finger, metacarpals connect to wrist
    left_hand_edges = [
        (20, 22), (22, 23), (23, 24), (24, 25),  # index finger chain
        (20, 26), (26, 27), (27, 28), (28, 29),  # middle finger chain
        (20, 30), (30, 31), (31, 32), (32, 33),  # pinky finger chain
        (20, 34), (34, 35), (35, 36), (36, 37),  # ring finger chain
        (20, 38), (38, 39), (39, 40), (40, 41),  # thumb chain
        (20, 42), (20, 43), (20, 44),             # metacarpal bases → wrist
    ]

    # ── Right hand kinematic edges (joints 45-67, wrist=21 is root) ─────────
    # Mirror of left hand, offset by +23 for joints 22→45
    right_hand_edges = [
        (21, 45), (45, 46), (46, 47), (47, 48),  # index finger chain
        (21, 49), (49, 50), (50, 51), (51, 52),  # middle finger chain
        (21, 53), (53, 54), (54, 55), (55, 56),  # pinky finger chain
        (21, 57), (57, 58), (58, 59), (59, 60),  # ring finger chain
        (21, 61), (61, 62), (62, 63), (63, 64),  # thumb chain
        (21, 65), (21, 66), (21, 67),             # metacarpal bases → wrist
    ]

    # ── Jaw and head_top (68-69, attached to head=15) ───────────────────────
    face_edges = [
        (15, 68),  # head → jaw
        (15, 69),  # head → head_top
    ]

    all_edges = body_edges + left_hand_edges + right_hand_edges + face_edges

    for i, j in all_edges:
        bias[i, j] = adjacent_val
        bias[j, i] = adjacent_val  # bidirectional

    # Soft self-suppression on pelvis token to reduce cross-contamination
    bias[0, 0] = pelvis_diag_val

    return bias
```

#### 1d. `Pose3dTransformerHead.__init__` — pass init tensor to `_DecoderLayer`

Add `attn_bias_type` parameter to `Pose3dTransformerHead.__init__`:

**New parameter** (add after `loss_weight_uv: float = 1.0`):
```python
attn_bias_type: str = 'none',
```

**In the body of `__init__`, replace:**
```python
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```

**With:**
```python
if attn_bias_type == 'skeleton_init':
    _bias_init = _build_skeleton_attn_bias(num_joints, adjacent_val=0.5,
                                            pelvis_diag_val=-0.5)
elif attn_bias_type == 'zero_init':
    _bias_init = torch.zeros(num_joints, num_joints)
else:
    _bias_init = None
self.decoder_layer = _DecoderLayer(
    hidden_dim, num_heads, dropout,
    num_joints=num_joints, attn_bias_init=_bias_init)
```

---

### 2. `config.py`

In the `head` dict inside `model`, add `attn_bias_type` as a string literal:

**Current head dict (lines 131-147):**
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
),
```

**New head dict:**
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
    attn_bias_type='skeleton_init',
),
```

---

## Exact Parameter Values

| Parameter | Value |
|-----------|-------|
| `attn_bias` shape | `(70, 70)` |
| `adjacent_val` | `+0.5` |
| `pelvis_diag_val` (bias[0,0]) | `-0.5` |
| All other entries | `0.0` |
| Number of new parameters | 4900 |
| Total kinematic edges encoded | 21 body + 23 left hand + 23 right hand + 2 face = 69 |
| `attn_bias_type` in config | `'skeleton_init'` (string literal) |

---

## Expected Behaviour

- **At init**: self-attention logits for adjacent joint pairs are boosted by `+0.5` before softmax, encouraging kinematically-linked joints to attend to each other from the first step. Pelvis token (0) self-attention is softly suppressed by `-0.5`.
- **During training**: the bias continues to evolve from this warm-start. Adjacent joints are expected to reinforce their attention; the pelvis suppression may reduce contamination from body-joint queries.
- **Convergence**: expected faster convergence than Design 001 due to the warm-start. Potential simultaneous improvement in body MPJPE and pelvis MPJPE.

---

## Constraints and Invariants the Builder Must Preserve

1. **`_build_skeleton_attn_bias` must be placed at module level** (not inside a class), after all imports and before `_DecoderLayer`, so it is available when `Pose3dTransformerHead.__init__` calls it.
2. **Bidirectionality**: all edge pairs `(i, j)` must set both `bias[i, j]` and `bias[j, i]` to `+0.5`.
3. **No import of external graph libraries**: the edge list is hardcoded as a Python list of tuples; only `torch` is needed.
4. **The `attn_bias_type='none'` path** (when Builder copies this head file to a design without the skeleton feature) must fall back to `attn_bias_init=None`, which causes `_DecoderLayer` to register `torch.zeros(num_joints, num_joints)` — identical to baseline.
5. **`attn_bias_type` is a string literal in `config.py`** — no import needed. Value is exactly `'skeleton_init'`.
6. **`_DecoderLayer` must call `.float().clone()` on `attn_bias_init`** before wrapping in `nn.Parameter` to ensure it is a proper float32 leaf tensor, not a view.
7. **`attn_mask` semantics**: same as Design 001 — additive before softmax, shape `(70, 70)`, broadcast across batch and heads.
8. **No changes to loss, metric, data pipeline, backbone, `pelvis_utils.py`, or any invariant files.**
9. **Hand joint index ranges**: left hand 22–44 (23 joints), right hand 45–67 (23 joints), jaw=68, head_top=69. Total: 22 body + 23 left + 23 right + 2 face = 70 joints. Builder must verify indices do not exceed 69.
