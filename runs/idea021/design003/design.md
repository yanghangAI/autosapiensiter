# Design 003 — Low-Rank Factored Spatial Bias with Anatomical Gaussian Warm-Start

**Design Description:** Same factored cross-attention bias as design002 but row-biases for body joints 0–21 are warm-started with Gaussians centered at anatomical row positions (σ=4, α=1.0); hand joints 22–69 remain zero-initialized.

**Starting Point:** `baseline/`

---

## Overview

The core algorithm change: identical factored parameterization as design002:
```
B_i[h, w] = u_i[h] + v_i[w]
```
The difference is in initialization: for the 22 body joints (indices 0–21), `u_i` is initialized as a Gaussian centered at the expected row position `μ_i` in the 40-row feature grid (H'=40):
```
u_i[h] = α * exp(-(h - μ_i)^2 / (2σ^2))
```
with `α=1.0` and `σ=4.0`. Column biases `v_i` remain zero for all joints. Hand joints 22–69 use zero-initialized row biases.

This warm-start provides each body joint query a soft spatial prior pointing to its expected vertical location in the crop, accelerating convergence during the first ~5 epochs compared to design002 (cold start). The model refines these priors during training.

---

## Files to Change

1. `pose3d_transformer_head.py` — same `_DecoderLayer` and head changes as design002; add warm-start logic in `_init_head_weights()`.
2. `config.py` — add `use_cross_attn_bias=True`, `cross_attn_bias_type='factored_warmstart'`, `feat_h=40`, `feat_w=24`, `joint_row_prior=[...]` to the head dict.

No changes to `pelvis_utils.py`.

---

## Exact Changes

### `pose3d_transformer_head.py`

#### 1. `_DecoderLayer.forward()` — identical to design001 and design002

```python
def forward(self, queries: torch.Tensor,
            spatial_tokens: torch.Tensor,
            cross_attn_bias: 'torch.Tensor | None' = None) -> torch.Tensor:
    # Self-attention (unchanged)
    q = self.norm1(queries)
    q2 = self.self_attn(q, q, q)[0]
    queries = queries + self.dropout1(q2)

    # Cross-attention
    q = self.norm2(queries)
    if cross_attn_bias is not None:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens,
                              attn_mask=cross_attn_bias.to(q.dtype))[0]
    else:
        q2 = self.cross_attn(q, spatial_tokens, spatial_tokens)[0]
    queries = queries + self.dropout2(q2)

    # FFN (unchanged)
    queries = queries + self.ffn(self.norm3(queries))
    return queries
```

#### 2. `Pose3dTransformerHead.__init__()` — same as design002

Same new kwargs:
```python
use_cross_attn_bias: bool = False,
cross_attn_bias_type: str = 'full',
feat_h: int = 40,
feat_w: int = 24,
joint_row_prior: list = None,
```

Store all as instance attributes including `self.joint_row_prior = joint_row_prior`.

Same parameter allocation block as design001/002:
```python
if use_cross_attn_bias:
    if cross_attn_bias_type == 'full':
        self.cross_attn_bias = nn.Parameter(
            torch.zeros(num_joints, feat_h * feat_w))
    else:  # 'factored' or 'factored_warmstart'
        self.cross_attn_bias_row = nn.Parameter(
            torch.zeros(num_joints, feat_h))   # (70, 40), initialized below for body joints
        self.cross_attn_bias_col = nn.Parameter(
            torch.zeros(num_joints, feat_w))   # (70, 24), all zeros
```

#### 3. `Pose3dTransformerHead._init_head_weights()` — add warm-start logic

After the existing weight initialization code (query embeddings + output projections), append:

```python
if (self.use_cross_attn_bias
        and self.cross_attn_bias_type == 'factored_warmstart'
        and self.joint_row_prior is not None):
    h_coords = torch.arange(self.feat_h, dtype=torch.float32)  # (40,)
    sigma = 4.0
    alpha = 1.0
    for i, mu in enumerate(self.joint_row_prior[:22]):
        gauss = alpha * torch.exp(-(h_coords - mu) ** 2 / (2.0 * sigma ** 2))
        self.cross_attn_bias_row.data[i] = gauss
    # hand joints (indices 22–69) remain zero-initialized (already zeros from torch.zeros)
```

Key details:
- `self.joint_row_prior[:22]` — slice to exactly 22 body joints, even if config provides more.
- `h_coords = torch.arange(40)` runs from 0 (top of feature grid) to 39 (bottom).
- `sigma=4.0` spans approximately 4 grid cells on each side of the center.
- `alpha=1.0` sets the initial peak bias to 1.0 logit unit (moderate strength).
- Column biases are not modified — they remain zero.
- `self.cross_attn_bias_row.data[i] = gauss`: direct `.data` assignment bypasses the autograd graph, which is correct for initialization.

#### 4. `Pose3dTransformerHead.forward()` — identical to design002

```python
if self.use_cross_attn_bias:
    if self.cross_attn_bias_type == 'full':
        bias = self.cross_attn_bias
    else:
        bias = (self.cross_attn_bias_row.unsqueeze(-1) +
                self.cross_attn_bias_col.unsqueeze(-2))   # (70, 40, 24)
        bias = bias.view(self.num_joints, -1)              # (70, 960)
    decoded = self.decoder_layer(queries, spatial, cross_attn_bias=bias)
else:
    decoded = self.decoder_layer(queries, spatial)
```

---

### `config.py`

In the `model` dict, under `head=dict(...)`, add after `loss_weight_uv=1.0`:

```python
use_cross_attn_bias=True,
cross_attn_bias_type='factored_warmstart',
feat_h=40,
feat_w=24,
joint_row_prior=[12.0, 10.0, 14.0, 12.0, 9.0, 15.0, 7.0, 19.0, 21.0, 5.0,
                  3.0, 2.0, 11.0, 13.0, 11.0, 13.0, 9.0, 9.0, 15.0, 15.0, 12.0, 12.0],
```

The `joint_row_prior` list has exactly 22 float entries (one per body joint, index 0=pelvis to index 21). Values are row positions in the 40-row feature grid (0=top of crop, 39=bottom). The mapping reflects the BEDLAM2 centred crop anatomy at H'=40:

| Joint index | Joint name (approx) | Expected row (H'=40) |
|---|---|---|
| 0 | Pelvis (root) | 12.0 |
| 1 | L hip | 10.0 |
| 2 | R hip | 14.0 |
| 3 | Spine 1 | 12.0 |
| 4 | L knee | 9.0 |
| 5 | R knee | 15.0 |
| 6 | Spine 2 | 7.0 |
| 7 | L ankle | 19.0 |
| 8 | R ankle | 21.0 |
| 9 | Spine 3 | 5.0 |
| 10 | L foot | 3.0 |
| 11 | R foot | 2.0 |
| 12 | Neck | 11.0 |
| 13 | L collar | 13.0 |
| 14 | R collar | 11.0 |
| 15 | Head | 13.0 |
| 16 | L shoulder | 9.0 |
| 17 | R shoulder | 9.0 |
| 18 | L elbow | 15.0 |
| 19 | R elbow | 15.0 |
| 20 | L wrist | 12.0 |
| 21 | R wrist | 12.0 |

Note: These row values are approximate soft priors — the Gaussian width (σ=4) means any systematic offset within ~4 grid cells is quickly corrected by gradient updates. The model treats these as starting points, not hard constraints.

All config values are bool/str/int/float/list literals. No Python import statements. Fully MMEngine-compliant.

---

## Parameter Count

- `cross_attn_bias_row`: `70 × 40 = 2,800` scalars (22 warm-started, 48 zero)
- `cross_attn_bias_col`: `70 × 24 = 1,680` scalars (all zero)
- Total: `4,480` float32 scalars = ~17.5 KB. Same as design002.

---

## `attn_mask` Semantics

Identical to design001 and design002.

At epoch 0:
- For body joint `i` (0–21): `u_i[h] = exp(-(h - μ_i)^2 / 8)` peaked at row `μ_i`. Column bias zero. The warm-start bias is **not** zero — this design intentionally differs from the baseline at epoch 0, unlike design001/002.
- For hand joint `i` (22–69): all biases zero.
- Combined effect: body joint cross-attention logits receive a soft vertical prior at training start. The model's body joint queries will preferentially attend to vertical bands matching each joint's expected row, from epoch 1 onward.

---

## Invariants to Preserve

All invariants from design001 and design002 apply. Additionally:
- `self.joint_row_prior[:22]` slicing ensures the warm-start loop never accesses out-of-bounds indices even if `len(joint_row_prior) != 22`.
- `self.cross_attn_bias_row.data[i]` writes to `.data` directly — does not create a computation graph node. This is standard practice for weight initialization.
- `torch.arange(self.feat_h)` uses `self.feat_h=40` at runtime — consistent with the config value.
- Hand joints (22–69): `cross_attn_bias_row.data[22:]` retains the `torch.zeros()` initialization. No explicit zeroing needed.

---

## Expected Behavior After Change

- At epoch 1: body joint queries start with a soft vertical routing prior. Joints like L ankle (row ~19), R ankle (row ~21), head-top joints (rows 2–5) have meaningful non-zero bias pointing to their expected crop regions.
- During training: the warm-started biases are refined. Column biases are learned from scratch (starting at zero).
- Compared to design002: expected ~5–10 mm improvement in `mpjpe_body_val` within the same 20-epoch budget due to faster convergence in early epochs.
- Expected composite_val < 328 at stage-1 (vs. design002 target < 333, baseline ~346).
- At stage-2 (10 epochs on train400.txt from scratch with same pretrained backbone): the warm-start re-applies, giving the same early-epoch benefit.
- Checkpoint saves `cross_attn_bias_row` and `cross_attn_bias_col` in `state_dict`. On resume (preempted job), the saved (already-trained) bias values are loaded — the warm-start logic in `_init_head_weights()` only runs at model construction, before any training. This is correct: resumed training continues with the learned values, not the warm-started init values.

---

## Edge Cases

- **`_init_head_weights()` called before training starts**: warm-start runs at `__init__` time. On checkpoint resume, the checkpoint overwrites these values. Correct behavior.
- **`joint_row_prior=None` in config**: the warm-start block is guarded by `self.joint_row_prior is not None`. If `None`, row biases remain zero (fallback to design002 behavior). The config for design003 must provide the list.
- **`len(joint_row_prior) < 22`**: `self.joint_row_prior[:22]` with `enumerate()` iterates only over the provided entries; remaining body joints retain zero init. The config provides exactly 22 entries.
- **`feat_h=40` in config vs. H=40 from backbone**: if `feat_h` in config doesn't match the actual backbone stride, `h_coords` will have wrong length and `bias.view(self.num_joints, -1)` may fail. Builder must ensure `feat_h=40, feat_w=24` are consistent with the 640×384 input at 1/16 stride.
- **AMP warm-start values (~1.0 peak) in float16**: Gaussian peak of 1.0 is well within float16 range. No overflow risk.
