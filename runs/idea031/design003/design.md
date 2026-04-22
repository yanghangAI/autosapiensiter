**Design Description:** 2D spatial heatmap classification head for pelvis UV; soft Gaussian target (sigma=2 cells); KL heatmap loss weight 0.5; learnable scalar softmax temperature via softplus.

**Starting Point:** `baseline/`

---

## Algorithm

Identical architecture to idea031/design001, but adds a learnable scalar softmax temperature applied to `uv_logits` before softmax. The temperature is a single `nn.Parameter(torch.tensor(1.0))` passed through `F.softplus` (with a small min-clamp) to ensure positivity. The model can learn to sharpen (lower temperature → more peaked distribution → precise localization) or to diffuse (higher temperature → broader distribution → robust under uncertainty) as training proceeds. Gaussian target sigma=2.0 and heatmap loss weight=0.5 match design001; only the softmax temperature becomes learnable.

## Overview

Design C from idea031 — adaptive-sharpness variant. Gives the model one additional degree of freedom to match its own confidence to the task difficulty. Mirrors idea020's per-query temperature, but here a single scalar suffices since the pelvis is a single target. Output interface `pred['pelvis_uv']` is unchanged.

---

## Files to Change

1. `pose3d_transformer_head.py` — same structural changes as design001, plus: construct `self.uv_heatmap_temp = nn.Parameter(torch.tensor(1.0))` when `uv_heatmap_learnable_temp=True`, and divide `uv_logits` by `F.softplus(self.uv_heatmap_temp).clamp(min=1e-3)` before softmax in `forward()`.
2. `pelvis_utils.py` — same two helpers as design001 (identical code).
3. `config.py` — same kwargs as design001 but with `uv_heatmap_learnable_temp=True`.

---

## `pelvis_utils.py` Changes

Identical to design001. Add `uv_to_grid_coords` and `build_gaussian_heatmap_2d` helpers.

---

## `pose3d_transformer_head.py` Changes

Identical to design001 with the following additions activated by `uv_heatmap_learnable_temp=True`:

### 1. `__init__` — register the learnable temperature

Inside the gated `if self.use_uv_heatmap:` block, after constructing `self.uv_heatmap_proj`, add (already shown in design001 for completeness — this design actually uses it):

```python
if self.uv_heatmap_learnable_temp:
    self.uv_heatmap_temp = nn.Parameter(torch.tensor(1.0))
```

Initialization at `1.0` produces `softplus(1.0) ≈ 1.3133`, giving baseline-like sharpness at step 0. The parameter is a leaf tensor included in the default optimizer parameter group (same LR, same weight decay as the rest of the head).

### 2. `forward()` — apply the temperature to the softmax

When `self.use_uv_heatmap and self.uv_heatmap_learnable_temp`:

```python
temp = F.softplus(self.uv_heatmap_temp).clamp(min=1e-3)
uv_attn = F.softmax(uv_logits / temp, dim=-1)     # (B, H*W)
```

The `.clamp(min=1e-3)` is a numerical safety rail — softplus output of a parameter initialized at 1.0 will not approach 0 during normal training, but the clamp prevents pathological divide-by-near-zero if the optimizer drives the parameter to a very negative value (softplus → 0). No effect at init or under normal training.

Everything else in `forward()` is identical to design001 (soft-argmax reductions, stashing `self._uv_attn`, etc.).

### 3. `loss()` — unchanged relative to design001

Temperature is baked into `self._uv_attn`, so the cross-entropy/KL loss uses the temperature-scaled distribution automatically. No new loss term is added for the temperature.

### 4. `predict()` — unchanged

`predict()` calls `forward()` which applies the temperature; no additional change.

---

## Optimizer / Weight Decay Considerations

- `self.uv_heatmap_temp` is a scalar `nn.Parameter`. It is automatically picked up by the MMEngine optimizer under the default head parameter group.
- The existing baseline optimizer uses AdamW with a single parameter group (the head and backbone share a single group in the baseline config). The scalar temperature will receive the same weight decay as other head parameters. For a single-scalar parameter this is a nearly negligible effect; the Designer does not require a dedicated no-decay parameter group for this design. The Builder must NOT introduce a new optimizer parameter group — this would couple to multiple other invariants in the config and is out of scope for this idea.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add:

```python
use_uv_heatmap=True,
uv_heatmap_loss_weight=0.5,
uv_heatmap_sigma=2.0,
uv_heatmap_target='gaussian',
uv_heatmap_learnable_temp=True,
feat_h=40,
feat_w=24,
```

All values are bool/int/float/str literals. No Python `import` statements.

Everything else in `config.py` is unchanged.

---

## Invariants to Preserve

Same as design001, plus:
- Single optimizer parameter group — unchanged. Do NOT introduce a separate no-decay group for `uv_heatmap_temp`.
- `pred['pelvis_uv']` shape `(B, 2)` in `[-1, 1]` — preserved (temperature affects only the internal attention distribution, not the output shape).
- `self.uv_heatmap_temp` is a tensor `nn.Parameter` on the same device as the rest of the head; no explicit `.to(device)` is needed because MMEngine's `to()` on the parent module handles it.

---

## Expected Behaviour After Change

- At init: `softplus(1.0) ≈ 1.3133`, giving a slightly smoother softmax than design001 (whose effective temperature is 1.0). With zero-init on `uv_heatmap_proj`, `uv_logits = 0` and `uv_logits / temp = 0` regardless of temperature, so `uv_attn` is uniform at step 0 — numerically identical first step to design001.
- During training: gradient on `self.uv_heatmap_temp` comes from both losses:
  - SmoothL1 via soft-argmax: prefers sharper peaks when the distribution is already near GT (smaller temp).
  - Cross-entropy against Gaussian sigma=2.0 target: prefers a distribution of comparable width to the target, i.e., not too sharp (larger temp).
  - The equilibrium temperature encodes the model's effective confidence in its localization. Empirically this often settles between 0.5 and 2.0 for similar heatmap heads.
- Parameter count delta: +1 scalar param relative to design001 (−257 + 1 = −256 vs baseline). Negligible.
- Memory/speed: identical to design001.
- Shape/interface of `pred['pelvis_uv']` unchanged.

---

## Edge Cases and Constraints

- Identical to design001 (row/col convention, UV normalization, GT out-of-range handling, AMP dtype, `self._uv_attn` lifetime).
- **Temperature extreme values**: softplus has range `(0, ∞)`. The `clamp(min=1e-3)` on softplus output prevents divide-by-zero if the parameter is driven to a large negative value. At `self.uv_heatmap_temp = -10`, `softplus ≈ 4.5e-5`, which clamps to `1e-3`. No NaN/Inf is produced for any reachable value.
- **AMP interaction**: `self.uv_heatmap_temp` is an fp32 parameter (default). `F.softplus` produces fp32; dividing an fp16 `uv_logits` by an fp32 temperature under AMP promotes `uv_logits` to fp32 for the division, which is safe. The subsequent softmax returns fp16 under AMP. No explicit cast is required from the Builder.
- **Checkpoint resume**: the learnable `uv_heatmap_temp` is part of `state_dict` and is saved/loaded automatically by `CheckpointHook`. No special handling. If the Builder switches from design001 to design003 mid-run (not expected), the state-dict keys will mismatch and MMEngine will report a key error; the user should start fresh for design003.

---

## Target Metrics (Stage 1)

- `composite_val < 325` (vs. best prior 323.75; similar target to design001 with robustness upside)
- `mpjpe_pelvis_val < 600` (vs. best prior 608)
- `mpjpe_abs_val < 780` (vs. baseline 833)
- `mpjpe_body_val` not expected to regress.

If design C matches or beats design A, the learnable temperature is a worthwhile permanent addition to the UV heatmap head. If design C underperforms design A, the fixed temperature of 1.0 is already near-optimal and the added parameter is not useful — a decisive null result.
