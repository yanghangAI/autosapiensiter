# Code Review — idea023/design002

**Verdict: APPROVED**

*(Revision: fixed `.sum()` → `.sum(dim=-1).mean()` at line 431. Test confirms `loss/heatmap/train: 1.356` ≈ expected 1.37 at init. All other aspects were correct.)*

---

*(Previous rejection below)*

**Previous Verdict: REJECTED**

---

## review-check-implementation

PASSED.

---

## Files Changed vs. Design

- `code/pelvis_utils.py` — required by design. PRESENT.
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT.

No unapproved file changes.

---

## Critical Issue: Gaussian Loss Reduction Mismatch

### What the design specifies (design002, section 2e):
```python
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum(dim=-1).mean()
```
This sums over the spatial (H'W') dimension, then averages over the 22 joints. Per-sample contribution is ~`mean_over_joints(KL_per_joint)` ≈ on the order of 1 nat at initialisation.

### What the code implements:
```python
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum()
```
This sums over **both** the spatial dimension AND the joint dimension. Per-sample contribution is ~`22 × sum_over_960_tokens(...)`, making the per-sample value approximately **22× larger** than specified.

### Impact:
At initialisation, `heatmap_proj` weights are zero → uniform distribution over 960 tokens → KL ≈ log(960) ≈ 6.87 nats per joint. The design specifies `.sum(dim=-1).mean()` which yields ~6.87 nats per sample, scaled by λ=0.2 → ~1.37 effective contribution to the total loss. The actual code's `.sum()` yields ~22 × 960 × (1/960) × 6.87 ≈ 22 × 6.87 ≈ 151 per sample (before batch normalisation). After `/B_hm`, the per-step heatmap loss is ~151/4 ≈ 38 — which matches the observed `loss/heatmap/train: 29.825396` at iter 50.

This means the effective heatmap loss weight is approximately 22× the specified λ=0.2, or ~4.4. The total loss (~32.7) is dominated almost entirely by the heatmap term (~29.8) rather than the 3D regression losses (~0.9 combined). This is the opposite of the design's intent (λ=0.2 was chosen specifically to keep the primary 3D regression loss dominant).

The test output confirms: `loss: 32.695`, `loss/heatmap/train: 29.825` — heatmap loss is 91% of total loss. This will destabilise 3D regression training.

---

## Additional Issues

### `config.py`
Config values are correct (`heatmap_loss_weight=0.2`, `heatmap_target='gaussian'`, `heatmap_sigma=2.0`, `heatmap_temperature=1.0`, `heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24`). No issues here.

### `pelvis_utils.py`
Correct and identical to design001.

### `_build_gaussian_heatmap_target`
Implementation is correct (uses `indexing='ij'`, normalises to sum=1, handles boundary via exp decay). The issue is not in the target construction but in the loss reduction applied to it.

---

## Required Fix

Change line 431 in `pose3d_transformer_head.py` from:
```python
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum()
```
to:
```python
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum(dim=-1).mean()
```
This matches the design002 specification and restores the intended loss scale (~1.37 at initialisation, not ~30).

---

## Invariant Check

Evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, and train.py wrapper were not modified.

---

## Test Output

- Training completed without crash: "Done training!" observed.
- `loss/heatmap/train: 29.825396` — confirms the 22× scaling bug. At λ=0.2 and correct `.sum(dim=-1).mean()`, expected value would be ~1.37.
- `grad_norm: 13.339081` — large but finite; however, the 3D regression signal is effectively swamped by the heatmap loss from the first iteration.
