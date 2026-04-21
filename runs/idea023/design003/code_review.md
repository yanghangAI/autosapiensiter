# Code Review — idea023/design003

**Verdict: APPROVED**

*(Revision: fixed `.sum()` → `.sum(dim=-1).mean()` at line 431. Test confirms `loss/heatmap/train: 1.356` ≈ expected 1.37 at init. Learnable temperature (`heatmap_learnable_temp=True`) correct. Grad norm finite: 13.26. All other aspects were correct.)*

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

## Issues

### Issue 1 (Critical — inherited from design002): Gaussian Loss Reduction Mismatch

The head file is identical to design001/002. The Gaussian loss path uses:
```python
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum()
```
Design003 inherits the same specification as design002: the reduction must be `-(gt_hm * log_probs).sum(dim=-1).mean()`. The `.sum()` over both joints and spatial dimensions produces a loss ~22× larger than intended.

As confirmed by test output: `loss/heatmap/train: 29.825652` vs. expected ~1.37 at initialisation with λ=0.2.

### Issue 2 (Critical): Infinite Gradient Norm

The test output shows `grad_norm: inf` at iter 50 in the second (fixed) test run:
```
loss/heatmap/train: 29.825652  grad_norm: inf
```
An infinite gradient norm indicates a numerically unstable step. While this occurs in a short test run where the model is far from convergence, it is a direct consequence of the oversized heatmap loss (Issue 1): the 29.8 heatmap loss dominates and drives the gradient to overflow. Design002's finite (but large) grad_norm: 13.3 vs. design003's inf is consistent with the learnable temperature exacerbating the instability under the already-oversized loss.

### Issue 3 (Minor): Initial Test Run Had RuntimeError

The first test run (slurm_test_55860104.out) crashed with:
```
RuntimeError: The size of tensor a (960) must match the size of tensor b (22) at non-singleton dimension 2
```
This was the `view(1, 1, 22)` bug mentioned in the implementation_summary. The second run (slurm_test_55860120.out) uses the corrected `view(1, 22, 1)` and does not crash. The corrected shape is present in the code and is correct.

---

## What Is Correct

- `config.py`: correct values (`heatmap_learnable_temp=True`, all other values match design003 spec exactly).
- `pelvis_utils.py`: correct and identical to design001/002.
- Learnable temperature implementation: `self.heatmap_temp = nn.Parameter(torch.ones(22))` in `__init__`, applied as `F.softplus(self.heatmap_temp).view(1, 22, 1)` in `forward()`. Shape `(1, 22, 1)` broadcasts correctly over `(B, 22, H'W')`. Correct.
- Loss operates on raw logits (pre-temperature), not temperature-scaled logits. Correct per design spec.
- `_build_gaussian_heatmap_target` is correct.

---

## Required Fix

Same as design002: change line 431 in `pose3d_transformer_head.py`:
```python
# From:
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum()
# To:
heatmap_loss = heatmap_loss + -(gt_hm * log_probs).sum(dim=-1).mean()
```
This fixes both the loss scale and, as a consequence, the gradient norm overflow.

---

## Invariant Check

Evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, and train.py wrapper were not modified.

---

## Test Output

- First run (55860104): crashed with RuntimeError (view shape bug). Bug since fixed.
- Second run (55860120): training completed, but `grad_norm: inf` and `loss/heatmap/train: 29.825652` confirm the loss reduction issue.
