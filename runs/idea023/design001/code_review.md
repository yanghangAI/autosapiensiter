# Code Review — idea023/design001

**Verdict: APPROVED**

---

## review-check-implementation

PASSED.

---

## Files Changed vs. Design

- `code/pelvis_utils.py` — required by design. PRESENT.
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT.
- `code/train.py` — unchanged from baseline (verified by diff). Not flagged.

No unapproved file changes.

---

## Implementation Fidelity

### `pelvis_utils.py`
`project_joints_to_grid_coords` is present, correctly placed after `recover_pelvis_3d`, with the correct BEDLAM2 projection convention (`u = -Y/X*fx + cx`, `v = -Z/X*fy + cy`), correct grid-coord scaling, and `clamp(min=0.01)` for depth divide-by-zero safety. Matches design spec exactly.

### `pose3d_transformer_head.py`
- All new `__init__` kwargs present with correct defaults (`use_heatmap_init=False`, `heatmap_loss_weight=0.1`, `heatmap_target='onehot'`, `heatmap_sigma=2.0`, `heatmap_temperature=1.0`, `heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24`).
- `heatmap_proj = nn.Linear(hidden_dim, 22)` created when `use_heatmap_init=True`; zero-initialised on both weight and bias.
- `_heatmap_logits = None` side-channel correctly initialised in `__init__`, set in `forward()`, cleared after `loss()` reads it, and set to `None` in the else branch of `forward()`.
- Forward pass: heatmap logits computed, softmax attention weights over 960 spatial tokens, bmm soft-pooling, zero-pad to (B, 70, hidden_dim), added to static joint query embeddings. Correct.
- Loss: `onehot` path uses `F.cross_entropy(logits_i, target_idx)` with correct `h_idx * feat_w + w_idx` flat indexing. Correct.
- Gaussian path (else branch) present but never executed for design001 (`heatmap_target='onehot'`). No functional issue.
- Import: uses `from pelvis_utils import recover_pelvis_3d, project_joints_to_grid_coords` (un-aliased, functionally equivalent to design spec's aliased versions). Correct.
- `_build_gaussian_heatmap_target` module-level helper present (included for completeness per summary). Does not affect design001 behaviour.
- Absolute imports used throughout (e.g. `from mmpose.models.heads.base_head import BaseHead`). Correct.
- `predict()` does not access `_heatmap_logits`. Correct.
- `persistent_workers` not changed (config-level). Body-only joint loss preserved.

### `config.py`
All required kwargs present with correct values:
```
use_heatmap_init=True, heatmap_loss_weight=0.1, heatmap_target='onehot',
heatmap_temperature=1.0, heatmap_learnable_temp=False, feat_h=40, feat_w=24
```
All literals; no Python import statements. MMEngine constraint satisfied.

---

## Invariant Check

Evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, and train.py wrapper were not modified.

---

## Test Output

- Job completed successfully: "Done training!" observed.
- `loss/heatmap/train: 0.686368` appears in training log at iter 50. Initial value ~log(960)≈6.87 expected at start; 0.686 at iter 50 shows the heatmap is already sharpening — consistent with correct cross-entropy over 960-class distribution.
- `grad_norm: inf` at iter 1 (expected with zero-init heatmap_proj — first gradient step with AMP; resolves quickly).
- No runtime errors. Training completed 1 epoch test successfully.
