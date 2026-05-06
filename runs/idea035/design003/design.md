# Design 003 — Spatially-Shuffled Depth Ablation

**Design Description:** Replace the depth channel (input channel index 3) of the preprocessed 4-channel `(B, 4, H, W)` tensor with a per-sample random pixel permutation of the same channel, in a thin `RGBDPoseDataPreprocessor` subclass `DepthAblationDataPreprocessor` (mode='shuffle'). The marginal histogram of depth values per sample is preserved exactly; spatial alignment with RGB is destroyed. RGB channels (indices 0..2) are untouched. Tests whether depth contributes as a per-pixel image-aligned signal vs as a bulk statistic.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add a new module-level class `DepthAblationDataPreprocessor` (subclass of `RGBDPoseDataPreprocessor`) registered with `@MODELS.register_module()`. Same class as designs 001 and 002 — only the `mode` kwarg in `config.py` differs.
2. `config.py` — change `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode='shuffle')`. No other edits.
3. `pelvis_utils.py` — untouched.

All invariant files (`bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `rgbd_pose3d.py`, `train.py`, `tools/train.py`, `infra/*`) are untouched.

---

## Algorithm

### 1. New class in `pose3d_transformer_head.py`

Add the import near the top of the file (after the existing `from mmpose.models.heads.base_head import BaseHead` line):

```python
from mmpose.models.data_preprocessors.rgbd_data_preprocessor import (
    RGBDPoseDataPreprocessor,
)
```

Append at the end of the file (module scope, after `Pose3dTransformerHead`):

```python
@MODELS.register_module()
class DepthAblationDataPreprocessor(RGBDPoseDataPreprocessor):
    """Depth-channel ablation wrapper for RGBDPoseDataPreprocessor.

    See design A (idea035/design001) for full docstring; supports modes
    'zero' | 'gauss' | 'shuffle'. RGB channels (0..2) are unchanged; only
    channel index 3 (depth) is replaced.
    """

    def __init__(self, mode: str = 'zero'):
        super().__init__()
        assert mode in ('zero', 'gauss', 'shuffle'), \
            f'unknown DepthAblationDataPreprocessor mode {mode!r}'
        self.mode = str(mode)

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=training)
        inputs = data.get('inputs', None)
        if inputs is None or not torch.is_tensor(inputs):
            return data
        if inputs.dim() != 4 or inputs.shape[1] < 4:
            return data
        if self.mode == 'zero':
            inputs[:, 3:4].zero_()
        elif self.mode == 'gauss':
            inputs[:, 3:4] = torch.randn_like(inputs[:, 3:4])
        elif self.mode == 'shuffle':
            B, _, H, W = inputs.shape
            flat = inputs[:, 3].reshape(B, H * W)
            for b in range(B):
                perm = torch.randperm(H * W, device=flat.device)
                flat[b] = flat[b][perm]
            inputs[:, 3] = flat.reshape(B, H, W)
        data['inputs'] = inputs
        return data
```

For Design 003, only the `'shuffle'` branch is exercised.

### 2. `config.py` change

Locate the `model = dict(...)` block and the `data_preprocessor` key (baseline `config.py:140`). Replace:

```python
    data_preprocessor=dict(type='RGBDPoseDataPreprocessor'),
```

with:

```python
    data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='shuffle'),
```

No other changes to `config.py`.

---

## Exact Expected Behaviour

- Every forward step (train and val), each sample's depth channel is independently shuffled across all `H*W` pixels using a fresh `torch.randperm`. The set of depth *values* in each sample is preserved exactly (it is a permutation, not a sample); only the 2D pixel arrangement is destroyed.
- RGB channels are unchanged. The exact mean, variance, min, max, and histogram of the depth channel per sample equal baseline.
- Stage 1 runs 20 epochs; stage 2 expected not to fire.
- All six tracked CSV columns are populated; semantics unchanged.

---

## Constraints / Invariants the Builder Must Preserve

1. **`RGBDPoseDataPreprocessor` itself is not modified.** Subclass lives only in `pose3d_transformer_head.py`.
2. **RGB channels (0..2) must NOT be modified.** Only channel 3 is shuffled.
3. **Per-sample independent permutation.** Use a fresh `torch.randperm(H * W, device=flat.device)` for each `b` in `range(B)`. Do NOT share one permutation across samples (that would still preserve some inter-sample alignment patterns).
4. **Permute by pixel index, not by row or column.** A full `H*W` permutation is required to fully destroy 2D alignment.
5. **Operate on the device of the inputs tensor.** `torch.randperm(..., device=flat.device)` avoids any host-device transfer.
6. **No fixed seed inside `forward`.** Permutations are fresh per step; global seed (2026) set once by the trainer governs whole-run determinism.
7. **Registry name `DepthAblationDataPreprocessor`** must exactly match the `type=` string in `config.py`.
8. **Defensive fallbacks:** if `inputs` is missing, not a tensor, fewer than 4 channels, or not 4-D — return parent's output unchanged.
9. **Body-only joint loss (indices 0–21)** — unchanged.
10. **`persistent_workers=False`** — unchanged.
11. **MMEngine config constraint:** `data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='shuffle')` uses only string literals.
12. **No change to head architecture, optimizer, schedule, or any other config key.**
13. **Output dict keys/shapes** from the head are unchanged.

---

## Edge Cases

- **Cost of `randperm` per sample:** `B=4`, `H*W = img_h * img_w = 640*384 = 245760`. Four `randperm(245760)` calls per step on GPU is well under 1 ms. Negligible relative to backbone forward.
- **AMP dtype:** indexing `flat[b][perm]` preserves dtype. No casts.
- **Reshape correctness:** `flat = inputs[:, 3].reshape(B, H * W)` and `inputs[:, 3] = flat.reshape(B, H, W)` round-trip exactly because the underlying tensor is contiguous after `cast_data`. If contiguity is ever in doubt, calling `inputs[:, 3].contiguous().reshape(...)` is acceptable but typically unnecessary.
- **In-place vs out-of-place:** the indexed assignment `inputs[:, 3] = ...` writes into the existing tensor; the `flat[b] = flat[b][perm]` line creates a temporary then writes back into `flat[b]` via fancy-indexed assign — this is correct since `flat` is a view of `inputs[:, 3]`. Builder may instead build `new_flat = torch.stack([flat[b][torch.randperm(...)] for b in range(B)], dim=0)` and assign once; semantically equivalent.
- **Validation pass:** val depth is also shuffled, by design. There is no train/val mismatch.
