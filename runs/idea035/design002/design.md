# Design 002 — Gaussian Noise Depth Ablation

**Design Description:** Replace the depth channel (input channel index 3) of the preprocessed 4-channel `(B, 4, H, W)` tensor with `torch.randn_like(...)` (per-step fresh i.i.d. unit-variance Gaussian noise) in a thin `RGBDPoseDataPreprocessor` subclass `DepthAblationDataPreprocessor` (mode='gauss'). RGB channels (indices 0..2) are untouched. Tests whether the backbone exploits depth content vs depth statistics.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add a new module-level class `DepthAblationDataPreprocessor` (subclass of `RGBDPoseDataPreprocessor`) registered with `@MODELS.register_module()`. ~25 lines, appended at end of file. Identical class definition across designs 001/002/003 — only the `mode` kwarg in config differs. (Build the same class once; runtime mode is selected from config.)
2. `config.py` — change `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode='gauss')`. No other edits.
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

For Design 002, only the `'gauss'` branch is exercised.

### 2. `config.py` change

Locate the `model = dict(...)` block and the `data_preprocessor` key (baseline `config.py:140`). Replace:

```python
    data_preprocessor=dict(type='RGBDPoseDataPreprocessor'),
```

with:

```python
    data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='gauss'),
```

No other changes to `config.py`.

---

## Exact Expected Behaviour

- Every forward step (train and val) receives an input tensor whose 4th channel is fresh i.i.d. `N(0, 1)` noise, sampled independently per step on the input tensor's device. RGB channels are unchanged.
- The depth-channel statistics differ from baseline: baseline depth is normalized via `PackBedlamInputs` to a roughly zero-mean, finite-variance distribution; `randn_like` is mean=0, std=1 by construction. The exact post-normalization scale of baseline depth is not unit-variance; this design intentionally substitutes a fixed unit-variance reference distribution. The point of the ablation is "noise with right *order-of-magnitude* scale, no signal," not exact distribution matching.
- Stage 1 runs 20 epochs; stage 2 expected not to fire.
- All six tracked CSV columns are populated; semantics unchanged.

---

## Constraints / Invariants the Builder Must Preserve

1. **`RGBDPoseDataPreprocessor` itself is not modified.** Subclass lives only in `pose3d_transformer_head.py`.
2. **RGB channels (0..2) must NOT be modified.** Only channel 3 is replaced.
3. **`torch.randn_like(inputs[:, 3:4])`** must use `randn_like` (not `randn(...)`), so the new tensor inherits dtype and device automatically. No manual `.to(...)` casts.
4. **No fixed seed inside `forward`.** Noise is fresh per step. Determinism of the whole training run is governed by the global seed (2026), set once by the trainer; do not call `torch.manual_seed` inside the preprocessor.
5. **Registry name `DepthAblationDataPreprocessor`** must exactly match the `type=` string in `config.py`.
6. **Defensive fallbacks:** if `inputs` is missing, not a tensor, fewer than 4 channels, or not 4-D — return parent's output unchanged.
7. **Body-only joint loss (indices 0–21)** — unchanged.
8. **`persistent_workers=False`** — unchanged.
9. **MMEngine config constraint:** `data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='gauss')` uses only string literals.
10. **No change to head architecture, optimizer, schedule, or any other config key.**
11. **Output dict keys/shapes** from the head are unchanged.

---

## Edge Cases

- **AMP dtype:** `randn_like` returns the same dtype as `inputs[:, 3:4]`; no autocast worries inside the preprocessor (which runs in fp32 by default; AMP autocast is scoped to the model forward).
- **Determinism:** the val pass at each epoch will see different per-step noise across epochs and across runs (modulo the global seed). The ablation conclusion is statistical, not point-equality. Acceptable.
- **CPU vs GPU:** `cast_data` moves to model device; `randn_like` produces noise on the same device — no host-device transfer.
- **No NaN/Inf risk** — `randn_like` is finite-valued.
