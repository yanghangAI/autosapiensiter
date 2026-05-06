# Design 001 — Zero Depth Ablation

**Design Description:** Replace the depth channel (input channel index 3) of the preprocessed 4-channel `(B, 4, H, W)` tensor with a constant zero tensor in a thin `RGBDPoseDataPreprocessor` subclass `DepthAblationDataPreprocessor` (mode='zero'). RGB channels (indices 0..2) are untouched. Tests pure information removal: backbone receives a deterministic constant in the depth slot.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pose3d_transformer_head.py` — add a new module-level class `DepthAblationDataPreprocessor` (subclass of `RGBDPoseDataPreprocessor`) registered with `@MODELS.register_module()`. ~25 lines, appended at end of file (after the existing `Pose3dTransformerHead` definition). No other changes to the head or its existing classes.
2. `config.py` — change `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode='zero')`. No other edits.
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

    After the parent's pass-through normalization, replaces channel index 3
    (the depth channel) of the 4-channel input tensor according to ``mode``.

    mode:
        'zero'    — fill depth channel with 0.0 (post-normalization mean ~ 0;
                    represents "no information").
        'gauss'   — replace depth channel with torch.randn_like(depth) so its
                    variance scale is preserved but signal is destroyed.
        'shuffle' — apply a per-sample random pixel permutation to the depth
                    channel; preserves the marginal histogram exactly while
                    destroying spatial alignment with RGB.

    RGB channels (indices 0..2) are passed through unchanged.
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
        # Depth lives at channel index 3 of the (B, 4, H, W) tensor.
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

For Design 001, only the `'zero'` branch is exercised; the other branches are inert.

### 2. `config.py` change

Locate the `model = dict(...)` block and the `data_preprocessor` key (baseline `config.py:140`). Replace:

```python
    data_preprocessor=dict(type='RGBDPoseDataPreprocessor'),
```

with:

```python
    data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='zero'),
```

No other changes to `config.py`. `custom_imports` already lists `'pose3d_transformer_head'`, so the new class will be registered automatically when the head module is imported.

---

## Exact Expected Behaviour

- Every forward step receives an input tensor whose 4th channel is exactly `0.0` everywhere. RGB channels are unchanged.
- The backbone, head, loss, evaluator, dataset, transforms, optimizer, LR schedule, batch size, AMP, seed, and number of epochs are all identical to baseline.
- Stage-1 training runs for 20 epochs on `train100.txt`. Stage 2 is **expected not to fire** (corrupted depth should not beat baseline composite_val).
- All six tracked CSV columns (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`) are populated unchanged in semantics.
- The depth conv at the stem of `SapiensBackboneRGBD` receives a constant input; gradients still flow but the depth pathway provides no per-sample signal.

---

## Constraints / Invariants the Builder Must Preserve

1. **`RGBDPoseDataPreprocessor` itself is not modified.** The new subclass lives only in `pose3d_transformer_head.py`. The invariant file `mmpose/models/data_preprocessors/rgbd_data_preprocessor.py` is untouched.
2. **RGB channels (indices 0..2) must NOT be modified.** Only channel 3 (depth) is replaced.
3. **In-place modification is acceptable** because the data tensor returned by `cast_data` in the parent is a fresh batch tensor; downstream consumers operate on the same tensor reference. Do not detach or clone unnecessarily.
4. **Registry name `DepthAblationDataPreprocessor`** must exactly match the `type=` string in `config.py`.
5. **Robustness:** if `inputs` is missing, not a tensor, has fewer than 4 channels, or is not 4-D, fall back to returning the parent's output unchanged (defensive only; baseline always provides 4-channel float tensors).
6. **Body-only joint loss (indices 0–21)** — unchanged.
7. **`persistent_workers=False`** — unchanged.
8. **MMEngine config constraint:** `data_preprocessor=dict(type='DepthAblationDataPreprocessor', mode='zero')` uses only string literals; no Python imports inside the config.
9. **`custom_imports` must remain ordered so that `'pose3d_transformer_head'` is imported.** It already is in baseline; do not remove it.
10. **No change to head architecture, optimizer, or schedule.** Only the data-preprocessor type and one kwarg change in `config.py`.
11. **Output dict keys/shapes** from the head (`joints`, `pelvis_depth`, `pelvis_uv`) are unchanged.

---

## Edge Cases

- **AMP dtype:** The parent's `cast_data` returns the inputs in their default dtype (typically fp32 on CPU/GPU device transfer; AMP autocast is applied inside the model, not the preprocessor). `inputs[:, 3:4].zero_()` and `torch.randn_like` both preserve dtype; no cast needed.
- **CPU vs GPU:** `cast_data` moves tensors to the model device first; the in-place op runs on the same device.
- **Validation pass:** the same preprocessor runs at val time, so val depth is also zeroed — this is intentional for the ablation. There is no train/val mismatch.
- **Backbone first-conv kernel for depth:** has a 1-channel weight slice; receiving zeros means it contributes only its bias term. This is the desired ablation semantics.
