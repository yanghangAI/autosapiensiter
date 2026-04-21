**APPROVED**

**Design:** idea022/design001 — 2-layer cascaded decoder with dynamic Gaussian reprojection bias (fixed σ=4, γ=2), no auxiliary loss.

**Verdict:** APPROVED

---

## Review Summary

The design is complete, explicit, and unambiguous. The Builder can implement it without guessing on any significant point.

### Feasibility and Completeness

All three allowed files are addressed:
- **`pelvis_utils.py`**: New helper `project_joints_to_feat_grid` is fully specified with exact signature, docstring, projection convention (BEDLAM2: X=forward, Y=left, Z=up), and clamping behavior.
- **`pose3d_transformer_head.py`**: All sub-changes are enumerated — imports (2a), `_build_gaussian_bias` function (2b), `_DecoderLayer.forward` modification (2c), `__init__` expansion (2d), `forward()` multi-layer loop (2e), `loss()` bias construction block (2f). Exact code snippets are provided for each.
- **`config.py`**: Exact kwargs and their literal values are specified.

### Architecture Correctness

1. **`nn.MultiheadAttention.num_heads` attribute**: `_DecoderLayer` uses `self.cross_attn = nn.MultiheadAttention(...)`. Section 2c references `self.cross_attn.num_heads` — this attribute exists on `nn.MultiheadAttention`. Correct.

2. **attn_mask shape**: The design correctly specifies `(B*nheads, J, H'W')` for per-sample dynamic bias with `batch_first=True`. PyTorch MHA accepts this shape. Correct.

3. **`_reproj_bias` side-channel**: Bias is set in `loss()`, consumed in `forward()`, and cleared to `None` at the end of `forward()`. This prevents stale bias leaking into validation (where `predict()` calls `forward()` directly without setting the bias). Correct.

4. **`torch.no_grad()` for intermediate layer-0**: For Design A (no auxiliary loss), the intermediate run is wrapped in `torch.no_grad()`. The full `self.forward(feats)` re-runs all layers with autograd. This means layer-0 is run twice per training step — once no-grad for bias construction, once with autograd for the main loss. The design explicitly accepts this overhead and states it is intentional (bias treated as a data-dependent but gradient-free prior). Correct.

5. **`recover_pelvis_3d` return shape**: Returns `(B, 3)`. Used as `recover_pelvis_3d(layer1_depth[i:i+1], ...) → (1, 3)`, then `layer1_joints[i] + pelvis → (J, 3) + (1, 3)` broadcasts to `(J, 3)`. Correct.

6. **Feature grid orientation**: The design confirms spatial tokens use `feat.flatten(2).transpose(1, 2)` (row-major H×W order) and specifies `indexing='ij'` in `torch.meshgrid` for consistency. `feat_h=40, feat_w=24` maps to the 640×384 input at stride 16. Correct.

7. **AMP cast**: The design explicitly requires `.to(q.dtype)` cast before passing `attn_mask` to MHA. Correct.

8. **Output shape invariant**: `forward()` returns `{'joints': (B, 70, 3), 'pelvis_depth': (B, 1), 'pelvis_uv': (B, 2)}` unchanged. Correct.

### Invariant Preservation

The design does not modify `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, or any infra files. Loss is restricted to body joints (indices 0–21). `persistent_workers=False` unchanged. No Python import statements in config.py.

### Minor Notes (not blocking)

- The design imports `torch.nn.functional as F` in pose3d_transformer_head.py, but Design A's `loss()` and `_build_gaussian_bias` do not use `F`. This is acceptable as a forward-compatibility import for Designs B/C which share the same file.
- The design correctly removes the single `self.decoder_layer` and replaces it with `self.decoder_layers` (ModuleList). The Builder must ensure no stale references to `self.decoder_layer` remain.

### Config

All config values are bool/float/int literals. No Python import statements. MMEngine constraint satisfied.
