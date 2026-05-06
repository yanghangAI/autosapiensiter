## Design Review — idea035 / design003 (Spatially-Shuffled Depth Ablation)

**Verdict: APPROVED**

### Coverage check
- Design Description: explicit (per-sample full `H*W` random pixel permutation of channel 3; preserves marginal histogram, destroys spatial alignment; RGB untouched; mode='shuffle').
- Starting point: `baseline/`.
- Files to modify: `pose3d_transformer_head.py` (same class) + `config.py` (`mode='shuffle'`). `pelvis_utils.py` untouched. All invariants listed.
- Algorithmic change: complete code for `'shuffle'` branch — flatten to `(B, H*W)`, per-sample `torch.randperm(H*W, device=flat.device)`, reshape back. Acceptable equivalent (`torch.stack([...])`) noted.
- Config values: `mode='shuffle'` set; assert on allowed modes.
- Training/loss/data: only depth content changes; loss, schedule, AMP, persistent_workers preserved.
- Constraints/edge cases: cost analysis for `randperm` (B=4, H*W=245760), AMP dtype, contiguity of the depth-channel view, in-place vs out-of-place semantics, val-pass behavior all enumerated. Explicit warning to use per-sample independent permutations (do not share across batch).

### Invariant compliance
- `rgbd_data_preprocessor.py` untouched; subclass lives in `pose3d_transformer_head.py`.
- Dataset, transforms, backbone, metric, train.py, infra/* untouched.
- MMEngine config uses only string literals.

### Notes
- Class definition shared with design001/002; only the runtime-selected branch changes.

No fixes required.
