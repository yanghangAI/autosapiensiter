# Code Review — idea018/design002

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea018/design002` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are within the allowed set. `pelvis_utils.py` verified identical to baseline (no diff).

---

## Implementation vs. Design Checklist

### `_DecoderLayer.forward()` — optional `attn_logit_bias` argument

Design requires: identical to Design 001 modification.

Code at lines 101–139: Identical implementation — shape expansion `(B, N_spatial) → (B*num_heads, Nq, N_spatial)` via unsqueeze+expand+reshape, float `attn_mask`, else-branch unchanged. All requirements met.

### `Pose3dTransformerHead.__init__()` — new kwargs and modules

Design requires:
- `depth_gate_type: str = 'none'`
- `depth_probe_loss_weight: float = 0.0`
- No `depth_gate_sigma` kwarg (sigma is learned, not fixed)
- When `depth_gate_type == 'gaussian_learnable_sigma'`: create `depth_probe_global: Linear(hidden_dim, 1)`, `depth_probe_token: Linear(hidden_dim, 1)`, `self.log_sigma = nn.Parameter(torch.zeros(1))`
- Store `self.depth_probe_loss_weight = depth_probe_loss_weight`

Code at lines 163–225:
- `depth_gate_type` and `depth_probe_loss_weight` kwargs present with correct defaults
- No `depth_gate_sigma` kwarg — correct (learnable sigma)
- `self.depth_probe_loss_weight` stored at line 216
- `self.log_sigma = nn.Parameter(torch.zeros(1))` at line 223
- Both linear probes created under `gaussian_learnable_sigma` gate
- No `depth_gate_sigma_buf` buffer — correct (not needed for learnable sigma)

All requirements met.

### `_init_head_weights()` — zero-init depth probes

Design requires: when `depth_gate_type == 'gaussian_learnable_sigma'`, zero-init weight and bias of both probes; note that `log_sigma` is already zero via `torch.zeros(1)` in `__init__`.

Code at lines 240–245: Zero-init of all four weights/biases for both probes. Comment confirms `log_sigma` is already 0.0. Correct.

### `forward()` — compute gate with learnable sigma, cache z_hat

Design requires:
- `z_hat = depth_probe_global(spatial.mean(dim=1))`
- `z_tok = depth_probe_token(spatial).squeeze(-1)`
- `sigma = torch.exp(self.log_sigma).clamp(min=0.01)`
- `depth_err = (z_tok - z_hat) / sigma`
- `attn_logit_bias = -0.5 * depth_err ** 2`
- `self._depth_probe_z_hat = z_hat` (cache for auxiliary loss)

Code at lines 285–297: All requirements met. `clamp(min=0.01)` prevents division-by-zero. `self._depth_probe_z_hat = z_hat` cached correctly.

### `loss()` — auxiliary depth probe loss

Design requires: after `loss/uv/train`, add:
```
if self.depth_probe_loss_weight > 0.0 and hasattr(self, '_depth_probe_z_hat'):
    losses['loss/depth_probe/train'] = depth_probe_loss_weight * loss_depth_module(z_hat, gt_depth)
```

Code at lines 359–361: Exactly this pattern. Loss key is `'loss/depth_probe/train'`. Uses `self.loss_depth_module` — reusing existing module, not creating a new one. Guard conditions correct.

### `config.py` — head kwargs

Design requires: `depth_gate_type='gaussian_learnable_sigma'`, `depth_probe_loss_weight=0.1` as literals.

Config at lines 162–163: Both present as str/float literals. All other config sections match baseline.

---

## Invariant File Check

- `pelvis_utils.py`: identical to baseline (diff = empty)
- `train.py`: identical to baseline
- No changes to invariant infra files

---

## Output Shape Check

`forward()` returns `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)` — unchanged from baseline.

---

## Test Output Check

SLURM log (`slurm_test_55859005.out`):
- Training ran successfully to completion ("Done training!")
- Epoch 1 loss line: `loss: 3.201260  loss/joints/train: 0.201257  loss/depth/train: 2.561794  loss/uv/train: 0.154650  loss/depth_probe/train: 0.283559  grad_norm: 7.983410`
- Four loss keys present: the three baseline keys plus `loss/depth_probe/train` — exactly as designed
- `grad_norm: 7.983410` (finite) — the auxiliary loss on `z_hat` provides a direct gradient signal; this explains the finite grad norm vs. Design 001's `inf` at step 50
- `loss/depth_probe/train: 0.283559` is non-zero and finite — probe is active and receiving gradients
- `iter_metrics.csv`: 72 rows, all three tracked columns populated. The `loss/depth_probe/train` key is intentionally not tracked by `MetricsCSVHook` (which only logs the three fixed columns) — this is expected behavior.
- Checkpoint saved at epoch 1. Training started and ended cleanly.

---

## Issues Found

None. The implementation faithfully matches every required detail in design.md.
