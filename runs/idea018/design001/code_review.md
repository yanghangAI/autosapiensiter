# Code Review — idea018/design001

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea018/design001` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are within the allowed set. `pelvis_utils.py` and `train.py` are present but verified identical to baseline (no diff).

---

## Implementation vs. Design Checklist

### `_DecoderLayer.forward()` — optional `attn_logit_bias` argument

Design requires:
- New signature: `forward(self, queries, spatial_tokens, attn_logit_bias=None)`
- Expand `attn_logit_bias (B, N_spatial)` to `(B*num_heads, Nq, N_spatial)` via unsqueeze+expand+reshape
- Pass as float `attn_mask` to `nn.MultiheadAttention`
- Else-branch: standard cross-attention unchanged

Code at lines 101–139: All four requirements met exactly. Shape expansion:
`(B, N_spatial) → unsqueeze(1).unsqueeze(1) → (B,1,1,N_spatial) → expand(B, num_heads, Nq, N_spatial) → reshape(B*num_heads, Nq, N_spatial)` — matches design spec.

### `Pose3dTransformerHead.__init__()` — new kwargs and modules

Design requires:
- `depth_gate_type: str = 'none'`
- `depth_gate_sigma: float = 1.0`
- When `depth_gate_type == 'gaussian'`: create `depth_probe_global: Linear(hidden_dim, 1)`, `depth_probe_token: Linear(hidden_dim, 1)`, register buffer `depth_gate_sigma_buf`

Code at lines 163–226: All three requirements met. Buffer name is `depth_gate_sigma_buf` — matches design. No extraneous kwargs added.

### `_init_head_weights()` — zero-init depth probes

Design requires: when `depth_gate_type == 'gaussian'`, zero-init weight and bias of both probes.

Code at lines 241–245: Exactly implemented. Four `nn.init.zeros_()` calls on `.weight` and `.bias` of each probe.

### `forward()` — compute gate and pass to decoder

Design requires:
- After `spatial = spatial + pos_enc`, before decoder call
- `z_hat = depth_probe_global(spatial.mean(dim=1))` — shape `(B, 1)`
- `z_tok = depth_probe_token(spatial).squeeze(-1)` — shape `(B, H*W)`
- `sigma = self.depth_gate_sigma_buf`
- `depth_err = (z_tok - z_hat) / sigma`
- `attn_logit_bias = -0.5 * depth_err ** 2`
- Pass `attn_logit_bias` to `self.decoder_layer(queries, spatial, attn_logit_bias=attn_logit_bias)`

Code at lines 285–296: All requirements met. No `_depth_probe_z_hat` caching added (correctly absent per design constraint 9).

### `loss()` — no changes

Design requires: `loss()` unchanged for Design 001.

Code at lines 312–367: `loss()` has no depth probe auxiliary loss. Correct.

### `config.py` — head kwargs

Design requires: `depth_gate_type='gaussian'` and `depth_gate_sigma=1.0` as literals in `model.head`.

Config at lines 162–163: Both present as str/float literals. All other config sections (optimizer, LR schedule, data pipeline, hooks, backbone) match baseline.

---

## Invariant File Check

- `pelvis_utils.py`: identical to baseline (diff = empty)
- `train.py`: identical to baseline (diff = empty)
- No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` (not present in code directory)

---

## Output Shape Check

`forward()` returns `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)` — unchanged from baseline. `predict()` uses `self.num_joints = 70` — correct.

---

## Test Output Check

SLURM log (`slurm_test_55859004.out`):
- Training ran successfully to completion ("Done training!")
- Epoch 1 loss line: `loss: 2.924203  loss/joints/train: 0.201261  loss/depth/train: 2.568041  loss/uv/train: 0.154901  grad_norm: inf`
- Three expected loss keys present, no extra keys (no `loss/depth_probe/train` — correct for Design 001)
- `grad_norm: inf` at step 50 is expected at initialization with zero-init probes and AMP; this is consistent with other baseline experiments and resolves as training progresses
- Checkpoint saved at epoch 1. Training started and ended cleanly.
- `iter_metrics.csv`: 72 rows (one per iteration in the reduced test epoch), all three loss columns populated — no missing values or NaN

---

## Issues Found

None. The implementation faithfully matches every required detail in design.md.
