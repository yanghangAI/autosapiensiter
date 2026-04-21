**Verdict: APPROVED**

**Design:** design003 — Stack 4 decoder layers with intermediate supervision + shared output projection
**Idea:** idea001 — Multi-Layer Decoder with Intermediate Supervision
**Reviewer date:** 2026-04-16

---

## Review Summary

The implementation correctly realises all design requirements. The 4-layer decoder with a single shared `joints_out` projection used at every layer (including intermediates) is fully implemented. The test ran to completion with no errors, and training logs confirm all 3 auxiliary loss keys are present.

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea001/design003` — PASSED.

---

## Files Changed

`implementation_summary.md` lists `code/pose3d_transformer_head.py` and `code/config.py`. Both are permitted by `design.md`. No additional files modified. No invariant files touched.

---

## Code vs. Design Checklist

| Design requirement | Met? | Notes |
|---|---|---|
| `num_decoder_layers: int = 4` constructor param | Yes | Line 153 |
| `aux_loss_weight: float = 0.4` constructor param | Yes | Line 154 |
| `self.num_decoder_layers` and `self.aux_loss_weight` stored | Yes | Lines 173–174 |
| `self.decoder_layers = nn.ModuleList([...])` of 4 independent layers | Yes | Lines 189–192 |
| Single shared `self.joints_out = nn.Linear(hidden_dim, 3)` | Yes | Line 195 — one definition, called multiple times |
| `depth_out` and `uv_out` defined once (final layer only) | Yes | Lines 196–197 |
| `forward` collects `layer_outputs` from all 4 layers | Yes | Lines 257–260 |
| `intermediate_joints` key in return dict (length = num_layers - 1 = 3) | Yes | Lines 274–276 |
| All `intermediate_joints` projected via same shared `joints_out` | Yes | Line 275 calls `self.joints_out(h)` for each h |
| Pelvis computed from final layer only | Yes | Lines 266–268 |
| Final layer losses: `loss/joints/train`, `loss/depth/train`, `loss/uv/train` | Yes | Lines 317–322 |
| Aux losses: `loss/joints_aux0/train`, `loss/joints_aux1/train`, `loss/joints_aux2/train` at weight 0.4 | Yes | Lines 325–329 |
| No pelvis aux losses | Yes | Confirmed — pelvis only in final layer block |
| Body joint loss restricted to indices 0–21 (all joint losses) | Yes | `_BODY = list(range(0, 22))` applied to both final and aux |
| `predict()` unchanged and backward-compatible | Yes | Accesses only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']` |
| `_train_mpjpe` uses final layer only | Yes | Lines 335–341 |
| `num_decoder_layers=4`, `aux_loss_weight=0.4` in config head dict | Yes | config.py lines 138–139 |
| `persistent_workers=False` in both dataloaders | Yes | config.py lines 175, 194 |
| No Python `import` statements in config.py | Yes | Uses `__import__()` |
| Absolute imports in head file | Yes | Lines 34–36 |
| `_DecoderLayer` class unmodified — 4 independent instances | Yes | Confirmed |
| Do not create per-layer output heads | Yes | Only one `joints_out` defined |
| No changes to invariant files | Yes | Confirmed |

---

## Test Output Verification

- Training completed 1 epoch, 81 iterations, no errors or OOM.
- Training log at iter 50 shows loss keys: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/joints_aux0/train`, `loss/joints_aux1/train`, `loss/joints_aux2/train` — exactly the 6 expected keys for a 4-layer decoder.
- Validation ran to completion; all 6 metric columns populated in `metrics.csv`.
- `epoch_1.pth` and `metrics.csv` written as expected.
- Memory reported: 10720 MB — within 1080 Ti budget.

---

## Invariant Check

No changes to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py`. Confirmed.

---

## Decision

**APPROVED.**
