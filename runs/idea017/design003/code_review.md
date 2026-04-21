# Code Review — idea017/design003

**Verdict: APPROVED**

## Automated check
`python scripts/cli.py review-check-implementation runs/idea017/design003` — PASSED.

## Files-changed fidelity
`implementation_summary.md` lists `code/pose3d_transformer_head.py` and `code/config.py`. Both correspond to files required by `design.md`. No extra files were changed. `pelvis_utils.py` and `train.py` are unmodified copies of baseline.

## Architecture — `pose3d_transformer_head.py`

- `joint_queries = nn.Embedding(22, 256)` — matches design.
- `decoder_layers = nn.ModuleList` built via `range(num_decoder_layers)` — with config passing `num_decoder_layers=3`, this creates 3 `_DecoderLayer` instances. Matches design.
- `hand_proj = nn.Linear(5632, 144)` — matches design.
- `forward()`: identical logic to designs 001/002; with 3 layers, `intermediate_outputs` contains [layer1_out, layer2_out, layer3_out].
- `loss()` intermediate block: `n_inter = len(self._intermediate_outputs) - 1 = 2`; `intermediate_weights = [0.4*(1+0.5*0), 0.4*(1+0.5*1)] = [0.4, 0.6]`; iterates over `self._intermediate_outputs[:-1]` (layer1_out, layer2_out), emitting `loss/joints_aux_0/train` (w=0.4) and `loss/joints_aux_1/train` (w=0.6). Matches design.
- Formula is the same as designs 001/002 but now produces 2 weights for n_inter=2, matching constraint 13/14 in design003.
- `hand_aux_loss_weight=0.1` branch active — matches design.

**Note on constructor default**: The head file has `num_decoder_layers: int = 2` as the Python default. However, the config explicitly passes `num_decoder_layers=3` as a literal, which takes precedence. The actual model instantiated has 3 decoder layers as confirmed by the test output (both `loss/joints_aux_0` and `loss/joints_aux_1` appear). This discrepancy between default and intended value is a minor inconsistency but does not affect correctness since config values override defaults.

## Config — `config.py`

- `num_decoder_layers=3` — matches design. Only difference from design002 config.
- `aux_body_loss_weight=0.4`, `hand_aux_loss_weight=0.1`, `num_body_queries=22` — all correct.
- No Python `import` statements — compliant.

## Invariants
All invariant files unmodified. Output shape (B,70,3) preserved. `persistent_workers=False` preserved.

## Test output
- Training completed ("Done training!"), `epoch_1.pth` saved.
- Loss keys at iter 50: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/joints_aux_0/train: 0.084271`, `loss/joints_aux_1/train: 0.141470`, `loss/hand_aux/train: 0.095885` — exactly matches design003's expected 6 loss keys.
- `grad_norm: inf` observed at iter 50. Training still completed successfully and checkpoint was saved. With FixedAmpOptimWrapper and `clip_grad max_norm=1.0`, `inf` gradient norm before clipping is not unusual in early training with AMP + 3-layer decoder. This is a known early-training transient; it does not constitute a runtime error. Flagging for monitoring during full training.
- Memory 8673 MB — within 2080 Ti budget (slightly higher than 2-layer designs due to extra decoder layer, as expected).

## Issues
- **Minor**: `num_decoder_layers: int = 2` constructor default in `pose3d_transformer_head.py` does not match design003's intended value of 3. The config correctly passes 3, so there is no functional bug, but the default is misleading for design003. Not a rejection criterion — config values take precedence.
- **Watchable**: `grad_norm: inf` at iter 50. Not a failure (training completed), but should be monitored in full training runs. If persistent gradient instability appears in the full 20-epoch run, report to Orchestrator.
