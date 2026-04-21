# Design Review Log — idea011/design003

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Two-pass coordinate-conditioned decoder with INDEPENDENT pass-2 weights (new `self.decoder_layer_2 = _DecoderLayer(256, 8, 0.1)`) and intermediate supervision on pass-1 body joints (weight=0.5). `decoder_layer_2` is built conditionally only when `shared_decoder=False` and `num_refine_passes >= 2`, and routed via the existing `else` branch in `forward()`. `joints_out` remains shared; zero-init `coord_enc[2]` preserves coordinate-feedback no-op at init (with the caveat that `decoded_2 != decoded_1` due to fresh pass-2 weights — residual stays small because `joints_out` is std=0.02). Config adds `num_refine_passes=2`, `shared_decoder=False`, `intermediate_supervision_weight=0.5`. Only `pose3d_transformer_head.py` and `config.py` touched; `pelvis_utils.py` untouched. Optimizer `paramwise_cfg` unchanged — new params under default head LR 1e-4. All invariants preserved. Implementation-ready.
