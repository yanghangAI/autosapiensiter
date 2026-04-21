# Code Review Log — idea011/design003

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Two-pass decoder with INDEPENDENT pass-2 weights
(`self.decoder_layer_2 = _DecoderLayer(hidden_dim, num_heads, dropout)`)
and intermediate supervision is faithfully implemented.
`pose3d_transformer_head.py` adds the three new kwargs with correct
defaults, builds `self.coord_enc` with zero-init on the final Linear,
conditionally builds `self.decoder_layer_2` gated by `(not
self.shared_decoder) and self.num_refine_passes >= 2`, and routes pass 2
through it via the `else` branch in `forward()`. `loss()` emits the
`loss/joints_init/train` auxiliary term with weight 0.5. `config.py`
sets `num_refine_passes=2`, `shared_decoder=False`,
`intermediate_supervision_weight=0.5`; all int/bool/float literals.
`pelvis_utils.py` and `train.py` are bit-identical to baseline.
`review-check-implementation` passed. The reduced test-train ran without
exceptions; the MMEngine init summary contains
`head.decoder_layer_2.self_attn.in_proj_weight` and all other
`decoder_layer_2` parameters, confirming the new layer is registered.
The Epoch(train) log line shows `loss/joints_init/train: 0.104254`
alongside the main `loss/joints/train`, confirming the auxiliary
supervision is active. Training losses are healthy and decreasing over
66 iterations.
