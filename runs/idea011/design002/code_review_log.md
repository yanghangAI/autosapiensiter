# Code Review Log — idea011/design002

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Same two-pass shared-weight architecture as Design 001 with intermediate
supervision enabled. `pose3d_transformer_head.py` is bit-identical to
Design 001's (the conditional `loss/joints_init/train` branch already
supports both weights and only activates when the config sets the weight
to non-zero). `config.py` sets
`intermediate_supervision_weight=0.5` (the only difference from
Design 001), all three new kwargs are int/bool/float literals.
`pelvis_utils.py` and `train.py` are bit-identical to baseline.
`review-check-implementation` passed. The reduced test-train ran without
exceptions and the MMEngine training log confirms the auxiliary loss is
active: `loss/joints_init/train: 0.092674` appears alongside the main
loss terms in the `Epoch(train) [1][50/81]` summary, with a magnitude
consistent with `0.5 × body-joint SoftWeightSmoothL1` at early training.
