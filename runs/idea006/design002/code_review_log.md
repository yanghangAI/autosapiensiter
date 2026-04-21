# Code Review Log — idea006 / design002

## Entry: 2026-04-16

**Verdict: APPROVED**

All implementation changes match the design spec exactly. `_build_skeleton_attn_bias` placed at module level with correct edge list (21 body + 23 left hand + 23 right hand + 2 face), bidirectional, pelvis diagonal -0.5. `_DecoderLayer.__init__` accepts `attn_bias_init` and applies `.float().clone()`. `Pose3dTransformerHead.__init__` accepts `attn_bias_type='none'` with proper dispatch for `'skeleton_init'`, `'zero_init'`, and `'none'` paths. `config.py` adds `attn_bias_type='skeleton_init'` as string literal. Invariant files unmodified. Test run completed cleanly with valid metric output.
