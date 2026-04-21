# Design Review Log — idea009/design003

## Entry 1 — 2026-04-16

**Verdict: APPROVED**

Structured Spatial Token Dropout with Linear Annealing (p=0.30 → 0.10). All changes confined to `pose3d_transformer_head.py` and `config.py`. Full `SpatialDropAnnealHook` class body specified with correct epoch indexing, DDP-safe model access, and registration via `@HOOKS.register_module()`. Config specifies both `spatial_drop_prob_start`/`end` in head kwargs and hook in `custom_hooks`. All constraints and edge cases covered. No invariant violations. Implementation-ready.
