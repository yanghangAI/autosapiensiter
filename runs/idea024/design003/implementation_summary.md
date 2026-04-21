**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added module-level `_UPPER_IDX` (indices 0-12) and `_LOWER_IDX` (indices 13-21) constants; added 7 new constructor parameters (`per_joint_difficulty_weighting`, `ema_alpha`, `ema_momentum`, `weight_norm`, `weight_temperature`, `group_normalise`, `ema_warmup_epochs`); registered separate `upper_err_ema` (ones(13)) and `lower_err_ema` (ones(9)) buffers when `group_normalise=True`, or a single `joint_err_ema` (ones(22)) otherwise, plus `_train_iter`; added `_get_adaptive_weights()` method with group-normalised per-body-region weighting (alpha=1.0 linear within each group) and a linear warmup ramp that blends from uniform weights to difficulty-weighted over `ema_warmup_epochs * 328` iterations; replaced the body joint loss line with a conditional block updating the group EMAs and applying per-joint weights via manual smooth-L1.
- `code/config.py`: Added six literal kwargs to `model.head` dict: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='linear'`, `group_normalise=True`, `ema_warmup_epochs=5` to enable group-normalised per-joint difficulty weighting with 5-epoch warmup ramp.
