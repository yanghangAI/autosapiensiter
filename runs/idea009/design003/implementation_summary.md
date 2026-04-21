**Files changed:**
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pose3d_transformer_head.py`: Added `from mmengine.registry import HOOKS` and `from mmengine.hooks import Hook` imports. Added `spatial_drop_prob: float = 0.0` argument to `_DecoderLayer.forward` with fresh `key_padding_mask` generation during training. Changed `Pose3dTransformerHead.__init__` to accept `spatial_drop_prob_start: float = 0.30` and `spatial_drop_prob_end: float = 0.10`; stores both plus initialises `self.spatial_drop_prob = spatial_drop_prob_start` so epoch 1 starts at p=0.30. Added `set_drop_prob(p)` method for the hook to call. Updated `decoder_layer` call in `forward` to pass `spatial_drop_prob=self.spatial_drop_prob`. Added `SpatialDropAnnealHook` class (registered via `@HOOKS.register_module()`) that linearly interpolates p from 0.30 to 0.10 over 20 epochs, called via `before_train_epoch` with defensive DDP unwrap.

`config.py`: Added `spatial_drop_prob_start=0.30` and `spatial_drop_prob_end=0.10` to head kwargs; added `dict(type='SpatialDropAnnealHook', num_epochs=20, start_prob=0.30, end_prob=0.10)` to `custom_hooks`.
