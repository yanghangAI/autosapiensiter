**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added six new kwargs to `Pose3dTransformerHead.__init__` (`bilateral_sym_loss_weight`, `sym_pairs`, `sym_mirror_axis`, `sym_pair_weights`, `sym_adaptive_weight`, `sym_tau`) with defaults that reproduce baseline behaviour; registered `sym_pairs_buf` and (conditionally) `sym_pair_weights_buf` as buffers; appended a bilateral symmetry consistency loss block in `loss()` after `losses['loss/uv/train']` that computes smooth-L1 on the predicted vs GT asymmetry vectors (left joint minus Y-mirrored right joint) over 8 verified symmetric body pairs and logs the result as `loss/sym/train`.

`code/config.py`: Added `bilateral_sym_loss_weight=0.3`, `sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]`, and `sym_mirror_axis=1` to the `head` dict; all values are literals, no Python imports, fully MMEngine-compliant.
