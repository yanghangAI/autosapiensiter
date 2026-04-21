**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same additions as design001 — six new kwargs in `__init__`, buffer registration for `sym_pairs_buf` and `sym_pair_weights_buf`, and bilateral symmetry loss block in `loss()`; the `sym_pair_weights_buf is not None` branch is active in this design, multiplying per-pair weights `(1, P, 1)` against the `(B, P, 3)` smooth-L1 loss tensor before the mean.

`code/config.py`: Added `bilateral_sym_loss_weight=0.5`, `sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]`, `sym_mirror_axis=1`, and `sym_pair_weights=[0.5, 1.0, 2.0, 2.0, 0.5, 1.0, 1.5, 2.0]` to the `head` dict; per-pair weights upweight distal joints (ankle=2.0, foot=2.0, wrist=2.0) 4× over proximal joints (hip=0.5, collar=0.5).
