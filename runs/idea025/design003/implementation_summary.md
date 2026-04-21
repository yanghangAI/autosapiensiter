**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same additions as design001/002 — six new kwargs in `__init__`, buffer registration (no `sym_pair_weights_buf` for this design), and bilateral symmetry loss block in `loss()`; the `sym_adaptive_weight=True` branch computes per-sample per-pair soft weights `asym_w = 1 / (1 + ||asym_gt||_2 / tau)` under `torch.no_grad()` and multiplies them against the smooth-L1 tensor before the mean, reducing the penalty for genuinely asymmetric poses.

`code/config.py`: Added `bilateral_sym_loss_weight=0.5`, `sym_pairs=[[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21]]`, `sym_mirror_axis=1`, `sym_adaptive_weight=True`, and `sym_tau=0.1` to the `head` dict; `sym_tau=0.1` metres means poses with GT asymmetry magnitude ≥ 100 mm receive weight ≤ 0.5.
