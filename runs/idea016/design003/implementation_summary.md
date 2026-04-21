**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `film_pool_type`, `film_hidden_dim`, and `film_num_blocks` constructor args; builds `self.film_net` (a shared 2-layer MLP: `256→128→512`) with zero-initialised output layer when `film_pool_type='spatial_block'`; in `forward()`, inserts a hierarchical FiLM block after `spatial + pos_enc` that reshapes the 960 spatial tokens into a 40×24 grid, partitions it into 16 blocks (4×4 layout, each 10×6 tokens), computes per-block average-pool contexts, applies the shared MLP to each block to produce block-specific `(gamma, beta)`, expands and scatters back to per-token scale/shift, and applies FiLM modulation before cross-attention. **Shape fix (post-review):** After `mean(dim=(2,4))`, `gamma_b` is shape `(B, 4, 4, D)` (4 dims). A single `unsqueeze(2)` produced a 5-dim tensor `(B, 4, 1, 4, D)` which could not be expanded to the 6-dim target `(B, 4, H//4, 4, W//4, D)`. Fixed by adding a second `unsqueeze(4)` before `expand` to insert the missing W-block singleton dimension.
- `code/config.py`: Added `film_pool_type='spatial_block'`, `film_hidden_dim=128`, and `film_num_blocks=16` to the `head` dict to enable hierarchical spatial-block FiLM conditioning.
