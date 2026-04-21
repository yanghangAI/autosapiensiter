**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `film_pool_type` and `film_hidden_dim` constructor args; in the constructor body, builds `self.film_net` (a 2-layer MLP: `256→128→512`) with zero-initialised output layer when `film_pool_type='avg'`; in `forward()`, inserts a FiLM modulation block after `spatial + pos_enc` that computes global average pool over all 960 tokens, produces per-channel `(gamma, beta)` via the MLP, applies residual scaling (`gamma + 1`) and shifts the spatial tokens before cross-attention.
- `code/config.py`: Added `film_pool_type='avg'` and `film_hidden_dim=128` to the `head` dict to enable the global average-pool FiLM conditioning.
