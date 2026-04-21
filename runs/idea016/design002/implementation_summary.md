**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `film_pool_type` and `film_hidden_dim` constructor args; builds `self.film_net` (a 2-layer MLP: `512→128→512`) with zero-initialised output layer when `film_pool_type='avg_max'`; in `forward()`, inserts a FiLM block after `spatial + pos_enc` that computes both global average pool and global max pool over 960 tokens, concatenates them into a `(B, 512)` context, produces per-channel `(gamma, beta)` via the MLP, and applies residual-scaled FiLM modulation to the spatial tokens before cross-attention.
- `code/config.py`: Added `film_pool_type='avg_max'` and `film_hidden_dim=128` to the `head` dict to enable the dual-pool FiLM conditioning.
