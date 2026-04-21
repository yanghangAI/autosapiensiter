**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Note:** The `setup-design` CLI placed files in `code/code/` (extra nesting); files were moved up to `code/` and `output_dir` corrected to `…/design003` (no `/code` suffix), matching baseline convention.

**Changes:**

`code/pose3d_transformer_head.py`: Added `num_body_queries: int = 22` and `hand_aux_loss_weight: float = 0.3` kwargs to `__init__`; changed joint query embedding to 22 entries; added `self.hand_proj = nn.Sequential(Linear(num_body_queries * hidden_dim, hidden_dim), GELU(), Linear(hidden_dim, num_hand * 3))` — a 2-layer bottleneck MLP (`Linear(5632, 256) → GELU → Linear(256, 144)`) for nonlinear hand recovery from flattened body features; each Linear layer in the MLP is initialised with `trunc_normal_(std=0.02)` weight and zero bias. In `forward()`, after decoding 22 body queries, body features are flattened, passed through the MLP to `(B, 48, 3)`, then concatenated with body joints to yield `(B, 70, 3)`. In `loss()`, added auxiliary hand loss `loss/hand_aux/train = 0.3 * loss_joints_module(pred_joints[:, 22:70], gt_joints[:, 22:70])` using the existing `loss_joints_module`; the stronger weight (0.3 vs. 0.1 in design002) provides richer gradient regularisation to the body decoder via the MLP.

`code/config.py`: Added `num_body_queries=22` and `hand_aux_loss_weight=0.3` as literal values to `model.head` dict. All other config values are identical to baseline.
