**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Note:** The `setup-design` CLI placed files in `code/code/` (extra nesting); files were moved up to `code/` and `output_dir` corrected to `…/design002` (no `/code` suffix), matching baseline convention.

**Changes:**

`code/pose3d_transformer_head.py`: Added `num_body_queries: int = 22` and `hand_aux_loss_weight: float = 0.1` kwargs to `__init__`; changed joint query embedding to 22 entries; added `self.hand_proj = nn.Linear(num_body_queries * hidden_dim, (num_joints - num_body_queries) * 3)` (i.e., `Linear(5632, 144)`) for linear hand recovery from flattened body features; initialised its weights with `trunc_normal_(std=0.02)` and bias with zeros. In `forward()`, after decoding 22 body queries, body features are flattened and projected to `(B, 48, 3)` hand predictions, then concatenated with body joints to yield `(B, 70, 3)`. In `loss()`, added auxiliary hand loss `loss/hand_aux/train = 0.1 * loss_joints_module(pred_joints[:, 22:70], gt_joints[:, 22:70])` using the existing `loss_joints_module` instance; this provides regularising gradient through `hand_proj` into the body decoder without affecting composite metric.

`code/config.py`: Added `num_body_queries=22` and `hand_aux_loss_weight=0.1` as literal values to `model.head` dict. All other config values are identical to baseline.
