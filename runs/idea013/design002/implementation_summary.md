**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py` — Identical head implementation to design001: new kwargs (`kinematic_parametrization`, `bone_parents`, `bone_length_loss_weight`, `per_limb_heads`, `limb_index`), `_forward_kinematics` cumulative-sum recovery along the SMPL-X parent chain, `1/sqrt(21)` weight scale-init under `torch.no_grad()`, and the `forward()` kinematic-recovery block. The auxiliary bone-length loss block is the same as design001's implementation, but in this design it is active because `bone_length_loss_weight=0.3 > 0.0`. The block computes predicted and GT bone vectors from the recovered joint positions (`pred_bones = pred_body[:, child_idx] - pred_body[:, parent_idx]` and analogously for GT), takes their `.norm(dim=-1)` magnitudes, and adds `bone_length_loss_weight * (pred_bone_len - gt_bone_len).abs().mean()` as the `'loss/bone_length/train'` key, matching the project's `loss/<name>/<split>` convention for `MetricsCSVHook` pickup.
- `code/config.py` — Same five head kwargs as design001 except `bone_length_loss_weight=0.3` (active auxiliary bone-length prior with weight 0.3, as per the design spec). All values are plain Python literals; no imports introduced.
