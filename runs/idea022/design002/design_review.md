**APPROVED**

**Design:** idea022/design002 — 2-layer cascaded decoder with dynamic Gaussian reprojection bias (fixed σ=4, γ=2) and auxiliary body-joint loss (weight=0.4) on layer-1 output.

**Verdict:** APPROVED

---

## Review Summary

The design is complete, explicit, and implementation-ready. It correctly extends design001 with auxiliary supervision. All key differences from design001 are precisely specified.

### Feasibility and Completeness

All three allowed files are addressed. The design explicitly delineates what is identical to design001 and what changes:
- **`pelvis_utils.py`**: Identical `project_joints_to_feat_grid` helper — Builder instructed to skip if already present from design001.
- **`pose3d_transformer_head.py`**: Structural changes from design001 are inherited. Two precise differences in `loss()` are specified with exact before/after code.
- **`config.py`**: Exact kwargs with only `aux_loss_weight=0.4` differing from design001. All literal values.

### Key Difference 1: Gradient-enabled intermediate forward

Design001 wraps the intermediate layer-0 forward in `torch.no_grad()`. Design002 removes this wrapper, enabling gradient flow from the auxiliary loss back through layer-1. The design explains this clearly and provides exact code replacement. Correct.

**Double forward pass consequence**: Layer-0 is run twice per training step with autograd enabled — once in the bias construction block (for auxiliary loss), once in `self.forward(feats)` (for the main loss). Both passes accumulate gradients in layer-0 parameters. This is the intended behavior and is explicitly designed.

### Key Difference 2: Auxiliary joint loss

The design provides exact code for the auxiliary loss:
- Loss key: `'loss/joints_aux/train'` (distinct from `'loss/joints/train'`).
- Uses the same `loss_joints_module` (SoftWeightSmoothL1Loss, beta=0.05).
- Restricted to body joints `[:, _BODY]` where `_BODY = list(range(0, 22))`.
- Weighted by `self.aux_loss_weight = 0.4`.
- No intermediate depth or UV loss on layer-1 (to avoid pelvis degradation).
- Conditional on `self.aux_loss_weight > 0.0 and layer1_joints is not None`.

The design provides a clean scoping pattern: initialize `layer1_joints = None` before the `if self.use_reproj_bias` block and set it inside, making it accessible for the auxiliary loss below the main loss computation.

### Architecture Correctness

All points from design001 apply:
- `attn_mask` shape `(B*nheads, J, H'W')` — correct.
- `_reproj_bias` cleared to `None` at end of `forward()` — correct.
- `recover_pelvis_3d` return shape and broadcasting — correct.
- AMP `.to(q.dtype)` cast — correct.
- Feature grid orientation — correct.

### Invariant Preservation

Same as design001 — no invariant files modified. Loss restricted to body joints. `persistent_workers=False` unchanged. No import statements in config.

### Device Consistency

The design explicitly confirms `layer1_joints` and `gt_joints` are on the same device (both derived from the same feature tensor path and `gt_joints` is moved to `pred['joints'].device`). Correct.
