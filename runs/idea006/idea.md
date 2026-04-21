**Idea Name:** Skeleton-Guided Self-Attention via Learnable Query Bias

**Approach:** Add a learnable additive attention bias matrix of shape `(num_joints, num_joints)` to the joint query self-attention in the decoder, initialized to zero (recovering baseline behaviour), so the model can learn which joints should attend to which others based on the body's kinematic structure rather than discovering it purely from data.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder's single self-attention layer over the 70 joint queries uses standard scaled dot-product attention with no structural prior: every joint query starts with equal "access" to every other joint query. The model must therefore learn the kinematic body structure (e.g. wrist attends to elbow, shoulder, etc.) entirely from gradient signal over training data, using the same fully-connected attention pattern for a hand joint as for a torso joint.

Human body kinematics impose a sparse, hierarchical structure on how joints relate to one another:
1. **Kinematic chains**: adjacent joints in a limb are strongly coupled (elbow constrains wrist range).
2. **Body symmetry**: left-right homologous joints are highly correlated.
3. **Root-relative structure**: all joints are expressed relative to the pelvis root, so the pelvis query (token 0) is special relative to all others.

Evidence from the experiment log motivates this directly:

- **idea001** (multi-layer decoder): stacking layers improved body MPJPE by 9–14 mm but hurt pelvis MPJPE by 5–17 mm. The self-attention across layers is believed to be the mechanism that gradually specialises joints for body-structure reasoning — but apparently this also makes token 0 less suitable as a pelvis depth/UV predictor (it gets "pulled" into body-joint territory by unconstrained self-attention).
- **idea005** (uncertainty weighting): targets loss balance but leaves the attention mechanism unchanged.
- **idea004** (depth positional encoding): targets spatial token side of cross-attention; does not affect query self-attention.
- **idea003** (content-adaptive queries): conditions initial query embeddings on image content but still passes them through the same unconstrained self-attention.

None of the prior ideas have modified the *self-attention connectivity* in the query domain. A learnable structural bias does this efficiently and orthogonally.

### What this idea adds

In the `_DecoderLayer.forward()` method, the self-attention call is:

```python
q2 = self.self_attn(q, q, q)[0]
```

`nn.MultiheadAttention` accepts an `attn_mask` argument that is added (before softmax) to the attention logits. We register a learnable parameter `attn_bias` of shape `(num_joints, num_joints)` and pass it as the `attn_mask` to the self-attention:

```python
q2 = self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]
```

Initialised to zeros, `attn_bias` has no effect at the start of training — the model begins from exactly the same state as the baseline. Over training, gradient updates will push positive entries toward pairs of joints that benefit from attending to each other and negative entries toward pairs that should be suppressed. The matrix is learned end-to-end and requires no handcrafted skeleton graph.

This is parameter-efficient: `num_joints × num_joints = 70 × 70 = 4900` scalar parameters (~20 KB). No extra attention computation is added — it is a single element-wise addition to the existing attention logits.

**Implementation note for multi-head attention:** PyTorch's `nn.MultiheadAttention` with `attn_mask` of shape `(T_q, T_k)` broadcasts it across the batch and head dimensions. This means a single `(num_joints, num_joints)` matrix is used for all heads — a *shared* bias. An alternative is `(num_heads * batch, num_joints, num_joints)` for per-head biases, but the shared bias is simpler, more regularised, and sufficient for this idea.

---

## Proposed Variations

### Design A — Shared learnable attention bias, full query set (baseline variant)

Add a single `nn.Parameter` of shape `(num_joints, num_joints)` initialised to zero, passed as `attn_mask` to the self-attention. This is the minimal-change design: the decoder layer is unchanged except for the additive bias. Tests whether any learned query-domain structure improves over unconstrained self-attention. Changes: ~5 lines in `pose3d_transformer_head.py`.

### Design B — Shared attention bias with skeleton-graph-informed initialisation

Same as Design A but initialise the bias matrix using a soft skeleton-graph adjacency prior instead of zeros:
- Joints connected directly in the BEDLAM2 skeleton graph (parent–child edges) are initialized to `+0.5`.
- Non-adjacent joints are initialized to `0.0`.
- Pelvis query (token 0) diagonal entry is initialized to `-0.5` (soft self-suppression to reduce the specialisation pressure from body joints on token 0).

The skeleton graph for BEDLAM2's 70-joint set is hardcoded as a list of `(i, j)` edge pairs in `pose3d_transformer_head.py` (no imports needed). This design tests whether a graph-informed warm-start converges faster within the 20-epoch budget than learning the structure from scratch (Design A).

**Skeleton edges to hardcode (BEDLAM2 body + hand joint indices 0-69):** The body joints (0–21) follow the standard SMPL-X kinematic tree. The Designer should hardcode the 20 body kinematic edges (e.g. `[(0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,12), (12,15), (9,13), (13,16), (16,18), (18,20), (9,14), (14,17), (17,19), (19,21)]`) from the SMPL-X definition. Hand joints (22–69) can use the per-finger chain structure.

### Design C — Per-head attention bias (richer but still efficient)

Instead of a single shared bias matrix, learn `num_heads` independent bias matrices, each of shape `(num_joints, num_joints)`, all initialised to zero. Pass the appropriate per-head bias by expanding to `(B * num_heads, num_joints, num_joints)` at forward time.

This gives each attention head the freedom to specialise differently: one head might learn left-right symmetry biases, another might learn kinematic chain biases, another might learn to route the pelvis token (0) away from body-joint interactions. Parameter cost: `8 × 70 × 70 = 39200` scalars (~157 KB). Still negligible vs. the model size.

**Implementation note:** In PyTorch's `nn.MultiheadAttention`, when `batch_first=True` and `attn_mask` is `(B*num_heads, T, T)`, the mask is applied per-head. The head must expand `self.attn_bias` from `(num_heads, J, J)` to `(B * num_heads, J, J)` using `.repeat(B, 1, 1)` before passing it as `attn_mask`. This requires passing `B` (batch size) into the decoder layer's forward method — a minor refactor.

---

## Implementation Scope

All changes are confined to `pose3d_transformer_head.py`:

1. **`_DecoderLayer.__init__`**: Accept optional `attn_bias_init: Optional[torch.Tensor]` and `num_heads_for_bias: int = 1`. Register `self.attn_bias = nn.Parameter(attn_bias_init or torch.zeros(num_joints, num_joints))`.
   - Design A: `torch.zeros(num_joints, num_joints)`
   - Design B: pre-filled tensor from hardcoded edge list
   - Design C: `torch.zeros(num_heads, num_joints, num_joints)`

2. **`_DecoderLayer.forward`**: Pass `attn_mask=self.attn_bias` to `self.self_attn(...)`.
   - Design C: expand to `(B * num_heads, J, J)` before passing.

3. **`Pose3dTransformerHead.__init__`**: Accept `attn_bias_type: str = 'none'` as a config kwarg (`'zero_init'`, `'skeleton_init'`, `'per_head'`). Construct `_DecoderLayer` with the appropriate `attn_bias_init`.

4. **`config.py`**: Add `attn_bias_type` to head kwargs as a string literal.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, data pipeline, backbone, or training infrastructure.

---

## Expected Outcome

- **Primary gain**: improved body MPJPE through better inter-joint information routing during self-attention. Adjacent joints (e.g. elbow ↔ wrist) will accumulate positive bias, reinforcing kinematically consistent pose decoding.
- **Secondary gain**: potential improvement in pelvis MPJPE if the skeleton bias suppresses cross-contamination from body joint queries into pelvis token (0). Design B explicitly initialises a negative self-suppression bias on token 0 to test this.
- **Design A**: diagnostic baseline — does learned query structure help at all? Expected: −5 to −10 mm body MPJPE.
- **Design B**: expected to converge faster due to warm-start from skeleton prior. May recover both body and pelvis gains simultaneously if pelvis token suppression works.
- **Design C**: richest variant; each head specialises independently. Highest ceiling but may need more epochs to converge.
- **Composite target**: aim for composite_val < 160 (vs. baseline 169.75, idea001 best 162.00).

---

## Risk and Mitigation

- **attn_mask sign convention in PyTorch**: `nn.MultiheadAttention` treats `attn_mask` as an *additive* bias to the attention logits *before* softmax. Positive values increase attention weight; large negative values (e.g., `float('-inf')`) mask tokens. The Designer should verify this convention and ensure the bias is passed correctly. Since we initialise to zero, any sign error will manifest as no change from baseline, not as a crash.
- **Multi-head broadcast**: the `(J, J)` mask is broadcast to all heads and all batches. For Design C, the Designer must expand correctly to `(B * num_heads, J, J)`. This requires passing `B` into `_DecoderLayer.forward` — a minor refactor of the decoder layer signature.
- **Interaction with idea001 (multi-layer decoder)**: if combined, the attn_bias parameter is shared (or independent) across decoder layers. The Designer should use independent biases per layer if stacking, to allow specialisation per refinement stage.
- **Memory**: 4900 to 39200 extra float32 scalars. Negligible.
- **Training time**: single element-wise addition to attention logits. No measurable overhead.
- **MMEngine config constraint**: `attn_bias_type` is a simple string literal in config.py. The skeleton edge list is hardcoded in `pose3d_transformer_head.py` as a Python list. No imports in config.py required. Fully compliant.
