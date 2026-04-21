**Idea Name:** Multi-Layer Decoder with Intermediate Supervision

**Approach:** Stack multiple transformer decoder layers (2–4) in the pose head and add auxiliary joint losses at each intermediate layer so that gradient signal flows to early layers and attention maps specialize progressively rather than being learned in a single pass.

**Expected Designs:** 3

**Baseline Source:** baseline/

---

## Motivation

The baseline decoder is a single `_DecoderLayer` (self-attn → cross-attn → FFN).  With only one layer, the joint queries must simultaneously learn to attend to the correct spatial regions *and* regress accurate 3D coordinates.  This is a hard joint optimisation.

In DETR and related detection/pose work, stacking decoder layers with shared or independent weights consistently improves performance because:

1. **Progressive refinement**: early layers establish coarse spatial attention; later layers refine within-body structure.
2. **Intermediate supervision**: auxiliary losses attached to each layer's output force every layer to produce a valid pose estimate, preventing gradient vanishing and promoting feature reuse.
3. **Body-structure inductive bias**: across multiple self-attention passes, joints learn to attend to their kinematic neighbours (e.g., elbow queries pull from shoulder queries), improving relative pose accuracy without hardcoding a skeleton graph.

## Proposed Variations

**Design A — Stacked layers, no aux loss (2 layers)**
Add a second decoder layer on top of the first.  No auxiliary loss — just deeper decoding.  This is the cheapest ablation, testing whether capacity alone helps.

**Design B — 3 layers + intermediate supervision**
Stack 3 decoder layers and attach an auxiliary joint-coordinate loss at the output of each intermediate layer (weighted lower than the final layer, e.g. 0.4 × aux vs. 1.0 × final).  Pelvis depth/UV losses only on the final layer to avoid noisy gradient from early layers on the harder absolute task.

**Design C — 4 layers + intermediate supervision + shared output head**
Same as Design B but with 4 layers and a *shared* output projection across all layers (parameter-efficient). The shared head forces a common pose-space representation to emerge across refinement stages, acting as a regulariser.

## Implementation Scope

Changes are localised to `pose3d_transformer_head.py`:
- `_DecoderLayer` already exists; just instantiate `nn.ModuleList` of N layers.
- Loop over layers in `forward()`, collecting intermediate outputs.
- In `loss()`, compute auxiliary joint losses on intermediate outputs, sum with weight coefficients.
- `config.py` exposes `num_decoder_layers` and `aux_loss_weight` as head kwargs — no other config changes needed.

No changes to `pelvis_utils.py`, `bedlam_metric.py`, or the data pipeline.

## Expected Outcome

Intermediate supervision consistently yields 5–15 % MPJPE improvement in comparable transformer pose models.  Even Design A (pure capacity) is expected to improve over the baseline.  Design B is the primary bet; Design C tests whether parameter sharing can match the quality of Design B at lower model size.

## Risk and Mitigation

- **Memory**: extra decoder layers increase GPU memory.  On the 1080 Ti (8 GB) with batch 4, two additional layers (~2× hidden_dim=256 cross-attn) add roughly 200 MB — acceptable.  If OOM occurs, reduce `hidden_dim` to 192.
- **Training time**: stacking layers proportionally increases compute per iteration.  With hidden_dim=256, this is dominated by backbone cost — marginal impact.
- **Aux loss weight tuning**: aux weights are a hyperparameter; Designer should sweep {0.25, 0.5} for Design B.
