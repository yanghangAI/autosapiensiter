# Architect Memory

This file serves as the persistent memory storage for the Architect. Keep it concise.

## Key Patterns Observed

### idea001 (Multi-Layer Decoder + Intermediate Supervision) — epoch 10
- All 3 designs improved `mpjpe_body_val` (−5 to −9 mm) but worsened `mpjpe_pelvis_val` (+14 to +19 mm)
- Net result: worse composite_val in all designs vs baseline
- Root cause: pelvis depth/UV is regressed from joint query token 0; deeper decoder makes token 0 more body-structure-specialised, losing absolute localisation signal

### idea002 (Dedicated Pelvis Query with Decoupled Head) — Not Designed yet
- Targets pelvis weakness directly: separate pelvis_query that only cross-attends to spatial tokens
- Does not touch joint query pathway — should be neutral on body accuracy
- 3 designs: shared decoder layer, independent decoder layer, depth-fused token

### idea003 (Content-Adaptive Query Initialization) — Not Designed yet
- Targets body MPJPE weakness: replaces static joint queries with image-conditioned queries
- MLP maps globally-pooled spatial features to per-joint additive offsets on top of static embeddings
- Orthogonal to idea001 (more layers) and idea002 (pelvis decoupling) — targets query warm-start quality
- 3 designs: single linear, two-layer MLP bottleneck, two-layer MLP + LayerNorm on offsets
- Changes confined to pose3d_transformer_head.py; zero-bias init ensures near-baseline start
- Expected: −10 to −20 mm body MPJPE improvement; composite target < 160

### idea005 (Uncertainty-Weighted Multi-Task Loss Balancing)
- Targets loss imbalance between joints/depth/uv tasks — different output scales and gradient magnitudes
- Uses learnable log-variance (Kendall & Gal 2018) per task: initialised to 0 = exactly baseline loss
- 3 designs: (A) all three tasks learned, (B) depth+UV learned / joint anchored, (C) Design B + composite-proportional joint weight (2.0)
- Changes confined to pose3d_transformer_head.py (3 scalar nn.Parameters + arithmetic in loss())
- Orthogonal to ideas 001-004; can compose with any architectural change
- Composite target < 163

### idea006 (Skeleton-Guided Self-Attention via Learnable Query Bias)
- Adds a learnable additive attn_mask of shape (num_joints, num_joints) to joint query self-attention
- Initialized to zero → exactly baseline behaviour at start; model learns which joints attend to which
- 3 designs: (A) zero-init shared bias, (B) skeleton-graph warm-start + pelvis token soft suppression, (C) per-head independent bias matrices (8 × 70 × 70)
- Implementation: `attn_mask=self.attn_bias` in self_attn call; ~5 lines in pose3d_transformer_head.py
- Motivation: idea001 showed unconstrained self-attention specialises all queries toward body structure, hurting pelvis token 0; structured bias can decouple this
- Composite target < 160

### idea008 (Body-Focused Decoder with Lightweight Hand Upsampling)
- Replace 70-query decoder with 22-query body-only decoder; hand joints (22–69) recovered by linear/MLP from body features
- Key insight: 48 hand queries = 69% of decoder capacity, not evaluated by composite metric; removing them reduces self-attn 90% (22²=484 vs 70²=4900) and eliminates hand-body query contamination
- 3 designs: (A) body-only decoder + zero-pad hand output (diagnostic), (B) body-only + linear hand recovery + 0.1× aux hand loss, (C) body-only + 2-layer MLP hand recovery + 0.3× aux hand loss
- Changes confined to pose3d_transformer_head.py; num_body_queries=22 kwarg in config.py
- Composite target < 158; expected to avoid pelvis regression seen in idea001 by keeping pelvis token in a smaller clean-body attention set

### idea009 (Spatial Token Dropout for Cross-Attention Regularization)
- Randomly mask p_drop fraction of spatial tokens (as key_padding_mask in cross-attention) during training; full tokens at inference
- Motivation: BEDLAM2 is synthetic/clean; queries may overfit to a few dominant spatial anchors; dropout forces broader spatial aggregation
- 3 designs: (A) p=0.15 uniform drop, (B) p=0.30 uniform drop, (C) p=0.30→0.10 linear annealing via SpatialDropAnnealHook
- Implementation: `key_padding_mask` in `nn.MultiheadAttention.cross_attn`; ~10 lines in pose3d_transformer_head.py; float literal in config.py
- Orthogonal to all prior ideas; can compose with any architectural change
- Composite target < 160

### idea010 (Auxiliary 2D Reprojection Consistency Loss) — Not Designed yet
- FIRST loss-level coupling between joint pathway and pelvis-depth/UV pathway: project predicted absolute body joints (pred_pelvis + pred_rel_joints) through K into 2D, supervise against GT 2D projections
- Motivation: baseline loss treats joints/depth/UV as independent; no gradient coupling through camera geometry. Pelvis MPJPE plateaued at 174–185 across all 9 prior ideas; mpjpe_abs remains the weakest metric (baseline 454, best 320). Reprojection error backpropagates into ALL three heads simultaneously.
- 3 designs: (A) body-joint reprojection L1 with λ=0.5 (minimal), (B) body+pelvis reprojection with λ=1.0, (C) depth-weighted reprojection (geometry-aware, weights 2D error by predicted X/fx)
- Implementation: new `project_joints_to_2d` helper in `pelvis_utils.py` (torch-differentiable, X-clamp for stability); loss() in `pose3d_transformer_head.py` reuses existing `recover_pelvis_3d` per-sample K loop; float/bool literals in config.py
- Orthogonal to ALL prior ideas (pure loss addition); ideal composition partner with idea002 (architectural decoupling + loss-level consistency)
- Composite target < 160, with primary focus on mpjpe_pelvis_val < 170 and mpjpe_abs < 400

### idea011 (Iterative Pose Refinement via Coordinate-Conditioned Decoder) — Not Designed yet
- First **output-feedback** design: run decoder twice, encode pass-1 predicted 3D joints via small MLP, add to queries for pass 2, which predicts a residual correction. Final output = pass_1 + pass_2_residual.
- Differs from idea001 (multi-layer decoder, hidden-state only, no coord feedback), from idea003 (query init once from global feature, no prediction-based feedback), from all other ideas (architectural/loss-level, not iterative)
- 3 designs: (A) shared-weight two-pass + zero-init coord_enc, (B) Design A + intermediate supervision on pass-1 joints (weight 0.5), (C) Design B + independent decoder layer for pass 2
- Implementation: confined to pose3d_transformer_head.py + config.py (int/bool/float literals). coord_enc MLP zero-init on final linear for baseline-equivalent start (Deformable-DETR trick).
- Orthogonal to all prior ideas (output-space operation). Target: composite_val < 153; primary body MPJPE and mpjpe_abs gains.

### idea012 (Pairwise Joint Distance-Matrix Structural Prior Loss) — Not Designed yet
- First **structural prior** loss: supervises the 22×22 pairwise Euclidean distance matrix of predicted body joints against GT, enforcing bone-length + cross-body geometric consistency in 3D.
- Distinct from idea006 (skeleton attention bias — operates on attention, not loss), idea010 (2D reprojection — operates in image space, not 3D structure), idea011 (iterative refinement — architectural, not loss).
- Translation-invariant; amplifies gradient per joint (appears in 21 pair-entries); 231 unique pairs on upper triangle.
- 3 designs: (A) raw L1 distance loss λ=0.5 (minimal), (B) bone-length-weighted (skeleton edges up-weighted 2× via hard-coded bone_parents list), (C) log-scaled distance (proportion-aware, scale-invariant).
- Implementation: `torch.cdist` + upper-triangular indexing in pose3d_transformer_head.py loss(); float/str/list literals in config.py. No new learnable params (Designs A/C); optional 231-dim buffer for Design B.
- Orthogonal to all prior ideas; composable with idea002/008/010/011.
- Composite target < 154; primary body MPJPE < 140 (breaking the floor at 140.96 set by idea002/design002).

### idea014 (Anchor-Based Pelvis Depth via Discretized Classification Head) — Not Designed yet
- First **pelvis depth head output restructuring**: replaces `Linear(hidden_dim, 1)` scalar regression with a softmax over K=64 log-uniform depth bins in [1, 15] m; continuous pelvis depth recovered as soft-argmax expectation (fully differentiable). Loss becomes SORD-style soft-cross-entropy with Gaussian soft labels centred at GT depth (σ = 1.5 × bin_width in log-space).
- Motivation: `mpjpe_pelvis_val` has plateaued at 174–185 mm across all 13 prior ideas (best = 174.43, idea009/design003, only −1.67 mm vs baseline 176.10), while body MPJPE has moved 24 mm. All prior pelvis-targeted ideas attacked decoder, queries, attention, or loss balance — none changed the scalar-regression structure of the depth head itself. Known from monocular depth literature (DORN, AdaBins, BinsFormer) that classification-with-soft-argmax outperforms direct scalar regression.
- Distinct from idea002 (architectural pelvis query decoupling — still regression), idea004 (input-side depth PE), idea010 (loss-level 2D reprojection). First change to *what the depth head outputs*.
- 3 designs: (A) fixed log-uniform K=64 bins + soft-argmax + SORD soft-target CE, (B) Design A + aux SmoothL1 on expectation λ=0.3 (hybrid stability), (C) Design B + AdaBins-style adaptive per-sample bin widths (second `Linear(hidden_dim, 64)` for bin-width logits → cumulative edges).
- Implementation: `depth_head_type` str literal in config.py (values 'classification'/'classification_hybrid'/'classification_adaptive'); bin range/num/sigma/aux_weight as float/int literals. `pose3d_transformer_head.py` gains an `exp(log_bin_centres)` buffer + softmax expectation. `pelvis_utils.py`, metric, data pipeline unchanged. Downstream `pred['pelvis_depth']: (B,1)` scalar shape preserved.
- Orthogonal to all prior ideas; composable with idea002 (decoupled pelvis), idea005 (uncertainty weighting auto-rescales CE), idea010 (reprojection uses soft-argmax expectation — fully differentiable).
- Composite target < 160; primary focus on `mpjpe_pelvis_val < 170` and `mpjpe_abs < 440`.

### idea013 (Kinematic Chain Bone-Vector Output Parameterization) — Not Designed yet
- First **output-parameterization** change: body joints predicted as 21 bone vectors (parent→child offsets) + zero root, recovered via differentiable forward kinematics (cumsum along the skeleton tree). Every child's loss backpropagates through all ancestor bones.
- Distinct from idea012 (loss-level structural prior on unchanged outputs) and idea006/idea011 (attention/architectural). This is the *representation*, not a loss term.
- 3 designs: (A) bone-vec head with 1/sqrt(21) init rescale for baseline-equivalent variance, (B) Design A + auxiliary bone-length magnitude loss λ=0.3, (C) per-limb decoupled output heads (5 heads: spine/L-arm/R-arm/L-leg/R-leg).
- Implementation: `forward_kinematics` function in pose3d_transformer_head.py; `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]` (shared with idea012); bool/float/list literals in config.py.
- Orthogonal to all prior ideas; composable with idea002/008/011/012.
- Composite target < 153; primary body MPJPE < 140 (breaking the floor).

### idea021 (Learnable Cross-Attention Spatial Bias for Anatomically-Grounded Joint Localization)
- Adds a learnable additive bias `B_i ∈ R^{H'×W'}` to the cross-attention logits for each joint query over the spatial feature grid, implemented via `attn_mask` in `nn.MultiheadAttention`. Decouples *spatial routing* (where to look) from *semantic matching* (what to look for) within query embeddings.
- Zero-init → exact baseline at training start. 3 designs: (A) full `(70, 960)` bias matrix (~67K params), (B) low-rank factored row+column bias `(70, 24) + (70, 40)` = 4.5K params, (C) factored + Gaussian warm-start from anatomical joint row priors.
- First idea to bias cross-attention logits spatially per query. Distinct from idea006 (self-attention bias), idea007 (post-softmax value gating), idea019 (deformable sparse sampling), idea020 (per-query temperature/sharpness).
- NOTE: feat_h=40, feat_w=24 (H=640/16=40, W=384/16=24) — Designer must verify flatten ordering in baseline forward(). Config should set `feat_h=40, feat_w=24`.
- Implementation: `cross_attn_bias` parameter passed to `_DecoderLayer.forward()` as optional `attn_mask`; ~15 lines in pose3d_transformer_head.py, int/str/bool/list literals in config.py.
- Orthogonal to all prior ideas; composable with idea008 (body-focused 22 queries), idea020 (temperature scaling), idea006 (self-attn bias).
- Composite target < 218 (stage-2); primary body MPJPE < 152 and mpjpe_rel_val < 380.

### idea025 (Bilateral Symmetry Consistency Loss for Body Joint Pairs)
- Adds a bilateral symmetry consistency loss that couples gradient flow between 10 symmetric left-right joint pairs (shoulder, elbow, wrist, hip, knee, ankle, ball, eye, ear, heel — indices per SMPL 22-joint body).
- Penalizes predicted asymmetry (`asym_pred = joints_L - mirror(joints_R)`) deviating from GT asymmetry — not zero-symmetry, so genuinely asymmetric poses contribute zero loss.
- Mirror convention: negate Y-axis (BEDLAM2: X=forward, Y=left, Z=up); `sym_mirror_axis=1`.
- 3 designs: (A) uniform pair weights λ=0.3, (B) distal-upweighted pair weights (wrist/ankle/heel 2×) λ=0.5, (C) adaptive per-pair weight inversely proportional to GT asymmetry magnitude λ=0.5.
- Motivation: stage-1 body MPJPE plateau at 183–196 mm across 24 prior ideas; idea012 pairwise distance matrix degraded performance because unconstrained all-pairs; bilateral symmetric pairing is selective and directional.
- Designer must verify `sym_pairs` indices from `infra/constants.py` before hardcoding.
- Implementation: ~20 lines in `pose3d_transformer_head.py` loss(); list/float/bool/int literals in config.py. No new parameters.
- Composite target < 328 (stage-1, Design C); stage-2 target < 220.

### idea026 (Per-Joint Laplace NLL Uncertainty Regression for Body Joint Loss)
- Replaces fixed-scale SoftWeightSmoothL1 body-joint loss with Laplace NLL: model predicts joint mean + per-joint log-scale; NLL = log(2s) + |μ-y|/s.
- First idea to introduce per-joint, per-coordinate output uncertainty — distinct from idea005 (per-task coarse log-variance, 3 scalars). Per-joint routing means wrists/ankles get appropriate gradient damping.
- Zero-init of log_scale_out → s=1 at training start → exact L1 baseline recovery. Safe initialisation.
- 3 designs: (A) shared scalar per joint (Linear(hidden_dim,1) per body token), (B) per-axis scale (Linear(hidden_dim,3) per token), (C) Design A + entropy weight annealing (0.1→1.0 over 500 steps).
- Key impl note: log_scale_out applied per-token (B,22,hidden_dim)→(B,22,1 or 3); clamp log_s ∈ [-10,5]; AMP fp16 safety via autocast.
- Composable with idea013 (bone vectors — Laplace loss acts on recovered coordinates), idea005 (orthogonal granularities).
- Composite target < 330 (stage-1), < 225 (stage-2).

### idea028 (Decoupled Pelvis Coordinate Queries with Axis-Specific Cross-Attention)
- Root cause addressed: token 0 in the joint decoder carries conflicting objectives — relative pose encoding (via self-attention with body queries) AND absolute pelvis localization (for depth/UV output heads). Every idea that improved body MPJPE degraded or failed to improve pelvis MPJPE because of this entanglement.
- Mechanism: two dedicated "pelvis coordinate" queries (one for depth, one for UV) run a separate, lightweight cross-attention pass over spatial tokens (no self-attention, no FFN). `depth_out` and `uv_out` read from these dedicated queries; joint token 0 becomes a pure body joint query.
- 3 designs: (A) 8-head dedicated pelvis decoder, full 70-query joint decoder; (B) 4-head lighter pelvis decoder (simpler task, fewer heads); (C) decoupled pelvis + body-only 22-query joint decoder (combines idea008 hand-contamination removal with pelvis decoupling).
- Key impl: new `_PelvisCrossAttnDecoder` module (~10 lines), `pelvis_coord_queries = nn.Embedding(2, hidden_dim)`, `use_decoupled_pelvis` bool kwarg in config.py. Gradient flow to depth/UV now entirely via dedicated pelvis queries.
- Design C is most novel: joint self-attention sees only 22 body queries — no hand contamination (idea008) AND no pelvis-objective contamination (this idea).
- Composite target: Design C < 325 (stage-1), < 220 (stage-2).