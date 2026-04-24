# Reviewer Memory

This file serves as the persistent memory storage for the Reviewer. Keep it concise.

## idea001 — Design Review (2026-04-16)

All 4 designs APPROVED. Verdicts written to:
- `runs/idea001/design001/design_review.md` + `design_review_log.md`
- `runs/idea001/design002/design_review.md` + `design_review_log.md`
- `runs/idea001/design003/design_review.md` + `design_review_log.md`
- `runs/idea001/design004/design_review.md` + `design_review_log.md`

`sync-status` ran successfully after all designs passed.

Key patterns verified across all 4 designs:
- Starting point: `baseline/`
- Only `pose3d_transformer_head.py` and `config.py` modified
- `pelvis_query = nn.Embedding(1, in_channels)` at index 0, joint queries 1:71
- `decoder_layers = nn.ModuleList([...])` replaces `decoder_layer` (singular)
- D1/D2: no aux loss; D3/D4: aux loss at 0.4× intermediate, 1.0× final
- D3 loss keys: aux0, aux1, train; D4 loss keys: aux0, aux1, aux2, train

## idea005 — Code Review (2026-04-16)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea005/design001/code_review.md` + `code_review_log.md` — Design A (full uncertainty weighting, all 3 tasks)
- `runs/idea005/design002/code_review.md` + `code_review_log.md` — Design B (pelvis-only uncertainty, joint anchored)
- `runs/idea005/design003/code_review.md` + `code_review_log.md` — Design C (pelvis uncertainty + joint_loss_scale=2.0)

`sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs:
- All changes confined to `pose3d_transformer_head.py` and `config.py`. `pelvis_utils.py` and `train.py` unchanged.
- D1: `use_uncertainty_weighting=True` in config; 3× `nn.Parameter(torch.zeros(1))` (`log_var_joints`, `log_var_depth`, `log_var_uv`). Formula: `exp(-lv)*raw + lv` with `.clamp(-4,4)` on local variable.
- D2: `uncertainty_pelvis_only=True` in config; 2× `nn.Parameter` (depth, uv only, no joints). Joint loss fixed at weight 1.0.
- D3: `uncertainty_pelvis_only=True` + `joint_loss_scale=2.0` in config; `joint_loss_scale` is plain float applied to raw_joints before the conditional. `_train_mpjpe` not scaled by `joint_loss_scale`. Confirmed by training log: `loss/joints/train ≈ 0.377` (~2× D1/D2's 0.192).
- All configs: `persistent_workers=False`, no Python import statements, seed=2026, correct `output_dir`.

## idea006 — Design Review (2026-04-16)

All 3 designs APPROVED. Verdicts written to:
- `runs/idea006/design001/design_review.md` + `design_review_log.md`
- `runs/idea006/design002/design_review.md` + `design_review_log.md`
- `runs/idea006/design003/design_review.md` + `design_review_log.md`

Design `sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs:
- Starting point: `baseline/`
- Only `pose3d_transformer_head.py` modified (D001); `pose3d_transformer_head.py` + `config.py` (D002, D003).
- All add `attn_bias` as `nn.Parameter` to `_DecoderLayer`, passed as `attn_mask` to `self.self_attn`.
- D001: single `(70,70)` zero-init bias; unconditional (no new config key). No config.py change.
- D002: `(70,70)` skeleton-graph warm-start (`+0.5` adjacent, `-0.5` pelvis diag); `_build_skeleton_attn_bias` helper at module level; `attn_bias_type='skeleton_init'` in config.
- D003: `(8,70,70)` per-head zero-init bias; expanded to `(B*8,70,70)` at forward; `attn_bias_mode` string param; `attn_bias_type='per_head'` in config.
- All: `batch_first=True` already set on baseline `self.self_attn`; `(J,J)` and `(B*H,J,J)` shapes are valid PyTorch `attn_mask` forms.

## idea006 — Code Review (2026-04-16)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea006/design001/code_review.md` + `code_review_log.md` — Design A (shared zero-init attn bias, 4900 params)
- `runs/idea006/design002/code_review.md` + `code_review_log.md` — Design B (skeleton-graph warm-start bias, 4900 params)
- `runs/idea006/design003/code_review.md` + `code_review_log.md` — Design C (per-head bias 8×70×70, 39200 params)

`sync-status` ran successfully after all designs passed.

Key code patterns verified:
- D001: `_DecoderLayer.__init__` adds `num_joints=70` param and `self.attn_bias = nn.Parameter(torch.zeros(J,J))`; `forward` passes `attn_mask=self.attn_bias`; `Pose3dTransformerHead` passes `num_joints=num_joints` to `_DecoderLayer`. Config unchanged (only output_dir).
- D002: adds `_build_skeleton_attn_bias` at module level with hardcoded edge list, bidirectionality and pelvis diag confirmed; `_DecoderLayer` accepts `attn_bias_init` with `.float().clone()`; `Pose3dTransformerHead` dispatches `'skeleton_init'`/`'zero_init'`/`'none'`; config adds `attn_bias_type='skeleton_init'`.
- D003: `_DecoderLayer` accepts `attn_bias_mode='none'/'shared'/'per_head'`; per_head path uses `.unsqueeze(0).expand(B,-1,-1,-1).contiguous().reshape(B*H,J,J)`; `self.attn_bias=None` for `'none'`; config adds `attn_bias_type='per_head'`.
- All: `pelvis_utils.py` and `train.py` unchanged; all test runs completed cleanly with valid metrics in epoch 1.

## idea007 — Design Review (2026-04-16)

All 3 designs APPROVED. Verdicts written to:
- `runs/idea007/design001/design_review.md` + `design_review_log.md`
- `runs/idea007/design002/design_review.md` + `design_review_log.md`
- `runs/idea007/design003/design_review.md` + `design_review_log.md`

`sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs:
- Starting point: `baseline/`
- Only `pose3d_transformer_head.py` and `config.py` modified.
- All add `self.cross_attn_bias = nn.Parameter(...)` to `_DecoderLayer`, passed as `attn_mask` to `self.cross_attn`.
- D001: single `(70, 960)` zero-init bias; `num_joints`/`num_spatial` kwargs to `_DecoderLayer`; `num_spatial=960` in config.
- D002: same structure + `cross_attn_bias_init: str` kwarg; Gaussian vertical-band prior (lower center=30.0, upper center=10.0, sigma=5.0, scale ±0.5); `cross_routing_type='band_prior'` in config; `rounding_mode='floor'` required for row index.
- D003: per-head bias `(num_heads, J, S)`; expanded to `(B*num_heads, J, S)` at forward; `per_head_routing: bool` kwarg; `B` passed from `Pose3dTransformerHead.forward`; `_per_head` and `_num_heads` stored as instance attrs; "Revised approach" in section 1c supersedes partial signature in section 1a.
- All: `batch_first=True` on `self.cross_attn`; assert `spatial_tokens.shape[1] == cross_attn_bias.shape[-1]` in forward; `_init_head_weights` must not touch `cross_attn_bias`.

## idea007 — Code Review (2026-04-16)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea007/design001/code_review.md` + `code_review_log.md` — Design A (zero-init shared cross-attn bias, 67,200 params)
- `runs/idea007/design002/code_review.md` + `code_review_log.md` — Design B (Gaussian band-prior warm-start, 67,200 params)
- `runs/idea007/design003/code_review.md` + `code_review_log.md` — Design C (per-head cross-attn bias, 537,600 params)

`sync-status` ran successfully after all designs passed.

Key code patterns verified:
- D001: `_DecoderLayer` accepts `num_joints`/`num_spatial`; `cross_attn_bias = nn.Parameter(torch.zeros(J,S))`; assert in forward; `attn_mask=cross_attn_bias` passed. `Pose3dTransformerHead` adds `num_spatial=960` kwarg; config adds `num_spatial=960`.
- D002: `_DecoderLayer` adds `cross_attn_bias_init: str` kwarg; Gaussian band prior (lower centre 30.0, upper 10.0, sigma=5.0, ±0.5 scale); float-safe `.div(_W_prime, rounding_mode='floor')` for row index; `cross_routing_type='band_prior'` mapped via dict; config adds `cross_routing_type='band_prior'`. Design002 epoch-1 composite (484.83) already lower than D001/D003 (491.04), consistent with warm-start advantage.
- D003: adds `per_head_routing: bool` to `_DecoderLayer`; per-head path: `cross_attn_bias` shape `(H,J,S)`; expanded via `.unsqueeze(0).expand(B,-1,-1,-1).reshape(B*H,J,S)`; `_per_head`/`_num_heads` stored; `_DecoderLayer.forward` accepts `B:int=1`; `Pose3dTransformerHead.forward` passes `B=B` explicitly; config adds `cross_routing_type='per_head'`.
- All: `pelvis_utils.py` and `train.py` unchanged; all test runs completed cleanly with valid metrics at epoch 1.

## idea008 — Code Review (2026-04-16)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea008/design001/code_review.md` + `code_review_log.md` — Design A (22-query body decoder, zero-pad hand region)
- `runs/idea008/design002/code_review.md` + `code_review_log.md` — Design B (22-query decoder + linear hand recovery Linear(5632,144), aux loss 0.1)
- `runs/idea008/design003/code_review.md` + `code_review_log.md` — Design C (22-query decoder + 2-layer MLP hand recovery, aux loss 0.3)

`sync-status` ran successfully after all designs passed.

Key code patterns verified:
- All 3: `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` (22 entries); `self.num_joints = 70` preserved; decoder produces `(B,22,hidden_dim)`; body_joints `(B,22,3)` from `joints_out`.
- D001: zero-pad `torch.zeros(B,48,3)` concatenated to body_joints; no hand loss; `_init_head_weights` unchanged.
- D002: `hand_proj = nn.Linear(5632, 144)` after `uv_out`; `body_flat = decoded.reshape(B, 22*256)`; `hand_joints` via reshape; `loss/hand_aux/train = 0.1 * loss_joints_module(pred[:,22:70], gt[:,22:70])`; reuses existing `loss_joints_module`.
- D003: `hand_proj = nn.Sequential(Linear(5632,256), GELU(), Linear(256,144))`; per-Linear init with `trunc_normal_(std=0.02)` + zero bias; aux loss weight 0.3; same loss formula as D002.
- All: `pelvis_token = decoded[:,0,:]` unchanged; `config.py` adds `num_body_queries=22` (D1,D2,D3) and `hand_aux_loss_weight=0.1` (D2) or `0.3` (D3); all are literals. `pelvis_utils.py` and `train.py` unchanged. All test runs completed cleanly.

## idea004 — Code Review (2026-04-16)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea004/design001/code_review.md` + `code_review_log.md` — Design A (scalar linear)
- `runs/idea004/design002/code_review.md` + `code_review_log.md` — Design B (depth sinusoidal)
- `runs/idea004/design003/code_review.md` + `code_review_log.md` — Design C (3-input MLP)

`sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs:
- `_extract_depth_map()` reads `depth_npy_path` from metainfo, crops to `img_shape`, bilinear resize to feature map resolution `(B,1,H',W')`.
- Depth normalisation: clamp [0,10] m / 10 → [0,1] in `forward()`.
- D1: `depth_proj = nn.Linear(1,256)` zero-init, additive on top of 2D sincos. `depth_map is None` branch skipped.
- D2: `_build_1d_sincos_enc` function, `depth_pos_proj = nn.Linear(384,256)` trunc_normal, fallback uses zero-padded depth (always goes through `depth_pos_proj`). `_get_pos_enc` still called.
- D3: `pos_mlp = Sequential(Linear(3,64), GELU, Linear(64,256))` trunc_normal. `_build_3d_pos_grid` returns `(B,H*W,3)` or `(1,H*W,3)` (fallback expanded in forward). `_get_pos_enc` NOT called in forward.
- D3 is significantly slower (~15s/iter vs ~1.8s) — noted, not a correctness issue.
- `depth_npy_path` included in `meta_keys` in both train and val pipeline for all designs.

## idea012 — Design Review (2026-04-17)

All 3 designs APPROVED. Verdicts written to:
- `runs/idea012/design001/design_review.md` + `design_review_log.md`
- `runs/idea012/design002/design_review.md` + `design_review_log.md`
- `runs/idea012/design003/design_review.md` + `design_review_log.md`

`sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs (Pairwise Joint Distance-Matrix loss):
- Starting point: `baseline/`.
- Only `pose3d_transformer_head.py` and `config.py` modified; `pelvis_utils.py` unchanged.
- New `__init__` kwargs: `dist_loss_weight: float = 0.0`, `dist_loss_mode: str = 'abs'`, `dist_loss_eps: float = 1e-3` (D002 adds `bone_parents: list = None`). Assert validates `dist_loss_mode ∈ {'abs','bone_weighted','log'}`. Defaults preserve baseline exactly.
- `loss()` insertion: after the three existing losses, before `with torch.no_grad():`. Guarded by `if self.dist_loss_weight > 0.0:`.
- `torch.cdist(pred_body, pred_body, p=2)` + `torch.triu_indices(22, 22, offset=1, device=pred_body.device)` on `_BODY = list(range(0,22))` slice; 231 pairs; `.mean()` not `.sum()`.
- Loss key `'loss/dist_matrix/train'`. Scale by `self.dist_loss_weight` applied AFTER the mean.
- D001 mode='abs', λ=0.5; D002 mode='bone_weighted', λ=0.5, `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`, `register_buffer('bone_weights', ..., persistent=False)` (231,), 21 bone edges × 2.0 / 210 non-bone × 1.0; D003 mode='log', λ=0.5, eps=1e-3 INSIDE each log.
- `forward()` and `predict()` unchanged. Training-only loss.
- All three share same three-branch `if/elif/else` — unreachable branches are documented no-ops.

## idea017 — Code Review (2026-04-21)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea017/design001/code_review.md` + `code_review_log.md` — Design A (22-query 2-layer decoder, no intermediate supervision)
- `runs/idea017/design002/code_review.md` + `code_review_log.md` — Design B (22-query 2-layer decoder, intermediate body loss weight 0.4)
- `runs/idea017/design003/code_review.md` + `code_review_log.md` — Design C (22-query 3-layer decoder, intermediate losses weights 0.4 and 0.6)

`sync-status` ran successfully after all designs passed.

Key code patterns verified:
- All 3: `joint_queries = nn.Embedding(22, 256)`; `decoder_layers = nn.ModuleList([_DecoderLayer(256,8,0.1)] * N)`; `hand_proj = nn.Linear(5632, 144)`; `forward()` iterates layers, stores `_intermediate_outputs`, concatenates body+hand to (B,70,3); `pelvis_token = queries[:,0,:]`.
- D001: `aux_body_loss_weight=0.0` → intermediate branch skipped; `hand_aux_loss_weight=0.1` → `loss/hand_aux/train` active. `num_decoder_layers=2`.
- D002: `aux_body_loss_weight=0.4`, `num_decoder_layers=2` → n_inter=1, `intermediate_weights=[0.4]` → `loss/joints_aux_0/train` at weight 0.4 confirmed in test log.
- D003: `aux_body_loss_weight=0.4`, `num_decoder_layers=3` → n_inter=2, `intermediate_weights=[0.4, 0.6]` via formula `0.4*(1+0.5*k)` for k=0,1 → both `loss/joints_aux_0` and `loss/joints_aux_1` confirmed in test log.
- D003: constructor default `num_decoder_layers=2` is misleading but config passes 3 explicitly — no functional issue.
- D003: `grad_norm=inf` at iter 50 in test run; training still completed successfully. Monitor during full training.
- All: `pelvis_utils.py` and `train.py` unchanged. `persistent_workers=False` preserved. No Python imports in configs.

## idea018 — Design Review (2026-04-21)

All 3 designs APPROVED. Verdicts written to:
- `runs/idea018/design001/design_review.md` + `design_review_log.md`
- `runs/idea018/design002/design_review.md` + `design_review_log.md`
- `runs/idea018/design003/design_review.md` + `design_review_log.md`

`sync-status` attempted but Python environment not directly accessible from shell. Orchestrator must run manually.

Key patterns across all 3 designs (Depth-Gated Cross-Attention):
- Starting point: `baseline/`.
- Only `pose3d_transformer_head.py` and `config.py` modified; `pelvis_utils.py` unchanged.
- Core mechanism: two zero-init `nn.Linear(256,1)` probes (`depth_probe_global`, `depth_probe_token`). `z_hat=(B,1)` global body depth estimate; `z_tok=(B,960)` per-token depth. Gate: `attn_logit_bias = -0.5 * ((z_tok - z_hat) / sigma)^2`, values ≤ 0.
- Gate passed as float `attn_mask=(B*num_heads, Nq, N_spatial)` to `nn.MultiheadAttention` cross-attn.
- `_DecoderLayer.forward()` modification identical in all 3 designs; accepts optional `attn_logit_bias` argument.
- D001: fixed `sigma=1.0` buffer (`depth_gate_sigma_buf`); no auxiliary loss. Config: `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0`.
- D002: learnable `log_sigma = nn.Parameter(torch.zeros(1))` → `sigma = exp(log_sigma).clamp(min=0.01)`; auxiliary probe loss `0.1 * loss_depth_module(z_hat, gt_depth)` key `'loss/depth_probe/train'`; `_depth_probe_z_hat` cached in `forward()`. Config: `depth_gate_type='gaussian_learnable_sigma'`, `depth_probe_loss_weight=0.1`.
- D003: combines 22-query body decoder (idea008/design002) + fixed depth gate; `hand_proj=Linear(5632,144)` trunc-normal init; `hand_aux_loss_weight=0.1`; 4 loss keys; gate broadcasts over 22 queries → `(B*8, 22, 960)`. Config: `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0`.
- All: zero-init probes → flat gate at step 0 = baseline. Output shapes unchanged (B,70,3), (B,1), (B,2). AMP-safe (bounded negative logits). All literals in config.

## idea023 — Code Review (2026-04-21)

design001 APPROVED. design002 REJECTED. design003 REJECTED.

All three designs share one head file. Critical bug in design002/003:

**Gaussian loss reduction bug:** Designs 002/003 use `heatmap_target='gaussian'` (KL loss path). The code at line 431 of `pose3d_transformer_head.py` uses `-(gt_hm * log_probs).sum()` (sum over both 22 joints and 960 spatial tokens), but the design002/003 spec requires `-(gt_hm * log_probs).sum(dim=-1).mean()` (sum over spatial, mean over joints). The `.sum()` inflates the heatmap loss by ~22×: observed `loss/heatmap/train ≈ 29.8` vs. expected ~1.37 at initialisation. This makes heatmap loss dominate (~91% of total) and swamps 3D regression signal. Design003 additionally gets `grad_norm: inf` due to the learnable temperature exacerbating the instability.

**Design001 is correct:** Uses `heatmap_target='onehot'` (cross-entropy path), which is never affected by the `.sum()` issue. Test run clean with `loss/heatmap/train: 0.686`.

**Design003 bug fix confirmed:** Initial test run had `RuntimeError` from `view(1,1,22)` (wrong broadcast axis for learnable temp). Fixed to `view(1,22,1)` in the code — correct.

**Required fix for designs 002/003:** Change line 431 from `.sum()` to `.sum(dim=-1).mean()`. Since all three designs share the same head file, the fix must be targeted to only affect design002 and design003 (or the shared file must be updated for both).

Note: `sync-status` NOT run because designs 002/003 rejected.

## idea024 — Code Review (2026-04-21)

All 3 designs APPROVED. Automated check passed for all. Verdicts written to:
- `runs/idea024/design001/code_review.md` + `code_review_log.md` — Design A (alpha=0.5, linear normalisation)
- `runs/idea024/design002/code_review.md` + `code_review_log.md` — Design B (alpha=1.0, softmax T=1.0) — high-risk degenerate confirmed
- `runs/idea024/design003/code_review.md` + `code_review_log.md` — Design C (alpha=1.0, group-normalised + 5-epoch warmup) — one-time inf grad_norm

`sync-status` ran successfully after all designs passed.

Key findings:
- All 3 changes confined to `pose3d_transformer_head.py` + `config.py`; `pelvis_utils.py` and `train.py` unchanged.
- **Design002 softmax degeneracy confirmed:** `softmax(ema_mm / T=1.0)` at realistic mm-scale EMA values (77–155 mm after epoch 1, 150–350 mm at convergence) concentrates ≈22.0 weight on the single hardest joint and ≈0 on all others. This is near-one-hot weighting and will likely produce worse-than-baseline `mpjpe_body_val`. The elevated `loss_joints_train` in epoch 1 (0.31–0.50 range vs 0.20–0.26 for design001) is already consistent with this degeneracy. Code is correct per spec; the flaw is at the design level.
- **Design003 inf grad_norm:** single occurrence at iter 50 (the only MMEngine log point in the test epoch); all 72 iter_metrics entries finite and normal (0.176–0.234); training completed; AMP GradScaler handled it. Warmup ramp at iter 50 is only 3% active, so design003 operates nearly like baseline during the test epoch.
- Design001 (alpha=0.5, linear) is the cleanest variant — no anomalies, correct mechanics, expected to be the diagnostic control.

## idea027 — Design Review (2026-04-21)

All 3 designs APPROVED. Verdicts written to:
- `runs/idea027/design001/design_review.md` + `design_review_log.md`
- `runs/idea027/design002/design_review.md` + `design_review_log.md`
- `runs/idea027/design003/design_review.md` + `design_review_log.md`

`sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs (Spatial Token Context Enrichment via Depthwise-Separable Conv):
- Starting point: `baseline/`.
- Only `pose3d_transformer_head.py` and `config.py` modified; `pelvis_utils.py` unchanged.
- New module `_SpatialContextNet` inserted before `_DecoderLayer`. Identical class definition used for all three designs; parameterized via kwargs.
- Insertion point in `forward()`: after `spatial = spatial + pos_enc`, before `queries = self.joint_queries...`.
- `H, W` sourced from `B, C, H, W = feat.shape` (already present in baseline forward).
- D001: `num_layers=1`, `norm='none'` (Identity), GELU, zero-init on only pointwise. Config: `spatial_ctx_norm='none'`, no `spatial_ctx_groups` in config (unused, defaults to 32 in head).
- D002: `num_layers=1`, `norm='groupnorm'`, GroupNorm(32,256), GELU, zero-init pointwise. Config: adds `spatial_ctx_groups=32`.
- D003: `num_layers=2`, `norm='groupnorm'`, GroupNorm(32,256), GELU. Only last (second) pointwise zero-init; first pointwise trunc_normal(0.02). Single outer residual only — no per-layer residual inside sequential.
- Zero-init guarantee holds in all cases: final pointwise weight=0 and bias=0 → net output=0 for any input → delta=0 → spatial unchanged at init.
- GroupNorm divisibility: 256/32=8. Satisfied.
- All config values are bool/int/str literals; no Python import statements. MMEngine constraint satisfied.

## idea030 — Design Review (2026-04-21)

All 3 designs APPROVED. Verdicts written to:
- `runs/idea030/design001/design_review.md` + `design_review_log.md`
- `runs/idea030/design002/design_review.md` + `design_review_log.md`
- `runs/idea030/design003/design_review.md` + `design_review_log.md`

`sync-status` ran successfully after all designs passed.

Key patterns across all 3 designs (Lightweight Spatial Encoder via Single-Layer Self-Attention over Spatial Tokens):
- Starting point: `baseline/`.
- Only `pose3d_transformer_head.py` and `config.py` modified; `pelvis_utils.py` unchanged.
- New `_EncoderLayer` class inserted before `_DecoderLayer`; full code listing provided in all three design.md files.
- Insertion point in `forward()`: after `spatial = spatial + pos_enc`, before `queries = self.joint_queries.weight.unsqueeze(0).expand(...)`.
- New `__init__` kwargs with defaults: `use_spatial_encoder=False`, `num_encoder_layers=1`, `encoder_num_heads=8`, `encoder_dropout=0.1`, `encoder_zero_init=True`.
- `nn.ModuleList` built when `use_spatial_encoder=True`; not created otherwise (no attribute registered).
- Zero-init on `self.self_attn.out_proj` and `ffn[-2]` (second-to-last of `nn.Sequential`) — both residual paths zeroed at init.
- Pre-norm architecture (consistent with `_DecoderLayer`).
- D001: 1 layer, 8 heads. Config: `encoder_num_heads=8`. Memory ~58 MB encoder attn.
- D002: 1 layer, 4 heads. Config: `encoder_num_heads=4`. Memory ~29 MB. `hidden_dim=256` divisible by 4 (head_dim=64).
- D003: 2 layers, 4 heads. Config: `num_encoder_layers=2`, `encoder_num_heads=4`. Memory ~58 MB total. Loop runs twice; each layer independently zero-init.
- All config values are bool/int/float literals; no Python import statements. MMEngine constraint satisfied.