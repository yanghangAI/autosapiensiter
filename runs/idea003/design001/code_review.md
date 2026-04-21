## Code Review — idea003/design001

**Verdict: APPROVED**

**Automated check:** PASSED (`python scripts/cli.py review-check-implementation runs/idea003/design001`)

---

### Files Changed

`implementation_summary.md` lists two files changed: `code/pose3d_transformer_head.py` and `code/config.py`. Both are permitted by `design.md`. `pelvis_utils.py` and `train.py` are confirmed unmodified (match baseline exactly).

---

### Code Fidelity vs. Design

**`pose3d_transformer_head.py`:**

1. `query_cond_type: str = 'linear'` added as parameter to `__init__` — CORRECT.
2. `self.query_cond_type` stored — CORRECT.
3. `self.query_cond_net = nn.Linear(hidden_dim, num_joints * hidden_dim)` created after `self.decoder_layer` — CORRECT.
4. `nn.init.trunc_normal_(self.query_cond_net.weight, std=0.02)` — CORRECT.
5. `nn.init.zeros_(self.query_cond_net.bias)` — CORRECT.
6. `else: raise ValueError(...)` guard — CORRECT.
7. `_init_head_weights` is NOT modified to touch `query_cond_net` — CORRECT.
8. `forward`: `static_q = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)` — CORRECT.
9. `global_feat = spatial.mean(dim=1)` computed AFTER `spatial = spatial + pos_enc` — CORRECT (positional encoding constraint satisfied).
10. `offsets = self.query_cond_net(global_feat)` — CORRECT.
11. `offsets = offsets.reshape(B, self.num_joints, self.hidden_dim)` — CORRECT.
12. `queries = static_q + offsets` — CORRECT.
13. Everything after `decoded = self.decoder_layer(queries, spatial)` is unchanged — CORRECT.
14. Loss restricted to `_BODY = list(range(0, 22))` — CORRECT.
15. `pelvis_depth`/`pelvis_uv` read from `decoded[:, 0, :]` — CORRECT.

**`config.py`:**
- `query_cond_type='linear'` added to `head` dict — CORRECT.
- `output_dir` patched to design001 path — expected.
- All other values (LR, weight decay, seed, batch size, hooks, schedule, pipeline, persistent_workers=False) are identical to baseline — CORRECT.

---

### Test Output

- Test run completed successfully: `[test] Finished.` with no errors.
- 1 epoch of training completed; validation ran on 76 batches.
- `metrics.csv` produced: `epoch=1, composite_val=425.79, mpjpe_body_val=350.40, mpjpe_pelvis_val=578.84` — values are plausible for epoch 1 of a randomly initialized head with pretrained backbone; no NaN/inf.
- Training loss at iter 50: `loss=1.508461` — finite and reasonable.
- GPU memory usage: 10681 MiB on GTX 1080 Ti (11 GB) — within limit.

---

### Invariant Check

No invariant files modified. `pelvis_utils.py`, `train.py` match baseline exactly. Backbone, data preprocessor, dataset, transforms, evaluation metric, and infra files are not in the code directory and were not touched.
