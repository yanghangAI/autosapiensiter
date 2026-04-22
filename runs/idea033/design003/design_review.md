**Verdict: APPROVED**

Design 003 (Variant C — Pelvis-Token FiLM at Output) is complete and implementation-ready.

Checks:
- Starting point `baseline/`; files limited to `pose3d_transformer_head.py` + `config.py`; `pelvis_utils.py` unchanged; no invariants touched.
- FiLM application site explicit: on `pelvis_token = decoded[:, 0, :]` after `joints = self.joints_out(decoded)` (so body-joint pathway sees unmodulated `decoded`) and before `self.depth_out`/`self.uv_out`. Ordering constraint called out.
- Shape of gamma/beta is `(B, hidden_dim)` applied directly without unsqueeze (pelvis_token is `(B, hidden_dim)`) — correct and explicit.
- Shares scaffolding (imports, `_KFilmMLP`, `__init__` kwargs, `_build_k_batch`, `loss/predict` routing) with Design 001 by reference.
- Zero-init step-0 baseline equivalence preserved; config kwargs `use_k_film=True`, `k_film_variant='pelvis'`, `k_film_hidden=64` given.
- Edge cases covered (token-0 = pelvis convention, FiLM in hidden_dim space before output Linears, gradient flow).

Builder can implement without guessing.
