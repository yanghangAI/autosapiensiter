**Verdict: APPROVED**

Design 001 (Variant A — Query FiLM) is complete, explicit, and implementation-ready.

Checks:
- Design Description present; starting point `baseline/` specified.
- Files to modify limited to allowed set: `pose3d_transformer_head.py`, `config.py`. `pelvis_utils.py` explicitly unchanged.
- No invariant files modified (metric, dataset, transforms, backbone, preprocessor, infra, `train.py`).
- K extraction, normalization (`_W_REF=384.0`, `_H_REF=640.0`), `_KFilmMLP` architecture, zero-init policy, FiLM application site (pre-decoder-layer on expanded queries, `q*(1+gamma)+beta`), `forward()` signature change with optional `k_batch`, and `loss()`/`predict()` routing are all concretely specified.
- Config kwargs and exact values given (`use_k_film=True`, `k_film_variant='query'`, `k_film_hidden=64`).
- Output dict keys/shapes, loss keys, body-only joint loss, telemetry, and step-0 baseline equivalence invariants explicitly preserved.
- Edge cases covered (missing K fallback, img_shape indexing, AMP dtype, single decoder layer → no per-layer repetition).

Builder can implement without guessing.
