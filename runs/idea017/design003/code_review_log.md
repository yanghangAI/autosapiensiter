
---
## 2026-04-21 — APPROVED
22-query 3-layer body decoder, intermediate supervision at layers 1 (w=0.4) and 2 (w=0.6). Code matches design. loss/joints_aux_0 and loss/joints_aux_1 confirmed in test output. Escalating weight formula yields [0.4, 0.6] for n_inter=2 as specified. Minor: constructor default num_decoder_layers=2 does not match design003 intent but config passes 3 explicitly — no functional issue. Watchable: grad_norm=inf at iter 50 in test run; training completed successfully. Monitor during full training.
