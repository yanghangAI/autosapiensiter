# Debug Log

Use this file to record unexpected automation or execution issues fixed by the Debugger.

---

## 2026-04-16 — setup-design creates code/code/ double-nesting (idea008)

**Reported by:** Orchestrator (on behalf of Builder)
**Affected designs:** idea008/design001, idea008/design002, idea008/design003

**Problem:** The `setup-design` CLI command created an extra `code/code/` nesting level in all 3 idea008 design directories (files landed in `runs/idea008/designXXX/code/code/` instead of `runs/idea008/designXXX/code/`). The SLURM sanity-test script does `cd "$TARGET_DIR/code"` then runs `python tools/train.py config.py` — with files one level deeper, `config.py` was not found, causing all 3 tests to fail with `FileNotFoundError: config.py`. The resulting `training_failed.txt` markers persisted after the Builder manually fixed the code placement and re-ran successful tests.

**Root cause:** The Builder passed `dst` ending with `code/` (e.g., `runs/idea008/design001/code/`) instead of the design directory (`runs/idea008/design001/`). `setup_design.py` unconditionally computed `code_dir = dst / cfg.setup_design.destination_subdir`, so `dst/code/` became `dst/code/code/`.

**Files changed:**
- `scripts/tools/setup_design.py` — added guard that detects when `dst.name == destination_subdir` and strips the trailing component, printing a warning, to prevent double-nesting.
- `runs/idea008/design001/training_failed.txt` — removed (stale)
- `runs/idea008/design002/training_failed.txt` — removed (stale)
- `runs/idea008/design003/training_failed.txt` — removed (stale)
- `runs/idea008/design_overview.csv` — all 3 designs updated from "Training Failed" to "Implemented"

**What to retry:** Run `python scripts/cli.py submit-implemented` — all 3 idea008 designs are now visible as "Implemented" and have valid `test_output/metrics.csv` (epoch=1, 6 metric columns). Full training jobs can be submitted.
