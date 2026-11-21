# PPML-SHUFFLE-LAST2-DPSGD-FINMEM1

Finite-horizon privacy-memory and final-state comparison experiment, executed CPU-only from the checksum-verified MEMFRONT2 review package.

## Current execution record

- Frozen DP-SGD kernel and hidden-support recurrence are unchanged.
- MEMFRONT2 core tests: 22 passed; saved one-worker/four-worker equivalence: exact.
- Finite-horizon memory table: completed for P1 and P2 through the requested horizons; P2 beyond direct response coverage is explicitly labeled `HYBRID_MEMORY`.
- Direct final-state runs: 32/32 complete, including all P1/P2 base grids and specified convergence anchors.
- Packet-size runs: 6/6 complete.
- P1 initialization sensitivity: 9/9 complete.
- Feldman--Shenfeld queries are still running in the pinned implementation; see `FINMEM1_PROGRESS.jsonl` and `FINMEM1_JOB_LEDGER.jsonl` for current status.
- No GPU was requested or used.

No final GO/REPAIR/STOP decision should be inferred from this README; use `FINMEM1_DECISION.json` after completion.
