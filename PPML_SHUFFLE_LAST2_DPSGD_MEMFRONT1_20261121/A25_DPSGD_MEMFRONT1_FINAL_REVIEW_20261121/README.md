# PPML-SHUFFLE-LAST2-DPSGD-MEMFRONT1

This archive records the gated MEMFRONT1 execution through the discrete privacy-memory stage. Terminal status: **REPAIR_MEMFRONT**. P2 memory failed to close by the preregistered `R=16384` cap; all final-state measurements were blocked before execution.

See `FINAL_MEMFRONT1_REPORT.md`, `MEMFRONT1_DECISION.json`, `MEMORY_SUMMARY.csv`, `MEMORY_RESPONSE.csv`, and `INDEPENDENT_AUDIT.json`. The canonical prior THEORYCOMPLY review archive is retained byte-for-byte in `INPUT/`. The prior `STOP_REALISTIC_DP_SGD` decision was not modified.

CPU only; seven process workers for independent memory lines; no GPU use. Every stage/checkpoint is recorded in `MEMFRONT1_JOB_LEDGER.jsonl`.
