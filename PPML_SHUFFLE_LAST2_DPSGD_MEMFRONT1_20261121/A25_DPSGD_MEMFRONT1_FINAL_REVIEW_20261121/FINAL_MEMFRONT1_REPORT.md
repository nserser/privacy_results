# MEMFRONT1 — interim-gate decision report

## Decision

**`REPAIR_MEMFRONT`**. The run reached the mandatory discrete-memory gate and stopped before final-state privacy measurements, Feldman–Shenfeld comparisons, asymptotic fits, and any practical GO/STOP claim.

## What ran

The canonical THEORYCOMPLY review package was verified against all 679 entries in its SHA-256 manifest. The pinned Feldman–Shenfeld checkout resolves to `fde83c45bf12c4f7821832c7d5d2ae2841fc0910`; the 67 tracked checkout files match the archived source. The THEORYCOMPLY core suite passed (22 tests), and its 1-worker versus 4-worker equivalence check passed exactly on all 8 records. A broad pytest invocation also reached 959 passes but stopped on one vendored PLD CTD/profile assertion differing by approximately one float64 ULP; this external assertion is documented in `M0_VERIFICATION.json`.

The new memory stage used seven CPU processes, one per independent `(eta,z,grid)` task, with numerical libraries limited to one thread per process. No GPU was requested or used. All 11 scheduled memory jobs completed and checkpointed their full response trajectories.

## Primary memory findings

The absent-user finite-grid kernel was iterated to a normalized Perron/quasi-stationary mass. The unnormalized `q K0 - q` L1 residual was below `1e-10` on every job. This explicitly accounts for the finite-domain kernel being sub-stochastic; the measured stationary escape was negligible.

- **P1** (`k=2, eta=0.10, z=2`): the impulse-response closure criteria passed at `R=2048`, with `L_TV=221.5049`. Base/grid/domain estimates were `221.5049`, `220.3047`, and `221.5049`; the maximum relative variation was 0.55%, within the 5% limit. Threshold crossings were `r_1e=106`, `r_1pct=523`, and `r_1e4=1083` on the base grid. The initial epoch sizes contain three points with measured `M/L_TV >= 4` (1024, 2048, 4096); no adaptive extension is triggered for P1.
- **P2** (`k=2, eta=0.10, z=4`): memory did **not** close by the frozen maximum `R=16384`. At that horizon `a_R/a_0=0.01526`, the relative change in `L_TV` from the preceding half-horizon was 48.71%, the final-quarter sum fraction was 13.06%, and the final-quarter squared-response fraction was 1.84%. The truncated estimate is `L_TV=4384.4243`; base/grid/domain values vary by 2.48%, but this does not cure the unclosed tail. `r_1e=112`; neither `r_1pct` nor `r_1e4` was reached within the horizon. The triggered epoch extension is 8192 and 16384, but even at 16384 the provisional ratio is only `M/L_TV=3.74`; the three-point asymptotic requirement cannot be met under the preregistered cap.

Secondary memory lines are included in `MEMORY_SUMMARY.csv` and the raw response file. They do not substitute for the failed P2 primary gate.

## Why no final-state jobs ran

The frozen execution order requires closed primary memory estimates before final-state scaling. P2 is not closed, so the final-state, convergence, matched-accountant, initialization, and packet-robustness jobs were marked `BLOCKED_BY_UNCLOSED_PRIMARY_MEMORY` in the expected-job manifest. This is the fail-closed `REPAIR_MEMFRONT` outcome; no final-state fit or baseline conclusion is inferred from memory-only data.

## Answers to the preregistered questions

1. P1's measured discrete memory horizon is `L_TV≈221.5`; P2 has only a capped, non-converged estimate `≈4384.4` and is unresolved.
2. The prior `M<=128` region is below the `M/L_TV>=4` asymptotic regime for P1 and far below it for P2.
3. Final privacy decay after memory normalization was not measured because final-state jobs are gated.
4. No confirmatory final-TV asymptotic exponents are available.
5. `k=4` was not run; no user-level packet robustness claim is available.
6. No Feldman–Shenfeld comparison was run.
7. P1 memory closes within the preregistered range; P2 crossover could not be placed within the permitted range.
8. P1 memory is stable under spatial refinement; P2's truncated `L_TV` has <5% spatial variation but its long tail is not closed.
9. Initialization sensitivity was not run because the final-state gate did not pass.
10. The frozen decision is **REPAIR**, not GO or STOP.

## Historical evidence separation

The prior SCALECERT result, including its official `STOP_REALISTIC_DP_SGD` decision for the earlier `M<=128` practical gate, remains immutable and is included in the canonical input package. Those prior results motivated MEMFRONT1 but are not pooled with these new measurements. The MEMFRONT1 measurements above are newly preregistered memory evidence only.

## Reproducibility

All progress events and horizon checkpoints are in `MEMFRONT1_JOB_LEDGER.jsonl`; memory jobs are resumable from their per-job JSON outputs. The exact input review archive is bundled under `INPUT/` and its SHA-256 is recorded in `ENVIRONMENT.json`. The frozen THEORYCOMPLY source was not modified; the new orchestration is under `SOURCES/memfront/`.
