# FINMEM1A P1 completion report

## Decision

**GO_REALISTIC_ULDP_ML**. The result concerns the frozen one-dimensional discrete numerical witness only. It does not establish a general neural-network DP-SGD theorem.

The historical SCALECERT, MEMFRONT1, and MEMFRONT2 decisions remain unchanged. The 14 P1 k=2 Feldman–Shenfeld (FS) outputs are reused from the immutable 19/40 snapshot and verified against IDs, parameters, and pinned commit. New P1 final-state jobs were run in this separate branch on CPU. The conditional k=4 FS queries used GPU only after same-node bitwise kernel parity plus the archived full-accountant CPU/GPU parity audit. Broad Slurm job 930 was cancelled; its partial outputs were excluded.

## Finite-horizon memory

| M | Richardson continuum L_TV(M) | x_M | Status |
|---:|---:|---:|---|
| 256 | 178.891 | 1.43104 | PASS |
| 512 | 214.903 | 2.38247 | PASS |
| 1024 | 219.838 | 4.65798 | PASS |
| 2048 | 219.913 | 9.31279 | PASS |
| 4096 | 219.936 | 18.6236 | PASS |
| 8192 | 219.983 | 37.2392 | PASS |
| 16384 | 220.077 | 74.4467 | PASS |

Each point passes domain-vs-base <=1%, fine-grid/Richardson <=2%, and finite-domain defect <=0.5%. The spectral continuation used beyond direct coverage passes the frozen held-out validation (<5% pointwise error). This uses L_TV(M), not L_TV(infinity).

## Final-state privacy

| M | L_TV(M) | x_M | Final TV | Epsilon upper, delta=1e-3 | Epsilon upper, delta=1e-5 |
|---:|---:|---:|---:|---:|---:|
| 256 | 178.891 | 1.43104 | 0.0198702 | 0.0852861 | 0.169381 |
| 512 | 214.903 | 2.38247 | 0.010954 | 0.0401391 | 0.0995405 |
| 1024 | 219.838 | 4.65798 | 0.0055435 | 0.0162806 | 0.0483684 |
| 2048 | 219.913 | 9.31279 | 0.00277239 | 0.00495883 | 0.0211826 |
| 4096 | 219.936 | 18.6236 | 0.00138625 | 0.000991841 | 0.00973164 |
| 8192 | 219.983 | 37.2392 | 0.000693145 | 0 | 0.00411972 |
| 16384 | 220.077 | 74.4467 | 0.000346588 | 0 | 0.00184301 |

All seven base points are fresh DIRECT_FULL_GRID computations. Anchor grids BASE/GRID/GRID2/DOMAIN pass at M=1024, 4096, and 16384. At x>=4, beta=0.999892, fit 95% regression interval [0.999788, 0.999995], R2=1, n=5. At x>=8, beta=0.999948, R2=1, n=4. Dropping the first x>=4 point gives beta=0.999948, R2=1. The last three local exponents are positive.

## Archived FS comparison

Ratios are conservative final epsilon upper / exact archived FS lower. Strict win requires ratio <0.99.

| M | Ratio delta=1e-3 | Ratio delta=1e-5 | Both-delta strict win |
|---:|---:|---:|---|
| 256 | 0.7268 | 0.790405 | WIN |
| 512 | 0.527716 | 0.678708 | WIN |
| 1024 | 0.333878 | 0.482088 | WIN |
| 2048 | 0.161075 | 0.308976 | WIN |
| 4096 | 0.0521409 | 0.208052 | WIN |
| 8192 | 0 | 0.129408 | WIN |
| 16384 | 0 | 0.0854994 | WIN |

The contiguous asymptotic region M=1024 through 16384 wins at both deltas, with the ratio declining as M increases. No archived P1 k=2 query was rerun.

## k=4 robustness

Three fresh k=4 final-state points were computed at M=1024, 2048, and 4096. The pinned accountant with deterministic GPU gather was triggered only after P1 k=2 passed and same-node parity passed; it stopped after two consecutive candidate points passed both deltas.

| M | delta | Final epsilon upper | FS lower | Result |
|---:|---:|---:|---:|---|
| 1024 | 0.001 | 0.0411571 | 0.117219 | WIN |
| 1024 | 1e-05 | 0.104763 | 0.213812 | WIN |
| 2048 | 0.001 | 0.0155152 | 0.0760072 | WIN |
| 2048 | 1e-05 | 0.0474313 | 0.14644 | WIN |

## Execution and limitations

The initial FINMEM1A specification described the final-state curve as missing. A prior local FINMEM1 tree contained older final-state artifacts; these were not silently relabeled. FINMEM1A regenerated its P1 memory and final-state computations. P2 FS remains deferred; k=1 FS was not run. The memory table generator also produced P2 memory diagnostics as a by-product; no P2 final-state or accountant result affects this decision.

Independent audit: **PASS**. P1 final-state and all memory calculations ran on CPU; P1 used 8 processes and k=4 final state used 3 processes with numerical-library thread counts fixed to 1. GPU was used only for triggered k=4 FS accountant queries. Archived P1 k=2 GPU outputs remain labeled and hashed as reused external-accountant results.
