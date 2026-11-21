# MEMFRONT2 final report

## Decision

**REPAIR_MEMFRONT2.** The spectral closure gate failed on both primary lines. No final-state privacy, Feldman–Shenfeld, packet-size, or initialization experiment was run. The frozen protocol requires this stop.

## Results and interpretation

- P1 (eta=.10,z=2): BASE/DOMAIN dominant slow tau is about 1.724e8 steps, with L1 mode amplitude 3.95e-7. GRID gives tau 3.772e8 and GRID2 4.658e8 (amplitude decreases to 1.58e-7). The fitted within-grid tail predicts a completed L_TV around 37,157 on BASE/DOMAIN, 39,321 on GRID, and 39,902 on GRID2. These are **diagnostic point estimates**, not certified memory closures. BASE/GRID/DOMAIN L_TV spread is over 5%, and tau is highly grid-sensitive. Therefore the old finite-horizon MEMFRONT1 L_TV=221.505 does not survive an infinite-tail completion on the BASE discretization; the discrepancy is attributable to a tiny-amplitude, extremely slow discrete mode, whose numerical stability is not adequate for scientific closure.
- P2 (eta=.10,z=4): BASE/DOMAIN slow tau is about 15,045.5, amplitude 1.130e-3; GRID gives tau 15,859.2, amplitude 1.090e-3; GRID2 gives tau 16,073.4, amplitude 1.080e-3. Completed diagnostic L_TV estimates are 8,588.4 (BASE/DOMAIN), 8,771.6 (GRID), and 8,818.4 (GRID2). BASE/GRID/DOMAIN L_TV spread is about 2.1%, but BASE-to-GRID tau variation is about 5.4%, failing the separate 5% spectral time-constant condition. P2 is therefore not `SPECTRALLY_CLOSED`.
- Held-out direct-vs-spectral reconstructions on each grid passed the 5% pointwise criterion for P1/P2; observed maximum errors are in `SPECTRAL_RECONSTRUCTION_VALIDATION.csv`. This validates the retained-mode reconstruction over the checked intervals on each discretization; it does **not** cure cross-grid disagreement.
- The selected leading modes have small amplitudes, especially P1. The dominant slow-mode residuals are near machine precision, but higher modes in the 32-mode basis have left-eigenpair residuals above 1e-9 (up to about 1.94e-3 on the primary grids). All residuals and conditioning indicators are reported; tail-completion estimates are diagnostic, not formal certified bounds. Cumulative escape was tracked and no probability renormalization was used.
- The secondary lines are an exploratory memory map only. Where direct responses hit a numerical floor, relative reconstruction error is large; those lines are explicitly marked unvalidated.

## Answers to the frozen questions

1. P1 is not robustly 221.5 after spectral tail completion; the BASE diagnostic is roughly 37.2k, but the closure is grid-sensitive and uncertified.
2. P2 has a slow mode near tau 15.0k–16.1k with L1 amplitude 0.00108–0.00113, but it fails the required cross-grid tau stability. Higher modes in the retained 32-mode basis also exceed the 1e-9 left-eigenpair residual target.
3. Completed P2 L_TV diagnostic range is about 8.59k–8.82k across tested grids; this is not an authorized certified estimate.
4. P1 final-state decay at x>=4 or x>=8 was not tested.
5. No P1 Feldman–Shenfeld comparison was run.
6. P2 final-state frontier was not run; practical status remains unresolved.
7. No collapse assessment is possible without final-state measurements.
8. The computed modes suggest multiple scales, but P1’s very slow mode is grid sensitive; a scientific claim of a validated spectrum is not supported yet.
9. k=4 was not run.
10. Initialization sensitivity was not run.
11. Secondary estimates vary widely and several fail direct/spectral held-out checks; no short/moderate/long/extreme memory classification is certified.
12. Final verdict is **REPAIR_MEMFRONT2**, not GO or STOP.

## Provenance and execution

M0 passed: input checksum manifest verified, 22 core tests passed, and deterministic one-worker/four-worker equivalence passed exactly. Six CPU workers were used for independent primary spectral tasks, two for GRID2, and five for secondary lines. No GPU was requested or used. Historical SCALECERT and MEMFRONT1 decisions remain unchanged.
