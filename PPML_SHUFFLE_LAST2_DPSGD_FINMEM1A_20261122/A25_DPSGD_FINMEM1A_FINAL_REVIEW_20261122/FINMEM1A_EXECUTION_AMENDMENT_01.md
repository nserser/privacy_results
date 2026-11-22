# FINMEM1A Execution Amendment 01

Frozen 2026-11-22. This amendment records a source-state discrepancy discovered during M0: the prior FINMEM1 working tree contains a complete 32/32 final-state batch, P1 finite-memory rows, and P1 numerical-convergence anchors, although the FINMEM1A prompt describes the P1 final-state curve as missing. These outputs will not be silently relabeled as newly run FINMEM1A evidence.

1. Historical P1 final-state/memory/convergence artifacts were found locally but were not silently relabeled as FINMEM1A results. FINMEM1A regenerated the P1 memory table, all seven BASE final-state horizons, and the required 3x3 convergence anchors in its separate branch.
2. The 14 P1 k=2 Feldman–Sheneld outputs were reused exactly from the immutable 19/40 snapshot after ID, configuration, bound, commit, and checksum validation. No completed P1 k=2 accountant query was recomputed.
3. The cancelled Slurm job 930 was the superseded broad GPU accountant sweep. Its 21 partial outputs remain preserved in the old FINMEM1 GPU investigation tree and are not used to replace the canonical 19-query archived snapshot.
4. P2 FS and k=1 FS were not run. k=4 final-state jobs were triggered only after the P1 k=2 gates passed. Any new k=4 FS queries are allowed only after same-node parity and must stop after two consecutive both-delta strict wins.
5. No threshold, mechanism, allocation law, or scientific parameter was changed.

This is an auditable execution clarification; it does not amend scientific gates.
