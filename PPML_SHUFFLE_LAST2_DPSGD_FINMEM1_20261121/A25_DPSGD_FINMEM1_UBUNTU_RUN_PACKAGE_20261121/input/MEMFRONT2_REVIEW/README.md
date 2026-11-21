# PPML-SHUFFLE-LAST2-DPSGD-MEMFRONT2 review package

Terminal status: `REPAIR_MEMFRONT2`.

M0 verification passed and the independently reconstructed one-shock memory spectra are packaged with direct trajectories. The primary P1 and P2 lines do not satisfy the frozen cross-grid spectral-closure gate, so final-state scaling, Feldman–Shenfeld comparison, user-packet robustness, and initialization analyses were blocked. No GPU was requested or used.

Start with:

- `FINAL_MEMFRONT2_REPORT.md`
- `MEMFRONT2_DECISION.json`
- `INDEPENDENT_AUDIT.json`
- `MEMORY_SUMMARY_FINAL.csv`
- `SPECTRAL_RECONSTRUCTION_VALIDATION.csv`
- `NUMERICAL_CONVERGENCE_MEMORY.csv`

`RAW_SPECTRAL/` contains the saved metadata and NumPy arrays for all 13 spectral/direct-memory jobs. `MEMORY_RESPONSE_DIRECT.csv` contains the per-step direct response, conservative TV upper, escape accounting, and spectral predictions on evaluated stride points. Diagnostic completed-memory estimates are not certified closures.

The original SCALECERT and MEMFRONT1 decisions remain unchanged. The full frozen implementation source, MEMFRONT2 orchestration, raw outputs, logs, figures, and SHA-256 manifest are included.
