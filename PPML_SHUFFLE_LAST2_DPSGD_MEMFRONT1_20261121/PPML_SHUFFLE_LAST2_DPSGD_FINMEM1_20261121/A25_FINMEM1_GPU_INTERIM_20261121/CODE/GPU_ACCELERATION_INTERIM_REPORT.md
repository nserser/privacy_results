# FINMEM1 GPU Acceleration — Interim Engineering Snapshot

Date: 2026-11-21 UTC  
Scope: engineering validation and a backend-only rerun of the frozen Feldman–Shenfeld accountant queries. This is not a FINMEM1 scientific decision or a replacement for the frozen CPU-only specification.

## Operator override

The user explicitly enabled GPU use after the original FINMEM1 specification froze CPU-only. The historical specification and results remain unchanged. GPU work is isolated in `gpu_investigation/` and outputs are separate from the historical CPU output tree.

## What maps to GPU

The remaining unfinished FINMEM1 jobs are external random-allocation/Feldman–Shenfeld accountant queries. Their pinned `GEOM` convolution has an optional Numba CPU kernel and a NumPy Kahan-scatter fallback. The current environment uses the NumPy fallback. A deterministic CuPy output-bin gather kernel reproduces that fallback’s per-bin Kahan update order; CUDA fused multiply-add is disabled. The rest of the accountant remains the pinned CPU implementation.

## Correctness evidence so far

- Synthetic convolution cases, including stochastic-dominating and connect-the-dots projections, matched the NumPy fallback bitwise in the tested range; the largest benchmark had input dimension 16,869.
- The full M=256 accountant call (`k=2,z=2,delta=1e-3`) checked all 80 geometric-kernel calls. Every GPU output was bitwise equal to the same-process NumPy fallback; L1, max-bin, and mass discrepancies were zero.
- On the same RTX A5000 node/runtime, full CPU and GPU accountant outputs matched exactly at M=256 and M=1,024. The M=256 timings were 29.883 s CPU and 0.803 s GPU (37.2x); M=1,024 timings were 81.9995 s CPU and 1.2290 s GPU (66.7x).
- A nondeterministic atomic-scatter prototype was rejected: at M=4,096 it exceeded the library’s strict mass-repair tolerance. No tolerance was relaxed.
- Historical/login CPU results differ from the GPU-node CPU result by about 1e-6 in one M=256 query. Same-node CPU/GPU equality shows this is cross-node floating-point variation, not a GPU-specific difference. The two environments must not be silently mixed for borderline comparisons.

## Current batch

Slurm job 930 is running two independent GPU workers on RTX A5000 devices. It reruns the full frozen 40-query accountant manifest into `RESULTS/fs_gpu_results/outputs/`, preserving prior CPU files. At snapshot creation, 17 of 40 outputs were complete. Per-worker JSONL logs and the expected manifest are included. The active queries are P2 at M=8,192 and M=16,384, both with delta 1e-3. Larger/high-accuracy queries are materially slower; total time depends on delta/refinement, not just M.

## Adoption guidance

The deterministic gather backend is a viable accelerator candidate for this pinned GEOM accountant. A full scientific package should use it only with the exact source commit locked, CPU fallback retained, pre-run same-node equivalence tests, deterministic per-query IDs, per-query progress and atomic outputs, and a final exact expected-vs-completed job audit. Keep the backend override explicit in provenance. Do not use atomic scatter or relax mass/error checks. The scientific conclusion remains pending until the complete query set and downstream independent audit finish.
