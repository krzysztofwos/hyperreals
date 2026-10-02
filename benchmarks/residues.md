# Exact finite-period Laurent benchmark

Measured 2026-10-01T08:47:39.216121+00:00 on macOS-26.5.1-arm64-arm-64bit-Mach-O (arm64) with Python 3.13.13 and leanprover/lean4:v4.33.0. Repetitions per backend/workload: 20.

Reproduce: `lake build residue_checker && uv run --extra benchmark python scripts/benchmark_residues.py --repeat 20 --output benchmarks/residues`.

Each row passed exact decision, final-support, and rational-standard-part agreement before timing. Each backend's reported comparison cutoffs were also checked by direct Fraction evaluation on every residue at two later indices. Those finite checks supplement the Lean theorems. They are not a proof of the independent baselines.

The Lean process remains alive across an untimed validation/warmup and all repetitions. Its steady timings include Python JSON serialization, pipe transport, native parsing/computation, and JSON decoding. Fraction and SymPy timings are direct Python calls including normalization, period enumeration, cutoff calculation, state refinement, and limit extraction. All cases begin with a fresh logical universe support. Each backend uses its own exact sufficient cutoff. Cutoffs need not be numerically identical.

The public LeanResidueSystem wrapper currently launches a checker process for every request. This benchmark's persistent session measures a different transport configuration. Public API end-to-end latency additionally includes repeated process launches. Expected mathematical results and cross-backend agreement are checked outside each timing sample. Native response validation and common result packaging remain inside.

Startup is measured separately for native process launch through a readiness request, SymPy import and symbol initialization, or Fraction-backend construction. SymPy's process-local caches remain enabled after warmup. The measured implementations provide different assurance and have different transport costs, so these numbers are not a claim about the performance cost of formal verification.

Peak RSS is the operating-system high-water mark for a fresh backend/workload worker, including interpreter/imports, reference validation, warmup, and repetitions. For Lean, runtime RSS instead reports the single native child and transport RSS separately reports its Python worker. macOS reports ru_maxrss in bytes. Linux reports it in KiB. Unsupported platforms emit null. These are process-lifetime peaks, not per-operation allocations or directly interchangeable memory scopes.

Expression nodes count serialized AST occurrences across the workload's query operands and extracted expression. Coefficient metrics describe the largest final normalized branch among those expressions, not intermediate allocation peaks. Dense periods are not minimized.

A standard-part result of `unknown` means the active residues include unequal finite limits or at least one divergent branch. Extraction leaves the support unchanged. Unknown is not treated as a numeric answer or a benchmark failure.

| Workload | Backend | Median µs | Min–max µs | Startup ms | Runtime peak MiB | Transport peak MiB | Period / active | Standard part |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| infinitesimal_order | fraction | 86.33 | 35.25–194.04 | 0.00 | 24.45 | — | 1 / 1 | 12 |
| infinitesimal_order | lean_native | 207.08 | 83.33–487.75 | 114.89 | 35.97 | 24.62 | 1 / 1 | 12 |
| infinitesimal_order | sympy | 1506.65 | 859.04–3616.88 | 443.44 | 62.08 | — | 1 / 1 | 12 |
| coprime_2_3 | fraction | 21.52 | 19.00–133.29 | 0.00 | 24.30 | — | 6 / 1 | 5 |
| coprime_2_3 | lean_native | 212.38 | 59.92–400.79 | 54.94 | 35.97 | 24.55 | 6 / 1 | 5 |
| coprime_2_3 | sympy | 569.54 | 173.29–1082.29 | 331.60 | 59.55 | — | 6 / 1 | 5 |
| coprime_3_5_7 | fraction | 68.58 | 66.79–1536.83 | 0.00 | 24.45 | — | 105 / 1 | 9 |
| coprime_3_5_7 | lean_native | 1027.96 | 377.92–2314.46 | 45.74 | 36.08 | 24.64 | 105 / 1 | 9 |
| coprime_3_5_7 | sympy | 1034.54 | 567.62–2227.75 | 333.80 | 59.55 | — | 105 / 1 | 9 |
| shared_factors_4_6 | fraction | 59.38 | 58.62–251.62 | 0.00 | 24.39 | — | 12 / 1 | 9 |
| shared_factors_4_6 | lean_native | 260.62 | 118.25–487.46 | 38.57 | 36.00 | 24.62 | 12 / 1 | 9 |
| shared_factors_4_6 | sympy | 1001.23 | 575.38–2347.83 | 274.47 | 60.09 | — | 12 / 1 | 9 |
| support_relative_limit | fraction | 50.19 | 19.08–105.67 | 0.00 | 24.66 | — | 2 / 1 | -1 |
| support_relative_limit | lean_native | 154.83 | 76.54–298.12 | 54.16 | 35.92 | 24.64 | 2 / 1 | -1 |
| support_relative_limit | sympy | 321.83 | 213.33–1132.08 | 590.80 | 56.86 | — | 2 / 1 | -1 |
| unknown_differing_limits | fraction | 10.81 | 10.54–15.21 | 0.00 | 24.45 | — | 2 / 1 | unknown |
| unknown_differing_limits | lean_native | 73.62 | 35.12–171.38 | 49.08 | 36.03 | 24.41 | 2 / 1 | unknown |
| unknown_differing_limits | sympy | 225.48 | 103.92–948.42 | 281.64 | 59.81 | — | 2 / 1 | unknown |
| unknown_divergent_branch | fraction | 18.00 | 17.08–244.21 | 0.00 | 24.33 | — | 3 / 1 | unknown |
| unknown_divergent_branch | lean_native | 666.29 | 149.12–1926.00 | 75.38 | 35.94 | 24.47 | 3 / 1 | unknown |
| unknown_divergent_branch | sympy | 634.62 | 203.71–1269.08 | 893.16 | 55.41 | — | 3 / 1 | unknown |
| laurent_growth_4 | fraction | 65.52 | 61.38–243.54 | 0.00 | 24.53 | — | 1 / 1 | 1 |
| laurent_growth_4 | lean_native | 551.79 | 343.29–1191.92 | 190.34 | 35.94 | 24.50 | 1 / 1 | 1 |
| laurent_growth_4 | sympy | 5261.69 | 2117.21–9399.67 | 355.33 | 61.95 | — | 1 / 1 | 1 |
| laurent_growth_8 | fraction | 396.44 | 295.71–1051.75 | 0.00 | 24.30 | — | 1 / 1 | 1 |
| laurent_growth_8 | lean_native | 726.40 | 285.25–1839.00 | 52.00 | 35.97 | 24.69 | 1 / 1 | 1 |
| laurent_growth_8 | sympy | 6888.69 | 3257.17–19922.25 | 410.82 | 62.52 | — | 1 / 1 | 1 |
| laurent_growth_12 | fraction | 425.04 | 288.79–1382.17 | 0.00 | 24.19 | — | 1 / 1 | 1 |
| laurent_growth_12 | lean_native | 953.81 | 520.96–1972.42 | 47.42 | 36.05 | 24.70 | 1 / 1 | 1 |
| laurent_growth_12 | sympy | 7217.17 | 4800.25–16002.29 | 499.36 | 62.25 | — | 1 / 1 | 1 |
| exact_cancellation_12 | fraction | 823.77 | 535.67–2625.46 | 0.00 | 24.50 | — | 1 / 1 | 7/3 |
| exact_cancellation_12 | lean_native | 2546.10 | 1652.71–4244.08 | 74.87 | 36.03 | 24.44 | 1 / 1 | 7/3 |
| exact_cancellation_12 | sympy | 600.73 | 216.25–1501.58 | 444.79 | 61.86 | — | 1 / 1 | 7/3 |
| dense_lcm_210 | fraction | 3565.69 | 2635.00–6495.96 | 0.00 | 24.50 | — | 210 / 105 | 0 |
| dense_lcm_210 | lean_native | 1773.35 | 1375.46–3103.71 | 44.91 | 36.17 | 24.70 | 210 / 105 | 0 |
| dense_lcm_210 | sympy | 27781.60 | 18459.33–134780.54 | 224.59 | 60.42 | — | 210 / 105 | 0 |

Native executable SHA-256: `03a0f449720e3b3eb9154674d47b773370485befa3caf52b49483ed4ccdcfd83`. Full timing samples, outputs, dependency versions, and structural metrics are in the adjacent JSON/CSV files.
