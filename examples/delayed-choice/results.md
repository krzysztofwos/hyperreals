# Measured delayed-choice case study

Generated 2026-10-02T12:19:17.617116+00:00 on macOS-26.5.1-arm64-arm-64bit-Mach-O with Python 3.13.13. Each implementation ran 5 repetitions after one warmup.

| Implementation | Median ms | Min–max ms | Accepted / attempted evidence | Remaining phases |
|---|---:|---:|---:|---:|
| lean_delayed_public_api | 873.9144 | 586.0103–938.9996 | 4 / 4 | 1 |
| python_delayed_mask | 0.0732 | 0.0698–0.1872 | 4 / 4 | 1 |
| python_eager_single_phase | 0.0363 | 0.0322–0.1871 | 1 / 2 | 1 |
| python_exhaustive_reference | 0.0897 | 0.0540–0.1145 | 4 / 4 | 1 |

The native row uses the public per-request wrapper, including checker launches, syntax construction, normalization, JSON transport, and stage reporting. The Python rows use exact expanded model limits, without general expression normalization or native transport. All rows enumerate the 60 candidate phases for reporting. The eager row stops after its second attempted observation, so it performs less work. These are implementation costs, not evidence of a general speed advantage for any choice strategy.

## Evidence and outcomes

| Stage | Stored mask period / active | Compatible phases out of 60 | Raw standard part | Fused standard part |
|---|---:|---:|---:|---:|
| initial | 1 / 1 | 60 | unknown | 12 |
| coarse: phase mod 4 is odd | 4 / 2 | 30 | unknown | 12 |
| shared clock: phase mod 6 = 5 | 12 / 2 | 10 | 19 | 12 |
| independent clock: phase mod 5 = 3 | 60 / 2 | 2 | 19 | 12 |
| fine phase: phase mod 4 = 3 | 60 / 1 | 1 | 19 | 12 |

The no-backtracking eager policy chooses residue 1 after the coarse observation. The later condition n mod 6 = 5 is compatible with the original evidence but not with that extra commitment. Its rejection is sound relative to its chosen branch. Accepting all supplied evidence would require backtracking. The retained support and exhaustive reference instead end at residue 23 modulo 60.

## Kernel replay

The early snapshots preceded later state changes. All three were checked after the full trace completed. Full verification wall times include the dependency-freshness build, Lean startup/elaboration, kernel checking, and axiom audit. They are measured once with existing build artifacts, separately from native execution timings.

| Snapshot | Full verification seconds |
|---|---:|
| [early-fused](snapshots/early-fused/Replay.lean) | 25.844 |
| [partial-raw](snapshots/partial-raw/Replay.lean) | 16.046 |
| [final-phase](snapshots/final-phase/Replay.lean) | 13.460 |

Full checker output, axiom reports, hashes, and timing samples are retained in results.json and the snapshot manifests.
