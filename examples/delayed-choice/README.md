# Delayed phase choices in a paired-channel calibration model

This worked example shows a concrete benefit of retaining unresolved choices: an invariant output can be obtained before choosing a recurring phase, and later compatible evidence can refine the phase without retracting an arbitrary earlier choice. It is an illustrative exact model, not an empirically validated instrument, an inference algorithm for real sensor data, or a claim that delayed choice is generally faster.

## Model and mathematical scope

Let `a(n)`, `b(n)`, and `c(n)` be rational periodic tables of lengths 4, 6, and 5. The two modeled responses at the same index are

\[
F_+(x,n)=x^3+b(n)x+a(n),\qquad F_-(x,n)=x^3-b(n)x+c(n).
\]

The opposite biases and shared cubic response are modeling assumptions. The script uses `a = [1/3, -2/5, 7/4, 0]`, `b = [3, -2, 5, 1, -4, 7]`, and `c = [0, 1/7, -2/3, 5/2, -1/4]`. All arithmetic is exact rational arithmetic. The common phase space has 60 residues.

At `x = 2` and `ε = 1/n`, take same-phase finite differences `D± = (F±(2+ε,n) - F±(2,n))/ε`. Their exact expressions at positive indices simplify to

\[
D_+=12+b(n)+6\varepsilon+\varepsilon^2,\qquad D_-=12-b(n)+6\varepsilon+\varepsilon^2.
\]

The offsets cancel within each difference. Averaging the channels cancels the unresolved bias, giving `(D+ + D−)/2 = 12 + 6ε + ε²`, whose standard part is 12 on every residue. An individual channel has standard part `12 + b(r)` after a residue is selected. Without sufficient phase information, the values can disagree.

An observation here restricts the recurring index classes a free-ultrafilter completion may select. It does not assert that `n mod 60` is constant at all indices, and a finite list of empirical readings does not establish these constraints. The application is a symbolic model of conditioning on periodic operating modes. Its connection to any physical device would need a separate justification.

## Why preserve the alternatives?

The following compatible observations arrive incrementally:

| Observation      | Stored support period / active residues | Compatible phases modulo 60 | Raw standard part | Fused standard part |
| ---------------- | --------------------------------------: | --------------------------: | ----------------: | ------------------: |
| None             |                                   1 / 1 |                          60 |           unknown |                  12 |
| `n mod 4` is odd |                                   4 / 2 |                          30 |           unknown |                  12 |
| `n mod 6 = 5`    |                                  12 / 2 |                          10 |                19 |                  12 |
| `n mod 5 = 3`    |                                  60 / 2 |                           2 |                19 |                  12 |
| `n mod 4 = 3`    |                                  60 / 1 |        1, namely residue 23 |                19 |                  12 |

After the coarse observation, the fused sensitivity is already available while thirty phases remain possible. After the shared-clock observation, the individual channel's standard part is 19 while ten phases remain possible. Solving every phase choice is unnecessary for either output. The exact finite-difference syntax, rather than a hand-supplied limit assertion, is sent to the Lean extractor.

For comparison, the eager policy selects the smallest full-period residue compatible with the first observation: residue 1. That extra choice is mathematically consistent. It also excludes the later observation `n mod 6 = 5`, so this policy must stop or backtrack. Its rejection is sound relative to its chosen branch. It is not a bug and does not show that every eager algorithm fails. Retaining all compatible residues avoids this particular need for backtracking. An exhaustive enumeration of the 60 phases makes the same accepted-evidence decisions as the delayed system and ends at residue 23.

Ordinary symbolic algebra and finite constraint sets can produce the same outputs. The example demonstrates obtaining those outputs through the verified completion interface while retaining unresolved phase alternatives. It does not establish a computational capability unavailable to those other methods.

## Reproduce the measurements and proof artifacts

From the repository root:

```sh
lake build residue_checker Hyperreals.ResidueReplay
uv run python scripts/delayed_choice_case_study.py --repeat 5
```

The script writes [results.json](results.json), [results.md](results.md), and three snapshot directories. It exits unsuccessfully if exact reference comparisons disagree, replay fails, or generated source/data hashes do not match the checked snapshot. `--skip-replay` measures the native and Python paths without claiming kernel verification. `--replay-only` adds freshly checked snapshots to an existing measurement report without repeating the timings. Kernel replay can take longer than native execution. Its timeout defaults to 180 seconds per snapshot.

The snapshots capture these particular points in the actual session:

| Snapshot                                         | Accepted observations at capture | Extracted result | Remaining phases modulo 60 |
| ------------------------------------------------ | -------------------------------: | ---------------: | -------------------------: |
| [early-fused](snapshots/early-fused/Replay.lean) |                                1 |               12 |                         30 |
| [partial-raw](snapshots/partial-raw/Replay.lean) |                                2 |               19 |                         10 |
| [final-phase](snapshots/final-phase/Replay.lean) |                                4 |               23 |                          1 |

The early snapshots are immutable and are replayed only after the session has reached its final state. Each export contains `snapshot.json`, generated `Replay.lean`, and `manifest.json`. Successful replay checks the actual finite trace, final support, and coefficient-extractor result by Lean kernel reduction, then applies the general all-compatible-completions theorem. The generated theorem says that the returned rational is the standard part in every free completion of that snapshot's observations. It does not prove that these observations occurred in a physical experiment or that exported data honestly records an external history. Hashes bind the artifact components and source provenance. They are not signatures authenticating that history.

## What the comparisons measure

Four implementations are timed after one untimed warmup: the public `LeanResidueSystem` delayed-support path, independent Python dynamic Boolean masks, a Python eager single-phase policy with no backtracking, and an exhaustive Python 60-phase reference. The native path builds and normalizes the original paired-polynomial syntax, while the Python policies use the independently expanded model formulas. All implementations enumerate candidates for reporting. Mathematical outcomes are checked outside each timing sample.

The native timings include a new checker process for every public API request, JSON transport, exact expression normalization, support updates, extraction, and stage reporting. They do not use the persistent transport configuration of the separate residue benchmark. The Python paths have no equivalent transport or general expression-normalization cost. The eager path stops after two attempted observations and therefore does less work than the four-observation paths. These timing scopes prevent a general algorithmic performance comparison. Accepted-evidence counts and retained phases explain the behavioral difference. Full verification wall time is measured once per snapshot and reported separately. It includes the dependency-freshness build, Lean startup and elaboration, kernel checking, and axiom audit. Existing build artifacts are reused, so this is not a clean-build measurement.

The independent companion checks all 60 phase candidates for every observation and compares complete candidate sets, not just their sizes. It also checks the finite-difference identities with `Fraction` at three positive indices in every residue class (180 sampled indices). That finite evaluation is corroborating evidence, not an all-indices convergence proof. The generated Lean replay supplies the standard-part guarantee for the captured exact expressions under the formal development's audited axioms.
