# Hyperreals

A computational interface to infinitesimal arithmetic that records finite observations without constructing a free ultrafilter.

The mathematical question is how a finite computation can use a nonconstructive object while leaving that object unspecified. Here, each accepted comparison records a set of sequence indices that a free-ultrafilter completion must contain. Lean connects checked adaptive execution to one fixed completion and proves that a successfully extracted real value is shared by every completion of that execution. The executable core computes with exact finite periodic Laurent expressions. It does not select or enumerate whole ultrafilters.

The main entry point is `LeanResidueSystem`. The broader Python analytic and SAT frontend remains available for experiments, with a separate trust boundary. See [formalization.md](formalization.md) for the precise statements, proof sources, and research questions.

## An adaptive infinitesimal calculation

Install the Python dependencies and build the exact native checker from this checkout:

```bash
uv sync
lake build residue_checker
```

For `f(x) = x³` at `x = 2`, let a periodic-sign observation choose which literal difference quotient to evaluate. Accepting `(-1)^n < 0` selects the backward quotient. Accepting its negation selects the forward quotient. These are two separate executions:

```python
from fractions import Fraction
from hyperreals import LeanResidueSystem

for backward in (True, False):
    sys = LeanResidueSystem()
    eps, x = sys.infinitesimal(), sys.constant(2)
    accepted = sys.commit(sys.alt(), sys.constant(0), truth=backward)
    assert accepted

    if backward:
        quotient = (x**3 - (x - eps)**3) / eps
    else:
        quotient = ((x + eps)**3 - x**3) / eps

    assert quotient.standard_part() == Fraction(12)
```

At every positive index the forward quotient is `12 + 6ε + ε²`, and the backward quotient is `12 - 6ε + ε²`, with `ε = 1/n`. One is above 12 and the other below it, but both have standard part 12. The observation changes control flow, the returned expression, and the compatible completions. The ordinary answer agrees across these two executions. Each quotient also converges to 12 before any observation, so the example uses a choice to select a calculation, not to make the derivative exist. No small floating-point step or truncated expansion is used.

The [worked infinitesimal case](examples/infinitesimal/README.md) connects the positive nonzero infinitesimal denominator, exact quotients, adaptive branch selection, and standard-part result. The forward calculation also has a literal standard-part equality in Mathlib's hyperreal field. Its companion exports and kernel-checks both adaptive runs, alongside the choice-free forward quotient, its error, and that quotient after a later odd-parity choice:

```bash
uv run python scripts/infinitesimal_case.py
```

The [finite program theorem](formalization.md#finite-adaptive-programs) relates a Lean decision tree to one fixed ultrafilter throughout each accepted run. Every completion of that run reproduces its branch decisions and selected expression. Different runs need not share a completion or return the same ordinary value. The cubic program's agreement across both branches is an additional property of this example.

The Python branch code and its correspondence with that Lean tree are tested, not a formally verified translation of Python. Ordinary native calls also trust transport, parsing, native compilation, and execution. The optional replay workflow below checks a particular exported trace and expression through the Lean kernel. Neither path constructs a free ultrafilter.

## Finite observations and deferred choices

Some comparisons are already forced by the sequence semantics. Others depend on the eventual completion. The exact backend retains every compatible residue until an observation rules it out:

```python
from fractions import Fraction
from hyperreals import LeanResidueSystem

sys = LeanResidueSystem()
a = sys.alt()                              # (-1)^n
b = sys.periodic([0, 1, 2])                # table[n % 3]
zero, one = sys.constant(0), sys.constant(1)

assert sys.probe(a, zero) == (True, True)  # false and true both feasible
assert a.standard_part() is None           # remaining limits disagree
assert a < zero                            # commit to the odd indices
assert b == sys.constant(2)                # also commit to n ≡ 2 (mod 3)
assert sys.support == (False, False, False, False, False, True)
assert sys.period == 6                     # only n ≡ 5 (mod 6) remains
assert (one + (a + b) / sys.infinite()).standard_part() == Fraction(1)
assert not sys.commit(b, one, "eq", truth=True)
```

The state represents recurring residues. Committing a comparison intersects its mask with the current support over their least common multiple. Earlier choices remain in force. Lean proves that every accepted finite trace admits a free completion and that the final support characterizes all completions consistent with the actual observations. Extraction returns a rational only when every active residue has the same finite limit. Neither extraction nor `probe` makes a choice or changes the state.

The grammar includes exact rational constants, `n`, `1/n`, nonempty rational periodic tables, addition, subtraction, multiplication, and division by an explicit nonzero rational monomial. Use `expression.divide_monomial(c, k)` for division by `c * n**k`, including negative integer `k`. `/` accepts primitive constants, `n`, and `eps`. General denominators and analytic functions are outside this backend. Float inputs denote their exact binary rational values. Use `Fraction` for intended rational constants.

Comparisons expose an inclusive natural-index cutoff through `last_cutoff`. The proved comparison mask agrees with the real sequence at every index from that cutoff onward. Coefficients are never truncated. Support masks are explicitly enumerated, so coprime periods can make the state large. This is an exact reference implementation, with no claimed performance advantage and no SAT dependency in its verified core.

## Kernel-checked session replay

A snapshot captures the accepted observations, current support, query expression, and actual native answer. Later choices do not change it. Export and verification are separate operations:

```python
from hyperreals import LeanResidueSystem, verify_export

sys = LeanResidueSystem()
eps = sys.infinitesimal()
x = sys.constant(2)
accepted = sys.commit(sys.alt(), sys.constant(0), truth=True)
assert accepted
snapshot = sys.snapshot((x**3 - (x - eps)**3) / eps)
assert snapshot.result == 12
snapshot.export("replay")
verified = verify_export("replay")
```

Or verify a saved export from the repository root:

```bash
uv run python scripts/verify_replay.py replay
```

The verifier checks data/source/manifest agreement, refreshes the Lean dependency build, regenerates fixed Lean syntax, and proves the concrete trace and answer using `decide +kernel`. It audits every generated theorem for unexpected axioms. A successful rational replay certifies the standard part in every completion of the exported observations. Incorrect claimed supports or answers fail replay. An unknown replay certifies only that the extractor returned `None`, not that no individual completion can have a standard part.

The proof concerns the exported expressions and choices. Python capture and its correspondence to the intended session remain tested provenance assumptions. Hashes do not authenticate that history. Native execution need not be trusted for an answer that passes replay. Export and verification require the pinned Lean checkout. Installed packages can supply `project_root`. Replay has explicit resource limits and a configurable timeout, without imposing those bounds on ordinary native commitments.

## Further examples and supporting evidence

```bash
# Exact arithmetic and successive choices across different periods
uv run python scripts/residue_demo.py

# Earlier two-parity Laurent example
lake build laurent_checker
uv run python scripts/verified_demo.py
```

The [paired-channel calibration model](examples/delayed-choice/README.md) illustrates outputs before all phase choices are fixed. Its fused sensitivity is certified while thirty recurring phases remain possible, and a channel-specific value becomes available while ten remain. Exact symmetry assumptions make this a constructed mathematical example, not empirical sensor validation. It includes an exhaustive reference, a specified eager policy requiring backtracking, and three replayable snapshots.

The [comparative benchmark](benchmarks/README.md) records exact agreement and execution costs against direct rational residue enumeration and SymPy on twelve hand-selected workloads. These measurements document the artifact. They do not establish scalability, application coverage, or a general speed advantage.

## Other APIs

| API | Scope | Status |
| --- | --- | --- |
| `LeanResidueSystem` | Exact finite Laurent expressions with arbitrary finite periodic coefficients | Main executable core with Lean refinement proofs and optional kernel replay |
| `LeanLaurentSystem` | Same arithmetic with even and odd coefficients | Earlier verified core, built with `lake build laurent_checker` |
| `LeanPeriodicSystem` | Rational constants and alternating signs with arithmetic, no `n`, `1/n`, division, or standard parts | Smaller verified comparison core, built with `lake build periodic_checker` |
| `HyperrealSystem` | Analytic expressions, approximate series/asymptotic analysis, and SAT-assisted choices | Experimental Python frontend, not a verified implementation of the Lean semantics |

The Lean-backed APIs send unsimplified expression trees to their native checkers. Missing executables raise an error, with no fallback to Python analysis. Installed packages can pass an explicit `checker_path`. Arithmetic and comparisons between different systems are rejected.

### Experimental Python analytic and SAT frontend

```python
from hyperreals import HyperrealSystem

sys = HyperrealSystem()
eps = sys.infinitesimal()
x = sys.constant(2.0)

def f(x):
    return x * x * x - sys.constant(3.0) * x

print(((f(x + eps) - f(x)) / eps).standard_part())  # 9.0

alt = sys.alt()
assert alt < sys.constant(0.0)
assert not (alt == sys.constant(1.0))
```

Supported comparisons are checked against an exact eventual-periodic model and an infinite joint support, in addition to SAT feasibility. Unsupported comparisons return `None` through `compare_lt`, `compare_eq`, or `compare_gt`. Using them as Python booleans raises `UnderdeterminedComparisonError`. `HyperrealSystem(allow_uncertified_choices=True)` enables the legacy SAT-only policy and invalidates the state's semantic certificate.

The Python comparison compiler uses exact rational coefficients and preserves expressions when constant folding would round. That compiler remains trusted code. The eventual-periodic infinitude checker is proved sound in Lean, but the complete Python pipeline is not. `standard_part()` returns candidates from floating-point series and asymptotic analysis, not Lean proof certificates, and returns `None` when it does not recognize a limit. Supplied standard-part brackets are checked jointly before a completion is selected.

Run the experimental demo and inspect or save its state:

```bash
uv run python scripts/demo.py
uv run python scripts/demo.py --save-ultrafilter ultrafilter_state.json
uv run python scripts/demo.py --clause-limit 0
uv run python scripts/demo.py --help
```

The Taylor script reads coefficients of powers of `ε = 1/n` to recover function values and derivatives in one pass. Its floating-point expansions are outside the verified fragment:

```bash
uv run python scripts/taylor.py
uv run python scripts/taylor.py --func exp --x 0.7 --order 5
uv run python scripts/taylor.py --func sin --func cos --x 0.0 --order 5
```

The older evaluation script records implementation diagnostics from seven tasks sharing one growing state. Its counters are not independent workload measurements or evidence of SAT scalability:

```bash
uv run python scripts/eval.py
uv run python scripts/eval.py --verbose
uv run python scripts/eval.py --out results.csv
uv run python scripts/eval.py --save-ultrafilter final_state.json
```

It generates `metrics.csv`, `metrics.tex`, and `metrics.md` for the selected output stem.

## Development and proof checks

```bash
uv run pytest
uv run ruff check src/hyperreals
uv run mypy src/hyperreals
lake build
make lean-audit
uv run pytest tests/test_verified_periodic.py tests/test_verified_laurent.py
uv run pytest tests/test_verified_residue.py tests/test_replay.py tests/test_replay_capture.py
```

The audit rejects unfinished proofs and project-local axioms and permits only the standard dependencies `propext`, `Classical.choice`, and `Quot.sound`. Classical choice is part of the completion-existence argument, not an executable construction of the completion. See [formalization.md](formalization.md) for the semantic definitions, theorem map, trust boundaries, and open obligations.

| Location | Contents |
| --- | --- |
| `Hyperreals/` | Semantic specification, completion theorems, and verified executable cores |
| `src/hyperreals/verified_residue.py` and `src/hyperreals/replay.py` | Main Python adapter and snapshot replay |
| `src/hyperreals/hyperreal.py`, `sequence/`, and `series.py` | Experimental analytic frontend |
| `src/hyperreals/algebra.py`, `sat.py`, and `ultrafilter.py` | Experimental set algebra and partial ultrafilter state |
| `scripts/`, `examples/`, and `benchmarks/` | Runnable examples, replay artifacts, and bounded evaluation |

## License

MIT
