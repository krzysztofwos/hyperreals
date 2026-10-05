# Hyperreals

A computational interface to infinitesimal arithmetic that records finite observations without constructing a free ultrafilter.

The mathematical question is how a finite computation can use a nonconstructive object while leaving that object unspecified. Here, each accepted comparison records a set of sequence indices that a free-ultrafilter completion must contain. Lean connects checked adaptive execution to one fixed completion. Within the exact fragment, extraction succeeds exactly when every compatible completion has the same finite real standard part. Generic polynomial differentiation connects that executable result to the ordinary derivative. The complete extraction core computes with exact finite periodic Laurent expressions. A separate symbolic compiler covers finite real vectors and elementary functions, with proved domains and literal infinitesimal quotient semantics. It does not select or enumerate whole ultrafilters.

`LeanResidueSystem` provides exact sequence arithmetic and finite observations. `DifferentiableProgram` provides symbolic elementary vector differentiation. Kernel checks verify exported residue computations and the derivative compiler's output. See [formalization.md](formalization.md) for the precise statements, proof sources, and research questions.

## Generic infinitesimal differentiation

Python 3.14 is the supported version, pinned in `.python-version` for local development and CI. Install the locked dependencies and build the exact native checker from this checkout:

```bash
uv sync --locked --extra dev
lake build residue_checker
```

For any rational polynomial `P`, rational point `a`, nonzero rational `c`, and positive integer `k`, the literal quotient `(P(a + c/n**k) - P(a)) / (c/n**k)` extracts to `P′(a)` on every nonempty residue support. The compiler accepts coefficients, a point, and a step. It receives no expansion or derivative certificate.

```python
from fractions import Fraction
from hyperreals import LeanResidueSystem, polynomial_quotient

sys = LeanResidueSystem()
# P(x) = 3/2 - 2x + x^4. Coefficients are in ascending degree order.
quotient = polynomial_quotient(
    [Fraction(3, 2), -2, 0, 0, 1], Fraction(3, 2), -3, 2, system=sys
)
assert quotient.standard_part() == Fraction(23, 2)
```

[PolynomialDifferentiation.lean](Hyperreals/PolynomialDifferentiation.lean) proves source denotation, ordinary differentiation, and actual extraction for all such inputs. `evaluate_polynomial(coefficients, argument)` constructs exact Horner syntax. `divided_difference(coefficients, a, increment)` constructs a polynomial `D` satisfying `h * D = P(a+h) - P(a)` for an arbitrary representable increment `h`. Whenever `h` has extracted standard part zero in the current state, `D` extracts to the derivative. It equals the literal quotient when `h` is nonzero. At a zero increment it is a polynomial extension, not permission to cancel zero.

## Verified elementary differentiation

A second language supports rational constants, finite vectors, addition, subtraction, multiplication, general division, negation, and `sin`, `cos`, `exp`, `log`, and `sqrt`. It compiles exact symbolic Jacobian-vector products (JVPs), Jacobians, and scalar gradients. Results such as `cos(1)` remain expressions.

```python
from hyperreals import DifferentiableProgram, variables

x, y = variables(2)
program = DifferentiableProgram(2, (x.sin() * y.exp(), (1 + x*x + y*y).log()))
compiled = program.jvp((1, 2))
verified = compiled.verify(point=(1, 0), timeout=180)
assert verified.domain_verified
# Approximate samples only. The exact result remains compiled.derivative.
print(compiled.derivative.approximate((1, 0)))  # (2.223244275..., 1.0)
jacobian = program.jacobian()  # rows are outputs, columns are inputs
```

Lean proves that the generated JVP evaluates to `DF(x)v`. For any compatible completion and any step `h` that is infinitesimal and eventually nonzero in that completion, it also proves `st((F(x + h v) - F(x)) / h) = DF(x)v`, componentwise. The source domain remains valid under sufficiently small perturbations. The [vector example](examples/differentiable/README.md) combines this theorem with the observation-based choice of a nonzero step.

Division requires a nonzero denominator. Logarithm and square root require positive arguments. `program.domain_conditions` exposes these obligations without discharging them. `compiled.verify()` checks the actual Python compiler output against Lean and proves correctness conditional on the domain. Supplying an exact rational `point` additionally asks Lean to prove the source, tangent, and derivative domains there. This bounded proof procedure can fail on a valid domain. Failure means unproved, not “no shared standard part.” `approximate()` uses floating-point arithmetic and is never a certificate. Constants accept `int` or `Fraction`.

This is forward symbolic differentiation, with no reverse-mode implementation, tensor accelerator, arbitrary Python tracing, or complete elementary-function equality/limit solver. The Laurent backend retains its separate complete decision procedure. Run `uv run python scripts/differentiable_case.py` for a kernel-checked example.

## An observation that establishes a division condition

Let `h(n)` be zero at even indices and `1/n` at odd indices. Its standard part is zero before any choice, but that does not establish invertibility. A checked observation of `h = 0` determines which calculation is legitimate:

```text
if observe(h = 0):
    use the fallback step 1/n²
else:
    use h, with the represented reciprocal periodic([0, 1]) * n
return the corresponding cubic quotient at 2
```

[DomainInfinitesimal.lean](Hyperreals/DomainInfinitesimal.lean) proves that the negative observation makes this represented reciprocal an actual inverse in every completion of that branch. The zero branch rules out any inverse to the original step and uses the fallback instead. Both accepted executions use a nonzero infinitesimal and extract 12. Before refinement, multiplying the original numerator by the represented reciprocal has limits 0 and 12 on the two residue classes, so it has no shared standard part. The example uses the residue language's multiplication and monomial division. Its represented reciprocal is justified only on the selected support.

```bash
uv run python scripts/domain_infinitesimal_case.py
```

The companion exports and kernel-checks the two accepted branches and the unrefined rejection. The formal [finite program theorem](formalization.md#finite-adaptive-programs) keeps one fixed completion throughout each accepted run.

## A simpler adaptive calculation

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

The state represents recurring residues. Committing a comparison intersects its mask with the current support over their least common multiple. Earlier choices remain in force. Lean proves that every accepted finite trace admits a free completion and that the final support characterizes all completions consistent with the actual observations. Extraction returns a rational exactly when every compatible completion has the same finite real standard part. Every shared value in this fragment is rational. A `None` result means that no one finite real value works for all retained completions, although an individual completion or later refinement may still have a standard part. Neither extraction nor `probe` makes a choice or changes the state.

`expression.explain_standard_part()` returns a `StandardPartDiagnostic` with `kind`, `period`, `residues`, and `limits`. A `finite` result carries the common rational. A `divergent` result identifies an active residue whose magnitude tends to infinity. A `disagreement` result gives two active residues and their different finite rational limits. Residues use the returned common period. Lean proves the diagnostic cases against the same exact normalization used by extraction.

```python
sys = LeanResidueSystem()
assert sys.alt().explain_standard_part().kind == "disagreement"
assert sys.infinite().explain_standard_part().kind == "divergent"
```

The grammar includes exact rational constants, `n`, `1/n`, nonempty rational periodic tables, addition, subtraction, multiplication, and division by an explicit nonzero rational monomial. Use `expression.divide_monomial(c, k)` for division by `c * n**k`, including negative integer `k`. `/` accepts primitive constants, `n`, and `eps`. General denominators and analytic functions are outside this backend. Float inputs denote their exact binary rational values. Use `Fraction` for intended rational constants.

Comparisons expose an inclusive natural-index cutoff through `last_cutoff`. The proved comparison mask agrees with the real sequence at every index from that cutoff onward. Coefficients are never truncated. Support masks are explicitly enumerated, so coprime periods can make the state large. This is an exact reference implementation, with no claimed performance advantage.

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

The verifier checks data/source/manifest agreement, refreshes the Lean dependency build, regenerates fixed Lean syntax, and proves the concrete trace and answer using `decide +kernel`. It audits every generated theorem for unexpected axioms. A successful rational replay certifies the standard part in every completion of the exported observations. Incorrect claimed supports or answers fail replay. A replay of `None` additionally certifies that no finite real value is shared by every completion of the exported trace. This does not rule out standard parts in individual completions or after further compatible observations. Replay checks the query result. The diagnostic witness fields are not part of its snapshot format.

The proof concerns the exported expressions and choices. Python capture and its correspondence to the intended session remain tested provenance assumptions. Hashes do not authenticate that history. Native execution need not be trusted for an answer that passes replay. Export and verification require the pinned Lean checkout. Installed packages can supply `project_root`. Replay has explicit resource limits and a configurable timeout, without imposing those bounds on ordinary native commitments.

## Further examples and supporting evidence

```bash
# Exact arithmetic and successive choices across different periods
uv run python scripts/residue_demo.py
```

The [paired-channel calibration model](examples/delayed-choice/README.md) illustrates outputs before all phase choices are fixed. Its fused sensitivity is certified while thirty recurring phases remain possible, and a channel-specific value becomes available while ten remain. Exact symmetry assumptions make this a constructed mathematical example, not empirical sensor validation. It includes an exhaustive reference, a specified eager policy requiring backtracking, and three replayable snapshots.

The [comparative benchmark](benchmarks/README.md) records exact agreement and execution costs against direct rational residue enumeration and SymPy on twelve hand-selected workloads. These measurements document the artifact. They do not establish scalability, application coverage, or a general speed advantage.

## Verified interfaces

| API                     | Computation                                                                                              | Verification                                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `LeanResidueSystem`     | Exact periodic Laurent arithmetic, comparison observations, and complete shared standard-part extraction | `lake build residue_checker` for native execution. `verify_export()` kernel-checks a saved trace and query. |
| `DifferentiableProgram` | Symbolic elementary vector JVPs, Jacobians, and scalar gradients on explicit domains                     | `CompiledJVP.verify()` kernel-checks compiler output and its derivative and quotient theorems.              |

The residue API sends unsimplified expression trees to its native checker. A missing executable raises an error. Installed packages can pass an explicit `checker_path`. Arithmetic and comparisons between different systems are rejected. The differentiable API builds dimension-checked symbolic expressions and generates Lean proof obligations when verification is requested. Both kernel-verification paths require the pinned Lean checkout and accept an explicit `project_root`.

## Development and proof checks

```bash
uv run pytest
uv run ruff check src/hyperreals
uv run mypy src/hyperreals
lake build
make lean-audit
uv run pytest tests/test_verified_residue.py tests/test_replay.py tests/test_replay_capture.py
uv run pytest tests/test_infinitesimal_case.py tests/test_domain_infinitesimal_case.py
```

The audit rejects unfinished proofs and project-local axioms and permits only the standard dependencies `propext`, `Classical.choice`, and `Quot.sound`. Classical choice is part of the completion-existence argument, not an executable construction of the completion. See [formalization.md](formalization.md) for the semantic definitions, theorem map, trust boundaries, and open obligations.

| Location                                                            | Contents                                                                   |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `Hyperreals/`                                                       | Semantic specification, completion theorems, and verified executable cores |
| `src/hyperreals/verified_residue.py` and `src/hyperreals/replay.py` | Residue arithmetic, observations, and snapshot replay                      |
| `src/hyperreals/differentiable.py`                                  | Symbolic elementary/vector derivative compiler and kernel verification     |
| `Hyperreals/Differentiable*.lean`                                   | Compiler correctness, domain preservation, and literal quotient limits     |
| `src/hyperreals/polynomial.py`                                      | Exact polynomial and divided-difference syntax constructors                |
| `scripts/`, `examples/`, and `benchmarks/`                          | Runnable examples, replay artifacts, and bounded evaluation                |

## License

MIT
