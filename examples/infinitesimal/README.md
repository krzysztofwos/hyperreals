# An exact infinitesimal difference quotient

Take `epsilon = 1/n`, `f(x) = x^3`, and `x = 2`. The increment `dx = epsilon` is positive and nonzero in every free-ultrafilter completion, while its standard part is zero. The difference `dy = (2 + epsilon)^3 - 8` gives the exact quotient

```text
dy/dx = 12 + 6*epsilon + epsilon^2
st(dy/dx) = 12
st(dy/dx - 12) = 0
```

The quotient is strictly greater than `12` at every positive index. Its difference from the ordinary derivative is a nonzero infinitesimal. Taking the standard part gives the derivative, rather than making the quotient exactly equal to it. The exceptional index `n = 0` does not affect any free completion.

The public API sends the original difference quotient to the verified core. It receives no supplied expansion or limit.

```python
from fractions import Fraction
from hyperreals import LeanResidueSystem

system = LeanResidueSystem()
epsilon = system.infinitesimal()
x = system.constant(Fraction(2))
dy = (x + epsilon)**3 - x**3
quotient = dy / epsilon
assert quotient.standard_part() == Fraction(12)
assert (quotient - system.constant(Fraction(12))).standard_part() == Fraction(0)
assert system.history == ()
```

The result needs no comparison choice. The script optionally accepts `(-1)^n < 0` afterward, retaining odd indices, and obtains the same standard part. The earlier immutable snapshot still records universal support and an empty trace. This illustrates a result shared by every completion without constructing an ultrafilter.

## An adaptive choice between two quotients

The script also runs an adaptive calculation twice, each time in a fresh system. It explicitly accepts one answer to `(-1)^n < 0` and then selects the expression to evaluate from that accepted answer.

| Accepted answer | Remaining indices | Selected quotient | Standard part |
| --- | --- | --- | --- |
| `true` | Odd | `(2^3 - (2 - epsilon)^3) / epsilon` | `12` |
| `false` | Even | `((2 + epsilon)^3 - 2^3) / epsilon` | `12` |

These are different original expressions. Their exact expansions at positive indices are `12 - 6*epsilon + epsilon^2` and `12 + 6*epsilon + epsilon^2`. The backward quotient is below `12` and the forward quotient is above it. Both errors are infinitesimal. The accepted choice therefore changes the computation while leaving its extracted standard part unchanged.

```python
def adaptive_snapshot(choice):
    system = LeanResidueSystem()
    epsilon, x = system.infinitesimal(), system.constant(Fraction(2))
    assert system.commit(system.alt(), system.constant(Fraction(0)), truth=choice)
    if choice:
        quotient = (x**3 - (x - epsilon)**3) / epsilon
    else:
        quotient = ((x + epsilon)**3 - x**3) / epsilon
    return system.snapshot(quotient)

backward = adaptive_snapshot(True)
forward = adaptive_snapshot(False)
assert backward.expression != forward.expression
assert backward.result == forward.result == Fraction(12)
```

Each snapshot records exactly its one accepted observation. Extracting the standard part does not add another choice or change the support. The opposite answers belong to separate runs, so their incompatible odd and even supports are never combined.

## Run and replay

From the repository root, build the native checker and replay library, then run the example.

```sh
lake build residue_checker Hyperreals.ResidueReplay Hyperreals.InfinitesimalCase Hyperreals.AdaptiveInfinitesimal
uv run python scripts/infinitesimal_case.py
```

The default run exports and kernel-checks five snapshots under `examples/infinitesimal/snapshots`: `quotient`, `error`, `after-odd-choice`, `adaptive-backward`, and `adaptive-forward`. The first three preserve the choice-free example and its later fixed parity choice. The last two capture the separate adaptive runs. Use `--no-parity` to omit `after-odd-choice` and `--no-adaptive` to omit both adaptive runs. Using both flags leaves only the two choice-free snapshots. Use `--skip-replay` to export without kernel checking. The generated `results.json` records each adaptive answer and selected branch, its result, and whether its snapshot was checked. No timings or performance comparisons are claimed.

An individual saved bundle can be checked with the existing verifier.

```sh
uv run python scripts/verify_replay.py examples/infinitesimal/snapshots/quotient
```

Manifests bind each snapshot and its generated Lean source to the current proof sources. After source changes, regenerate the artifacts before verification. The verifier regenerates Lean syntax from the typed snapshot and checks it by kernel reduction. It does not trust the native result or execute an arbitrary supplied Lean file.

## Proof scope

[InfinitesimalCase.lean](../../Hyperreals/InfinitesimalCase.lean) proves positivity and nonzeroness of the increment, its infinitesimal bound in every completion, the exact quotient identity for all positive indices, the quotient's standard part `12`, and the error's standard part zero. It also proves the usual real `HasDerivAt` statement for the cubic at `2`. The expression-denotation theorem links the explicit multiplication-tree AST to the mathematical quotient, and `quotient_extracts` checks the executable extractor on universal support.

The same module connects the sequence calculation to Mathlib's existing hyperreal field. `hyperreal_quotient_standardPart` proves that the literal field fraction `((2 + Hyperreal.epsilon)^3 - 8) / Hyperreal.epsilon` has `ArchimedeanClass.stdPart` equal to `12`. `hyperreal_quotient_gt_derivative` proves that the fraction itself is greater than `12`. This bridge uses Mathlib's existing ultrapower and one chosen free ultrafilter. The preceding sequence theorem applies to every compatible free completion.

[AdaptiveInfinitesimal.lean](../../Hyperreals/AdaptiveInfinitesimal.lean) defines the corresponding formal decision tree as `adaptiveProgram`. Its true branch returns the backward quotient and its false branch returns the forward quotient. `accepted_result` proves that every accepted execution reports `12`, and `accepted_standardPart` proves that result correct in every completion of that execution's trace. These theorems concern the Lean program and its checked observations. The Python implementation mirrors that tree, and tests independently evaluate both captured ASTs with exact rational arithmetic, check their accepted traces and supports, and confirm their common extracted result. This correspondence between Python control flow and the formal program is tested, not a formal refinement theorem.

The general runtime theorems justify successful extraction for every free completion compatible with the recorded observations. Replay instantiates those theorems for the exported syntax. The Python capture process and its relationship to the user's intended expression remain tested implementation boundaries. This example does not prove a general automatic-differentiation procedure.
