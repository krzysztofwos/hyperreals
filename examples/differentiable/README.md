# Elementary functions and literal infinitesimal quotients

This example differentiates a function from two real inputs to three real outputs:

\[
F(x,y)=\left(\sin(x)e^y,\ \log(1+x^2+y^2),\ \frac{\sqrt{1+x^2}}{1+y^2}\right).
\]

At `(1, 0)` in direction `(1, 2)`, its derivative is `(cos(1) + 2 sin(1), 1, 1 / sqrt(2))`. Those expressions remain symbolic. The numbers printed by the script are approximate samples, not certificates.

```bash
uv sync --locked --extra dev
lake build Hyperreals.DifferentiableExamples
uv run python scripts/differentiable_case.py
```

The Python compiler takes the source program and direction. `CompiledJVP.verify(point=(1, 0))` regenerates fixed Lean syntax, checks that the returned expression equals the formal compiler output by kernel reduction, and instantiates the derivative and literal quotient theorems. It also proves the source, direction, and output domains at that point. It does not take a supplied derivative or limit certificate.

[`DifferentiableExamples.lean`](../../Hyperreals/DifferentiableExamples.lean) additionally proves the displayed simplified derivative and links it to the observation-dependent step. The step is zero on even indices and `1/n` on odd indices. A checked equality observation either selects the nonzero branch or causes replacement by `1/n²`. For every completion of either accepted branch, the selected step is a nonzero infinitesimal and the literal vector quotient has the displayed standard part. The observation engine decides residue comparisons. The elementary quotient receives its standard-part guarantee from the differentiable compiler theorem.

The language has rational constants, finite vectors, arithmetic with general division, and `sin`, `cos`, `exp`, `log`, and `sqrt`. Division requires a nonzero denominator. Logarithm and square root require positive arguments. This is a conservative smooth domain. A failed domain proof means that this verifier has not established applicability. It is not a proof that the derivative does not exist.

`program.jacobian()` returns rows indexed by output and columns indexed by input. `program.gradient()` requires one output. These use forward basis JVPs. They do not implement reverse mode. Tangent expressions may depend on the base point, but their values are held fixed as the infinitesimal step varies. Host-language loops can construct finite syntax, but there is no verified translation of arbitrary Python control flow.

Complete Laurent extraction and its divergence/disagreement diagnostics apply to residue expressions. The differentiable compiler does not decide equality, arbitrary limits, or all domain conditions for elementary expressions. It introduces no rule setting an infinitesimal square to zero.
