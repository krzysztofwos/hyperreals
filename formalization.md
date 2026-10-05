# From finite observations to classical completions

The research question is how a finite computation can use a nonconstructive mathematical object without constructing it. This repository studies that question for sequences interpreted through a free ultrafilter. A program records finite observations about index sets, preserves the possibility of a completion, and extracts a real value exactly when every compatible completion has that same finite standard part within the specified exact fragment. The resulting implementation computes generic polynomial derivatives through literal infinitesimal difference quotients.

The classical object and the executable state have different roles. A free ultrafilter completes every set-membership decision. The runtime stores only a finite description of observations. Existence of a completion is a theorem, not an executable output. The development is pinned to Lean and Mathlib 4.33.0.

Mathlib already supplies a hyperreal field using a noncomputably chosen free ultrafilter. The present development concerns finite computational access to compatible completions and the results that do not depend on resolving the remaining choices. It does not construct a computable free ultrafilter or verify the complete Python package.

## The central semantic statement

[Semantics.lean](Hyperreals/Semantics.lean) treats a sequence as an exact total function `ℕ → ℝ`. A strict comparison denotes the set `{n | x n < y n}`. An equality denotes `{n | x n = y n}`. A negative observation commits the complement of its comparison set. Propositional consistency of names for sets is insufficient because the names have mathematical meanings. [Counterexample.lean](Hyperreals/Counterexample.lean) proves that `(-1)^n < 0` and `(-1)^n = 1` have no common completion, despite being compatible as independent Boolean atoms. Accepted observations must preserve an infinite joint support, which the executable residue checks enforce.

[Completion.lean](Hyperreals/Completion.lean) defines a family of commitments `Γ` and `Completion Γ`, a free ultrafilter containing every set in `Γ`. Freeness means extending the cofinite filter. `HasFreeFIP Γ` requires every finite subfamily of `Γ`, together with cofinite sets, to have nonempty intersection.

The compiled theorems establish:

- `hasFreeFIP_iff_extendible`: this semantic finite-intersection condition is equivalent to existence of a free-ultrafilter completion.
- `jointlyInfinite_iff_extendible`: for a finite family of observations, existence is equivalent to infinitude of their joint intersection.
- `extendible_insert_or_compl`: from an extendible state, at least one polarity of any new semantic query preserves extendibility. This is a classical existence result, not a general algorithm deciding which polarity to take.

[StandardPart.lean](Hyperreals/StandardPart.lean) defines `NearStandardAt U x r` as convergence of `x` to the real value `r` along `U`. `cofiniteLimit_completion_invariant` proves that ordinary cofinite convergence gives the same value in every completion. `nearStandardAt_unique` proves uniqueness for one completion. After observations have restricted the completions, a value may be shared by those remaining completions even when it was not shared before the observations.

The distinction between existential and universal claims matters. An accepted trace must have at least one completion. A successful standard-part answer must be correct in every completion of that trace. Neither claim selects a particular ultrafilter.

The reusable interface separates three obligations: check each finite observation against its semantic meaning, show that one classical completion realizes the entire adaptive run, and justify an ordinary output uniformly over the completions still allowed by that run. The following program theorem composes these obligations for the verified residue language. It does not supply an algorithm for arbitrary observations or claim a new classical extension principle.

### Finite adaptive programs

[ObservationPrograms.lean](Hyperreals/ObservationPrograms.lean) connects this specification to a finite executable decision tree. A `Program` either returns an expression or asks a comparison and continues in one of two subprograms. Later queries and the returned expression can therefore depend on earlier answers. `Program.execute` consumes a finite stream of proposed Boolean answers and checks each selected branch through the residue runtime. Invalid expressions on the visited path, an exhausted stream, or an inconsistent choice cause failure. Unvisited branches are not validated. A successful execution can still have an unknown standard part.

The mathematical `Program.interpret` follows the decisions of one fixed ultrafilter. It is noncomputable. The compiled theorems relate it to actual checked execution:

- `ObservationPrograms.Program.execute_interpret`: every completion of an accepted execution's observations reproduces its entire adaptive branch trace and returned expression under this interpretation.
- `ObservationPrograms.Program.execute_realized`: at least one such free completion exists. The executable run receives neither an ultrafilter nor a proof of this conclusion as input.
- `ObservationPrograms.Program.execute_standardPart_sound`: a rational result of the accepted execution is the standard part of its returned expression in every compatible completion.
- `ObservationPrograms.Program.execute_unknown`: an unknown result records the actual extractor outcome. The completeness result below explains rejection for valid expressions and nonempty supports. It rules out one common finite real value across the remaining completions, without ruling out a finite value in each individual completion.

The same module proves `extendible_iUnion_of_monotone`: if a sequence of commitment families grows monotonically and every prefix is extendible, one free completion contains their entire union. This is a classical finite-intersection argument. It does not compute that completion, certify an unexamined future transition, require one natural index satisfying the entire union, or establish liveness of an infinite program. The finite interpreter is not a formalization of arbitrary Python control flow.

[ObservationProgramExamples.lean](Hyperreals/ObservationProgramExamples.lean) supplies additional boundary checks, including different later queries, contradictory observations, a false cofinite choice, exhausted input, invalid visited syntax, and a successful unknown result. Its branches can return different ordinary values. The shared-result infinitesimal program below illustrates a stronger property that must be proved for the particular program.

## Generic executable polynomial differentiation

[PolynomialDifferentiationCore.lean](Hyperreals/PolynomialDifferentiationCore.lean) compiles dense ascending rational coefficient lists into the existing expression grammar. `polynomialExpr` constructs Horner syntax. `quotientExpr coefficients a c k` constructs the literal unexpanded quotient with step `c/n^(k+1)`. The successor exponent makes positivity structural. The public Python `polynomial_quotient(coefficients, a, c, order, system=...)` takes the positive exponent itself and corresponds to Lean's `k = order - 1`.

[PolynomialDifferentiation.lean](Hyperreals/PolynomialDifferentiation.lean) proves the full connection from those inputs to the executed extractor:

- `rational_polynomial_representable`: every rational polynomial has a dense coefficient-list representation. Degree is arbitrary.
- `polynomialExpr_eval` and `quotientExpr_denote`: the generated source denotes polynomial evaluation and the literal difference quotient.
- `polynomial_hasDerivAt`: the recursively computed rational `derivativeValue` is the ordinary real derivative at the rational point.
- `quotientExpr_cofiniteLimit`: for every nonzero rational scale and positive integer exponent, the quotient converges to that derivative.
- `quotientExpr_standardPart`: for every nonempty residue support, the existing extractor actually returns that rational.
- `quotientExpr_computes_derivative`: combines the executable equality with equality to the ordinary `deriv` result.

The extractor receives the raw expression and support. Neither its input nor the compiler contains a supplied derivative value, expansion, or convergence certificate. `derivativeValue` computes the expected result from the coefficients in the theorem statement. The proof obtains extraction from independently established convergence and the completeness theorem below.

The same core constructs `dividedDifference coefficients a increment`. This is an exact polynomial transformation `D` with `h * D = P(a+h) - P(a)` at every index, even where `h = 0`. `dividedDifference_eq_quotient` identifies it with the genuine quotient under a nonzero premise. `dividedDifference_standardPart` proves extraction of the derivative on every nonempty support whenever a valid represented increment has cofinite limit zero. The stronger `dividedDifference_standardPart_of_standardPart` needs only `standardPart support increment = some 0`. Observations may therefore establish infinitesimality on the retained completions even when the increment does not converge globally. Zero increments yield the polynomial extension. This transformation does not supply an inverse or justify division by zero.

The [Python constructors](src/hyperreals/polynomial.py), `evaluate_polynomial`, `divided_difference`, and `polynomial_quotient`, mirror the Lean syntax. Their correspondence is tested rather than formally refined. Kernel replay verifies a particular exported expression and trace. The generic Lean theorem establishes the all-input result for the formal compiler.

## Verified differentiable expressions

[DifferentiableCore.lean](Hyperreals/DifferentiableCore.lean) defines an executable `Expr n` with rational constants, `Fin n` variables, arithmetic including general division, negation, and `sin`, `cos`, `exp`, `log`, and `sqrt`. `Program n m` is a finite vector of expressions. `Expr.jvp` compiles a source expression and input tangent expressions into derivative syntax. `Program.jacobian` applies it to coordinate basis vectors. The core imports no real analysis and does not accept derivative certificates.

[Differentiable.lean](Hyperreals/Differentiable.lean) gives exact real evaluation and a compositional `Domain` predicate. Denominators must be nonzero. Logarithm and square root arguments must be positive. These sufficient smoothness conditions are deliberately conservative. The compiled theorems establish:

- `Expr.differentiableAt`: every expression is Fréchet differentiable on its specified domain.
- `Expr.hasDerivAt_eval`: the actual compiler obeys the chain rule along every differentiable input curve.
- `Expr.jvp_eq_fderiv`: the compiled output equals the Fréchet derivative applied to the tangent value at the base point.
- `Expr.domain_jvp`: the source and tangent domains imply the compiled output domain.
- `Expr.domain_eventually`: the source domain is open. Small enough perturbations remain valid.
- `Program.hasDerivAt_line` and `Program.jacobian_eq_fderiv`: vector outputs and coordinate Jacobian entries have the corresponding derivative guarantees.

[DifferentiableInfinitesimal.lean](Hyperreals/DifferentiableInfinitesimal.lean) defines the literal sequence quotient `(F(x + h(k)v) - F(x)) / h(k)`. `Expr.quotient_tendsto` derives its limit from the compiler theorem for any filter in which `h` tends to zero and is eventually nonzero. `Expr.quotient_standardPart` and `Program.quotient_standardPart` specialize this to each compatible completion. `Expr.quotient_eventually_domain` ensures the perturbed source operations are eventually in their domains. Neither a derivative nor a remainder estimate is a supplied premise. The proof uses ordinary differentiability to establish the quotient limit. The step is not replaced by a nilpotent.

[DifferentiableExamples.lean](Hyperreals/DifferentiableExamples.lean) connects this language to the existing checked step observation. `quotient_after_step_observation` handles every expression in the new grammar on either accepted branch. `vectorExample_after_observation` specializes it to the three-output elementary function in the [example](examples/differentiable/README.md), with symbolic standard part `(cos(1) + 2 sin(1), 1, 1 / sqrt(2))`. This reuses a residue observation trace to justify the step. It does not extend the finite observation interpreter with elementary comparisons or feed elementary quotients to Laurent normalization.

The [Python compiler](src/hyperreals/differentiable.py) returns immutable, dimension-checked expressions. `CompiledJVP.verify()` regenerates fixed Lean source and proves each output equals `Expr.jvp` by `decide +kernel`, then instantiates the Fréchet derivative, domain-preservation, and standard-part theorems. A supplied rational point additionally requests kernel proofs of the source, tangent, and result domains there. The bounded domain tactic is incomplete, and failure is an unresolved proof obligation. The API accepts no caller-provided Lean tactics or proof files. Verification audits every generated theorem against the project's allowed axiom set. Python capture remains an unproved boundary. `approximate()` is floating-point evaluation with no certified error bound.

The symbolic layer has no complete standard-part or equality decision procedure. Its output is an exact expression rather than a rational answer. The existing Laurent completeness and rejection theorems retain their original scope. Reverse mode, general control flow, tensors, certified numerical rounding, and efficient shared-expression compilation remain separate extensions. Formal dimensions are arbitrary finite naturals. The Python interface limits dimensions to 64, constant numerators and denominators to 4096 bits, and generated proofs to 20,000 expression nodes and depth 128. Verification uses the existing bounded timeout policy.

## An observation that establishes a division condition

[DomainInfinitesimal.lean](Hyperreals/DomainInfinitesimal.lean) uses `h(n) = periodic([0,1])(n) / n`. It is infinitesimal in every free completion, but its nonzeroness depends on the observations. The finite program queries `h = 0`. The negative branch keeps `h` and multiplies the cubic numerator by the represented reciprocal `periodic([0,1]) * n`. The positive branch replaces the zero step with `1/n²` before forming the quotient. Both branches return standard part 12 at the point 2.

The proofs distinguish infinitesimality, invertibility, and extraction:

- `step_infinitesimal`: the initial step has standard part zero in every completion.
- `observation_holds`: the equality or its negation holds eventually in the same completion that realizes the accepted trace.
- `negative_branch_inverse`: the represented reciprocal is an inverse on the nonzero branch.
- `zero_branch_has_no_inverse`: no sequence can serve as an inverse to the original step on the zero branch.
- `selected_step_nonzero` and `selected_step_infinitesimal`: the selected denominator meets both conditions, using the fallback when required.
- `accepted_quotient_correspondence`: the actual returned expression denotes the literal quotient with the selected denominator in every completion of that accepted branch.
- `accepted_result`, `accepted_realized`, and `accepted_standardPart`: both accepted executions extract 12, have a fixed realizing completion, and give the result in all their compatible completions.
- `guarded_no_common_standardPart`: before refinement, the represented original quotient has no shared finite real standard part. Its even and odd limits are 0 and 12.

The [companion script](scripts/domain_infinitesimal_case.py) exports and verifies both accepted branches and the unrefined rejection. This example makes an observation discharge a necessary mathematical condition. It uses a separately proved represented reciprocal and the existing monomial division grammar. It does not introduce a general inversion algorithm.

## One program, two infinitesimal quotients

For `ε(n) = 1/n`, the finite program observes whether `(-1)^n < 0` and selects a literal difference quotient for `f(x) = x³` at `x = 2`:

```text
if observe((-1)^n < 0):
    return (2³ - (2 - ε)³) / ε       # backward
else:
    return ((2 + ε)³ - 2³) / ε       # forward
```

Accepting the positive answer retains odd indices and selects the backward quotient. Accepting the negative answer retains even indices and selects the forward quotient. These are separate executions with incompatible traces, each of which admits a free completion. The observation determines the returned syntax. The denominators and the exact finite-index identities are part of the same calculation:

```text
q_backward = 12 - 6ε + ε²            (n > 0)
q_forward  = 12 + 6ε + ε²            (n > 0)
st(q_backward) = st(q_forward) = 12
```

The increment is positive and nonzero in every free completion and smaller than every positive real constant. The forward quotient is above 12 at every positive index, and the backward quotient is below 12. Their errors vanish. Taking a standard part removes an infinitesimal error without making either quotient pointwise equal to the derivative. No small floating-point step or truncated Taylor series is used. Each quotient already converges ordinarily to 12, so the role of the observation is to select a calculation, not to make its derivative exist.

The [README example](README.md#an-adaptive-infinitesimal-calculation) executes both alternatives through `LeanResidueSystem`. Each run has a fixed-completion interpretation, and its result is valid in every completion of that run. Agreement across the two different runs is an additional property of this particular program. The general adaptive-execution theorem does not say that all branches of every program return the same ordinary value.

[AdaptiveInfinitesimal.lean](Hyperreals/AdaptiveInfinitesimal.lean) defines the actual finite tree `Hyperreals.AdaptiveInfinitesimal.adaptiveProgram`. Its compiled results include:

- `backwardQuotientExpr_denote` and `backward_identity`: the raw backward AST denotes the literal quotient and has the stated exact expansion at every positive index.
- `backward_standardPart`: the backward quotient has standard part 12 in every free completion, independently of the sign observation.
- `execute_cons`: either proposed Boolean answer executes successfully, consumes one answer, and leaves any remaining answers unused. Its recorded trace, support, expression, and result are proved equal to the actual interpreter output.
- `accepted_result`: every accepted execution of this program, over every proposed answer stream, has extracted result `some 12`.
- `accepted_realized`: every accepted execution has one fixed free completion reproducing its trace and selected expression.
- `accepted_standardPart`: the selected expression has standard part 12 in every completion of that execution's observations.
- `accepted_branch_side`: in every compatible completion, the selected quotient remains below 12 on the backward branch and above 12 on the forward branch.

[InfinitesimalCase.lean](Hyperreals/InfinitesimalCase.lean) supplies the arithmetic foundation. In its `Hyperreals.InfinitesimalCase` namespace, `epsilon_positive_infinitesimal` and `epsilon_nonzero` establish the denominator properties. `quotient_identity` proves the forward identity, `quotient_standardPart` gives 12 in every completion, and `quotient_ne_derivative` with `error_infinitesimal` distinguishes equality from equality of standard parts. `cubic_hasDerivAt` independently identifies the usual real derivative. `quotientExpr_denote` connects the raw forward syntax to its mathematical sequence, and `quotient_extracts` checks exact extraction with `decide +kernel`.

That module also connects the forward sequence to Mathlib's actual `Hyperreal` field, which uses its fixed classical free ultrafilter. `hyperreal_quotient_ofSeq` identifies the field expression with the class of the quotient sequence. `hyperreal_quotient_gt_derivative` proves that this element is strictly greater than 12. `hyperreal_quotient_standardPart` proves the literal equality `ArchimedeanClass.stdPart (((2 + Hyperreal.epsilon)^3 - 8) / Hyperreal.epsilon) = 12`. This bridge uses the proved sequence limit, not an assumed differentiation rule. The sequence result remains valid for every free completion.

The [worked example](examples/infinitesimal/README.md) and [companion script](scripts/infinitesimal_case.py) expose the branch-dependent calculation through the public API. The default command `uv run python scripts/infinitesimal_case.py` exports and verifies both adaptive branches, the choice-free forward quotient, its error, and the forward quotient after a later odd-parity observation. `--no-adaptive` omits the two adaptive runs. `--no-parity` omits the later-choice snapshot. `--skip-replay` explicitly leaves exported claims unverified. No performance claim depends on this case.

Python branch selection and its correspondence with the finite Lean tree are tested rather than formally translated. Snapshot replay proves the exported trace and selected expression's result. It does not prove that an arbitrary Python program followed the intended tree. The finite program theorem and concrete arithmetic proofs establish the mathematical connection, while capture remains an implementation boundary.

## An executable fragment of the semantics

The main API, [`LeanResidueSystem`](src/hyperreals/verified_residue.py), sends raw expression trees to [ResidueChecker.lean](ResidueChecker.lean). Its grammar has rational constants, `n`, `1/n`, arbitrary nonempty rational periodic tables, addition, subtraction, multiplication, and division by an explicit nonzero rational monomial. Empty tables and zero monomial coefficients are rejected. General rational-function denominators and analytic functions are outside this grammar.

`periodic(values)` denotes `values[n % len(values)]`. Constants, `n`, and `1/n` have coefficient period one, although the last two sequences are not themselves periodic. Binary expressions combine coefficient periods by LCM. Each residue normalizes to an exact rational polynomial divided by `n^shift`. All coefficients are retained, including arbitrarily high-order terms. Use `divide_monomial(c, k)` for division by `c * n^k`, with negative integer powers allowed. Python `/` accepts primitive constants, `n`, and `1/n`.

| Layer          | Executable definitions                                                   | Proved connection                                                                                                                                                                     |
| -------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Support        | [ResidueSupportCore](Hyperreals/ResidueSupportCore.lean)                 | [ResidueSupport](Hyperreals/ResidueSupport.lean): lifting, LCM intersection, complement, and infinitude                                                                               |
| Expressions    | [ResidueExprCore](Hyperreals/ResidueExprCore.lean)                       | [ResidueExpr](Hyperreals/ResidueExpr.lean): positive coefficient periods and exact source denotation                                                                                  |
| Comparisons    | [ResidueComparisonCore](Hyperreals/ResidueComparisonCore.lean)           | [ResidueComparison](Hyperreals/ResidueComparison.lean): agreement from a computed cutoff onward                                                                                       |
| Standard parts | [ResidueLimitCore](Hyperreals/ResidueLimitCore.lean)                     | [ResidueLimit](Hyperreals/ResidueLimit.lean) and [ResidueLimitCompleteness](Hyperreals/ResidueLimitCompleteness.lean): exact characterization of completion-independent finite values |
| Diagnostics    | [ResidueLimitDiagnosticCore](Hyperreals/ResidueLimitDiagnosticCore.lean) | [ResidueLimitDiagnostic](Hyperreals/ResidueLimitDiagnostic.lean): finite answers, divergent active residues, and disagreement witnesses                                               |
| Observations   | [ResidueRuntimeCore](Hyperreals/ResidueRuntimeCore.lean)                 | [ResidueRuntime](Hyperreals/ResidueRuntime.lean), [ResidueTrace](Hyperreals/ResidueTrace.lean): accepted traces and all compatible completions                                        |

The state starts as `[true]`. A commitment intersects its comparison mask with the existing support over their LCM and succeeds only if the result is nonempty. Introducing a new period therefore preserves earlier observations. Shared factors retain correlations between residue choices. Since each nonempty periodic support is infinite, successful states satisfy the semantic extension condition.

The principal executable-refinement theorems are:

- `Residue.Support.carrier_lift` and `Residue.Support.carrier_inter`: period lifting preserves denotation, and LCM intersection represents the exact intersection of positive-period supports.
- `Residue.Support.carrier_infinite`: every nonempty finite-period support represents an infinite set of indices.
- `Residue.Expr.normalizeAt_residue_correct`: normalization modulo any multiple of the expression period preserves its direct real denotation at positive indices.
- `Residue.compile_correct`: the computed comparison mask agrees with the source `<` or `=` set at every index from the reported cutoff onward.
- `Residue.run_trace_extendible`: every accepted finite trace has a free completion containing all its actual observation sets.
- `Residue.run_universe_mem_iff`: a free ultrafilter contains the final support exactly when it contains every actual observation of the trace.
- `Residue.standardPart_tendsto` and `Residue.run_standardPart_sound`: successful extraction gives the returned rational in every compatible completion. These hypotheses refer to actual execution results, not supplied convergence certificates.
- `Residue.standardPart_iff_all_completions`: for valid expressions and nonempty support, extraction returns a rational exactly when all compatible completions converge to it.
- `Residue.standardPart_complete_real`: any finite real value shared by all compatible completions is necessarily rational and is returned by extraction.
- `Residue.standardPart_none_no_common_real`: rejection under those same validity and nonemptiness conditions rules out every shared finite real candidate, including irrational candidates.

Extraction enumerates the LCM of the expression and state periods and tests every active residue. It returns a rational only if all active scalar forms have that same finite limit. It never chooses a residue. Neither extraction nor `probe` changes the state. The result is invariant over completions consistent with the trace, not over choices that the trace has already excluded. Soundness and completeness together characterize success and rejection for this grammar. An individual completion can have a finite limit even when no one value works uniformly over the retained family.

`expression.explain_standard_part()` returns `StandardPartDiagnostic(kind, period, residues, limits)` without changing the state. Its common period identifies the residue classes used by normalization. [ResidueLimitDiagnostic.lean](Hyperreals/ResidueLimitDiagnostic.lean) proves `diagnoseStandardPart_finite_iff`, which equates a finite diagnostic with successful extraction. `diagnoseStandardPart_divergent_sound` proves that the displayed active branch escapes every bounded interval in magnitude. `diagnoseStandardPart_disagreement_sound` proves that the two displayed active branches converge to the unequal rational limits reported. Invalid syntax and empty supports have a separate internal diagnostic and are rejected by the public interface.

Build and run the core with `lake build residue_checker` and `uv run python scripts/residue_demo.py`. The demonstration accepts `(-1)^n < 0` and `periodic([0,1,2]) = 2`, retains precisely residue 5 modulo 6, and rejects an incompatible later observation. Integration tests also evaluate raw syntax with `Fraction` at reported cutoffs. These sampled checks test transport and native behavior. The tail-wide mathematical guarantee comes from the Lean proofs.

### Scalar arithmetic and restricted interfaces

The residue implementation reuses the scalar Laurent algorithms. [LaurentExpr](Hyperreals/LaurentExpr.lean) proves exact normalization, [LaurentSign](Hyperreals/LaurentSign.lean) proves computed sign bounds, and [LaurentLimit](Hyperreals/LaurentLimit.lean) proves convergence from the executable coefficient test. The sign algorithm traverses Horner coefficients. For nonzero tail leading coefficient `a` and head `b`, it raises the cutoff to at least `ceil(abs(b) / abs(a)) + 1`. Induction bounds the entire remaining tail, rather than checking selected sample indices.

The finite-limit test rejects nonzero numerator coefficients above the denominator shift and reads the coefficient at that shift. Lower powers vanish. `Laurent.Form.standardPart?_sound` connects success to ordinary convergence. [LaurentLimitCompleteness](Hyperreals/LaurentLimitCompleteness.lean) proves the converse along every nontrivial filter extending the cofinite filter and proves that rejected scalar forms diverge in magnitude. The residue completeness proof constructs a free ultrafilter concentrating on each active residue to recover every branch obligation. Exact finite sums need neither a truncation rule nor an approximate remainder estimate.

[`LeanLaurentSystem`](src/hyperreals/verified_laurent.py) exposes this arithmetic with only even and odd coefficients. Its compiled results include `Laurent.Expr.normalize_sequence_correct`, `Laurent.Form.sign_correct`, `Laurent.compile_correct`, `Laurent.run_universe_mem_iff`, and `Laurent.run_standardPart_sound`. [`LeanPeriodicSystem`](src/hyperreals/verified.py) is smaller still, with rational constants and alternating signs but no `n`, `1/n`, division, or standard parts. [Periodic.lean](Hyperreals/Periodic.lean) proves its evaluator, comparisons, transitions, and trace completion theorem. These restricted interfaces share the exact arithmetic approach. The arbitrary-period backend is the main executable realization.

The executable cores explicitly enumerate residue masks and retain dense scalar coefficient lists. Potential LCM growth limits this representation. A compact implementation would be an engineering extension requiring its own refinement proof, not the central mathematical research question.

## Kernel replay of a finite computation

[ResidueReplay.lean](Hyperreals/ResidueReplay.lean) defines a data-only snapshot containing ordered observations, claimed final support, query expression, and optional rational result. Its Boolean check validates the query and nonempty support, recomputes the trace from universal support, and checks the exact extraction result. The snapshot contains no supplied proof of its intended semantic conclusions.

- `Residue.Replay.Snapshot.check_iff` characterizes successful replay in terms of those executable checks.
- `Residue.Replay.Snapshot.trace_extendible` and `Residue.Replay.Snapshot.support_mem_iff` establish completion existence and the exact family of compatible completions.
- `Residue.Replay.Snapshot.standardPart_sound` establishes a replayed rational answer in every compatible completion.
- `Residue.Replay.Snapshot.unknown_result` establishes that extraction returned `None`.
- `Residue.Replay.Snapshot.no_common_standardPart` then proves that no finite real value is shared by every completion of the actual trace. Individual completions and later compatible refinements may still have finite standard parts.
- `Residue.Replay.Snapshot.failure_classified` proves that a replayed rejection has a computed divergence or disagreement diagnosis. The separately reported native witness fields are not part of the snapshot.

The [Python adapter](src/hyperreals/verified_residue.py) records accepted observations. Failed commitments, probes, and extraction add no choices. `snapshot(expression)` captures history, support, expression, and actual native extraction under the same state lock. The [replay module](src/hyperreals/replay.py) validates and freezes the data independently of later state changes. A captured snapshot is a claim until it is verified.

Export writes versioned `snapshot.json`, deterministic `Replay.lean`, and a manifest binding their hashes and relevant project sources. Verification rejects mismatched files, refreshes the replay-module build, and regenerates fixed Lean syntax from validated data. It does not execute arbitrary supplied Lean text. Each generated artifact proves `snapshot.check = true` with `decide +kernel`, instantiates the semantic theorems, and audits every generated headline declaration. Missing reports or unexpected axioms fail verification. Native decision shortcuts are not used. Replays of `None` now include the semantic rejection theorem. The snapshot format records the query result rather than the separate diagnostic witness fields.

A false native answer cannot pass this check for the recorded expression and trace. However, replay does not prove that Python captured the intended external session faithfully. Hashes identify artifact components and source revisions. They do not authenticate a historical execution. The Lean kernel, imported proof environment, and underlying platform remain trusted. Ordinary native calls additionally rely on parsing, transport, native compilation, and execution until separately replayed.

Replay accepts at most 128 observations, coefficient periods at most 4096, AST depth 64, 10,000 expression nodes/table entries, dense numerator degrees and denominator shifts below 4096, and 1024 decimal digits per integer. JSON input is limited to 8 MiB. The configurable timeout is positive and finite, defaults to 60 seconds, and is at most 300 seconds across dependency freshness and kernel checking. These are implementation resource guards, not mathematical completeness bounds. Ordinary native commitments retain their wider resource scope.

Tests cover immutable snapshots, negative observations, rejected transitions, large rationals, retained powers, unknown results, malformed data, wrong answers/supports, file association, and audit failures. [scripts/verify_replay.py](scripts/verify_replay.py) checks saved artifacts.

## Assumptions and unproved implementation boundaries

- Semantic sequences are exact total functions `ℕ → ℝ`. Freeness means extending the cofinite filter. Lean's total inverse gives `1/0 = 0`, and that single index does not affect free-filter limits.
- Completion existence is classical and noncomputable. The development uses Mathlib's ultrafilter existence theorem. Axiom audits allow `propext`, `Classical.choice`, and `Quot.sound` and reject unfinished proofs and project-local axioms.
- Lean-backed APIs serialize rational constants exactly. A supplied float denotes its binary rational value. Python conversion, capture, JSON parsing, and state handling are tested rather than formally refined to the Lean definitions.
- The normalization and extraction theorems cover the specified finite Laurent grammar. They do not cover general division, analytic expansions, or floating-point coefficients with approximate error bounds.
- A mathematical model's association with an external application is an assumption. Neither completion existence nor kernel replay validates an empirical interpretation of the recorded index sets.

### Supporting semantic lemmas

[EventuallyPeriodic.lean](Hyperreals/EventuallyPeriodic.lean) proves an infinitude checker for represented eventual-periodic sets. [Certificates.lean](Hyperreals/Certificates.lean) proves that an eventual comparison certificate selects a cofinite set and preserves existing completions. The generic `safePositiveChoice_extendible` theorem in [Completion.lean](Hyperreals/Completion.lean) requires the semantic finite-intersection condition explicitly. These are supporting mathematical interfaces. They do not extend the executable grammar.

[RuntimeInvariants.lean](Hyperreals/RuntimeInvariants.lean) records a quotient lower-bound rule and a truncation counterexample. The latter shows why a vanishing term cannot be discarded before a later Laurent shift makes it visible. The exact core avoids that loss by retaining every coefficient.

## Supporting examples and measurements

The [paired-channel calibration model](examples/delayed-choice/README.md) uses exact signals `F±(x,n) = x³ ± b(n)x + offset±(n)` with coefficient periods 4, 6, and 5. Opposite gain biases and a shared cubic response are supplied modeling assumptions. Incremental observations reduce sixty possible phases to thirty, ten, two, and finally residue 23. The averaged slope has standard part 12 before phase resolution. A raw channel's standard part becomes 19 while ten phases remain.

Three saved snapshots certify the early fused result, partially resolved raw result, and final phase after the live session has advanced. An exhaustive exact reference agrees on accepted observations and candidate sets. A specified eager policy selects the smallest compatible full-period residue and then needs backtracking to accommodate later evidence. Its rejection is sound for its extra choice. This is a constructive example of one early commitment's cost, not a finding that all eager algorithms fail or that physical sensors behave according to the model.

Exact evaluation at 180 selected positive indices corroborates the finite-difference identities. The generated Lean theorems establish the standard-part claims for the exported expressions. [Comparative benchmarks](benchmarks/README.md) record exact result agreement and timings for twelve hand-selected finite workloads against sparse rational enumeration and SymPy. Public API, persistent native execution, direct Python formulas, and full replay have different measured scopes. These artifacts document behavior and costs without establishing scalability, application coverage, or a speed advantage.

## Research direction and remaining obligations

The central question is which observations and outputs of a computation can be justified without resolving the underlying nonconstructive object. The proposed contribution is the proved connection from executable semantic checks to an adaptive run under one fixed classical completion, together with an ordinary output shared by all completions of that run. The domain-dependent infinitesimal program makes that connection concrete: its observation establishes which denominator is invertible, both accepted executions return 12, and the result is the ordinary derivative. Generic polynomial differentiation establishes an all-degree family beyond the concrete cubic. Complete extraction characterizes exactly which finite ordinary values survive all remaining choices. The generic program theorem is reusable across the finite syntax. Agreement across different runs remains a separate, program-specific obligation.

The following are separate obligations rather than claims of an already complete system:

1. **Beyond the finite program model.** The finite adaptive interpreter and conditional monotone-union completion theorem are proved. Relating them to a larger host language, infinite operational behavior, or liveness would require new semantics and refinement arguments. The whole-run existence result does not supply executable witness extraction.
2. **Rational-function division.** Periodic rational functions are a natural next grammar extension. They require a verified representation, eventual denominator-nonzero checks on every retained residue, and complete sign and limit algorithms. The polynomial divided-difference compiler and the example's represented reciprocal do not provide arbitrary inversion.
3. **A broader analytic bridge.** Further analytic computation needs explicit domain and convergence guarantees. Certified approximate evaluation additionally needs coefficient and rounding error bounds. The new differentiable language proves symbolic elementary JVPs and their quotient limits on explicit smooth domains. Complete elementary limit decisions, certified numerical evaluation, and a richer observation language remain open.
4. **Faithful execution capture.** Refine serialization, parsing, and transcript capture to the formal semantics, or keep their assumptions explicit. Kernel replay already checks the exported mathematical instance, not its historical provenance.
5. **Independent application value.** Find a problem whose naturally occurring observations and required outputs benefit from this interface. Compare the same tasks with ordinary exact algebra and finite-constraint methods. The constructed calibration model and small benchmarks do not settle that question. Compact state and runtime improvements are supporting engineering work, not substitutes for it.

### Prior work and the publication question

The contribution must be stated relative to existing nonstandard analysis and formalization, not as the first formalized hyperreals or first correct infinitesimal computation:

- Fleuriot and Paulson, [Mechanizing Nonstandard Real Analysis](https://doi.org/10.1112/S1461157000000267) (2000), formalized hyperreal construction and analysis in Isabelle/HOL.
- [Mathlib's hyperreal development](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Analysis/Real/Hyperreal.html) provides a hyperreal field and standard-part infrastructure.
- Beeson, [Using Nonstandard Analysis to Ensure the Correctness of Symbolic Computations](https://www.michaelbeeson.com/research/papers/nsappt.pdf) (journal version 1995), used nonstandard reasoning and checked side conditions in Mathpert, including infinitesimal elimination.
- Beeson and Wiedijk, [The meaning of infinity in calculus and computer algebra systems](https://doi.org/10.1016/j.jsc.2004.12.002) (2005, [author PDF](https://www.cs.ru.nl/~freek/pubs/limits.pdf)), give filter semantics for symbolic calculations with infinity, including partially informative outcomes and refinement. This is direct prior work for the semantic bridge studied here.
- Harrison and Théry, [A Skeptic's Approach to Combining HOL and Maple](https://doi.org/10.1023/A:1006023127567) (1998), separate external symbolic computation from formal checking. Kernel replay follows this established architecture rather than introducing the general idea.
- Kido, Chaudhuri, and Hasuo, [Abstract Interpretation with Infinitesimals](https://arxiv.org/abs/1511.00825) (2015 preprint), established soundness and termination for nonstandard static analysis and evaluated hybrid-system examples.
- Dou and Yu, [Formalization of the Filter Extension Principle in Coq](https://arxiv.org/abs/2407.06222) (2024 preprint), mechanized the classical extension principle.

This remains a targeted prior-work comparison, not an exhaustive novelty search. The extension and compactness arguments, use of filters for symbolic semantics, and checking of externally computed results are established ideas. The specific contribution proposed here is their verified operational connection: exact executable observations determine a family of compatible completions, every accepted adaptive run is realized under one fixed member of that family, and complete extraction characterizes the finite outputs shared by the family. Generic polynomial differentiation supplies an executable all-degree application. The elementary compiler extends the quotient interpretation to finite differentiable vectors while retaining explicit domain obligations. The domain-dependent infinitesimal program additionally makes a recorded observation establish a necessary division condition. Whether this connection extends usefully beyond the present finite syntax remains open. A fully verified general hyperreal runtime is not established.

## Validation

```bash
lake build
make lean-audit
uv run pytest
uv run pytest tests/test_verified_periodic.py tests/test_verified_laurent.py
uv run pytest tests/test_verified_residue.py tests/test_replay.py tests/test_replay_capture.py
uv run pytest tests/test_infinitesimal_case.py tests/test_domain_infinitesimal_case.py
uv run pytest tests/test_differentiable.py
uv run python scripts/differentiable_case.py
uv run python scripts/infinitesimal_case.py
uv run python scripts/domain_infinitesimal_case.py
uv run python scripts/verified_demo.py
uv run python scripts/residue_demo.py
uv run python scripts/verify_replay.py examples/delayed-choice/snapshots/early-fused
```

The audit rebuilds the import graph and native checker, rejects unfinished proofs and local axioms including private declarations, re-elaborates every audited module, and rejects dependencies outside `propext`, `Classical.choice`, and `Quot.sound`. Lean CI runs the Python/Lean integration tests and verifies the maintained saved replay artifacts. The domain example adds checked instances for both accepted branches and for the unrefined rejection. Ordinary Python-only installations skip tests requiring built Lean artifacts. Reproducing the calibration timings with `uv run python scripts/delayed_choice_case_study.py --repeat 5` is optional evaluation work, not a prerequisite for the completion theorem.
