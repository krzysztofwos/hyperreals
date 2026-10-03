import Hyperreals.ObservationPrograms
import Hyperreals.InfinitesimalCase
import Mathlib.Tactic.IntervalCases

/-!
# An observation that discharges a division obligation

The step h is zero on even indices and 1/n on odd indices. It is infinitesimal
in every free completion, but only the negative answer to h = 0 licenses its
use as a denominator. On that branch, a represented reciprocal gives the
literal cubic quotient. The positive branch replaces h by 1/n² before dividing.
Both accepted branches extract 12. This uses the existing Laurent grammar,
not a general-purpose division operator.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.DomainInfinitesimal

open Residue ObservationPrograms InfinitesimalCase

def indicatorExpr : Expr := .periodic [0, 1]
def stepExpr : Expr := .mul indicatorExpr .reciprocalIndex
def reciprocalExpr : Expr := .mul indicatorExpr .index

def guardedQuotientExpr : Expr :=
  .mul (.sub (cubeExpr (.add (.constant 2) stepExpr)) (cubeExpr (.constant 2)))
    reciprocalExpr

def fallbackStepExpr : Expr := .mul .reciprocalIndex .reciprocalIndex

def fallbackQuotientExpr : Expr :=
  .divMonomial
    (.sub (cubeExpr (.add (.constant 2) fallbackStepExpr)) (cubeExpr (.constant 2)))
    1 (.negSucc 1)

private theorem indicator_zero_or_one (n : ℕ) :
    indicatorExpr.denote n = 0 ∨ indicatorExpr.denote n = 1 := by
  have hn := Nat.mod_lt n (by decide : 0 < 2)
  interval_cases h : n % 2 <;>
    norm_num [indicatorExpr, Expr.denote, Expr.eval, h]

theorem step_denote (n : ℕ) :
    stepExpr.denote n = indicatorExpr.denote n * epsilon n := rfl

theorem reciprocal_denote (n : ℕ) :
    reciprocalExpr.denote n = indicatorExpr.denote n * n := rfl

/-- Infinitesimal does not imply invertible. The unrefined step has standard
part zero even though one of its two possible values is exactly zero. -/
theorem step_extracts_zero : standardPart Support.universe stepExpr = some 0 := by
  decide +kernel

theorem step_infinitesimal {Γ : Commitments} (completion : Completion Γ) :
    NearStandardAt completion.ultrafilter stepExpr.denote 0 := by
  simpa using nearStandardAt_of_cofiniteLimit completion
    (standardPart_universe_cofinite step_extracts_zero)

theorem represented_inverse_of_nonzero (n : ℕ) (hstep : stepExpr.denote n ≠ 0) :
    stepExpr.denote n * reciprocalExpr.denote n = 1 := by
  rw [step_denote] at hstep
  rcases indicator_zero_or_one n with hi | hi
  · simp [hi] at hstep
  · have hn : (n : ℝ) ≠ 0 := by
      intro hn
      simp [epsilon, reciprocalIndex, hn] at hstep
    rw [step_denote, reciprocal_denote, hi]
    simp [epsilon, reciprocalIndex, hn]

theorem guarded_quotient_denote (n : ℕ) (hstep : stepExpr.denote n ≠ 0) :
    guardedQuotientExpr.denote n =
      ((2 + stepExpr.denote n) ^ 3 - 2 ^ 3) / stepExpr.denote n := by
  have hinverse := represented_inverse_of_nonzero n hstep
  have hreciprocal : reciprocalExpr.denote n = (stepExpr.denote n)⁻¹ := by
    apply mul_left_cancel₀ hstep
    simpa [hstep] using hinverse
  have hexpr : guardedQuotientExpr.denote n =
      ((2 + stepExpr.denote n) ^ 3 - 2 ^ 3) * reciprocalExpr.denote n := by
    dsimp only [guardedQuotientExpr, cubeExpr, Expr.denote, Expr.eval]
    norm_num
    ring_nf
    simp
  rw [hexpr, hreciprocal, div_eq_mul_inv]

theorem fallback_step_denote (n : ℕ) : fallbackStepExpr.denote n = epsilon n ^ 2 := by
  change epsilon n * epsilon n = epsilon n ^ 2
  ring

theorem fallback_quotient_denote (n : ℕ) :
    fallbackQuotientExpr.denote n =
      ((2 + fallbackStepExpr.denote n) ^ 3 - 2 ^ 3) / fallbackStepExpr.denote n := by
  dsimp only [fallbackQuotientExpr, fallbackStepExpr, cubeExpr, Expr.denote, Expr.eval,
    Laurent.monomial]
  norm_num
  ring_nf
  simp
  ring

def zeroObservation (choice : Bool) : Observation :=
  ⟨.eq, stepExpr, .constant 0, choice⟩

def branchSupport (choice : Bool) : Support :=
  if choice then [true, false] else [false, true]

def branchExpression (choice : Bool) : Expr :=
  if choice then fallbackQuotientExpr else guardedQuotientExpr

def selectedStep (choice : Bool) : Expr :=
  if choice then fallbackStepExpr else stepExpr

def domainProgram : Program :=
  .query .eq stepExpr (.constant 0)
    (.returnExpr fallbackQuotientExpr) (.returnExpr guardedQuotientExpr)

theorem branch_commit (choice : Bool) :
    commit Support.universe (zeroObservation choice) = some (branchSupport choice) := by
  cases choice <;> decide +kernel

theorem branch_valid (choice : Bool) : (branchExpression choice).valid = true := by
  cases choice <;> decide +kernel

theorem branch_nonempty (choice : Bool) : (branchSupport choice).nonempty = true := by
  cases choice <;> decide +kernel

theorem branch_extracts (choice : Bool) :
    standardPart (branchSupport choice) (branchExpression choice) = some 12 := by
  cases choice <;> decide +kernel

/-- On even indices the represented inverse is zero, so the multiplication
expression evaluates to zero. It does not justify cancelling a zero step. -/
theorem guarded_even_extracts_zero :
    standardPart (branchSupport true) guardedQuotientExpr = some 0 := by
  decide +kernel

theorem guarded_unrefined_unknown :
    standardPart Support.universe guardedQuotientExpr = none := by
  decide +kernel

def branchExecution (choice : Bool) (remaining : List Bool) : Execution :=
  { observations := [zeroObservation choice]
    support := branchSupport choice
    expression := branchExpression choice
    result := some 12
    unusedChoices := remaining }

theorem execute_cons (choice : Bool) (remaining : List Bool) :
    domainProgram.execute (choice :: remaining) = some (branchExecution choice remaining) := by
  have hc := branch_commit choice
  have hv := branch_valid choice
  have hn := branch_nonempty choice
  have he := branch_extracts choice
  cases choice <;>
    simp only [domainProgram, Program.execute, Program.executeFrom,
      zeroObservation, branchSupport, branchExpression, Bool.false_eq_true,
      ↓reduceIte] at hc hv hn he ⊢ <;>
    simp only [hc, Option.bind_some, hv, hn, Bool.and_self, ↓reduceIte, he,
      Option.map_some, branchExecution, zeroObservation, branchSupport,
      branchExpression, Bool.false_eq_true, ↓reduceIte]

theorem empty_stream_rejected : domainProgram.execute [] = none := rfl

theorem accepted_result {choices : List Bool} {execution : Execution}
    (h : domainProgram.execute choices = some execution) : execution.result = some 12 := by
  cases choices with
  | nil => simp [empty_stream_rejected] at h
  | cons choice remaining =>
      rw [execute_cons, Option.some.injEq] at h
      subst execution
      rfl

/-- Existence concerns one fixed completion for the complete accepted trace. -/
theorem accepted_realized {choices : List Bool} {execution : Execution}
    (h : domainProgram.execute choices = some execution) :
    ∃ completion : Completion execution.commitments,
      domainProgram.interpret completion.ultrafilter =
        (execution.observations, execution.expression) :=
  Program.execute_realized h

theorem accepted_standardPart {choices : List Bool} {execution : Execution}
    (h : domainProgram.execute choices = some execution) :
    ∀ completion : Completion execution.commitments,
      NearStandardAt completion.ultrafilter execution.expression.denote 12 :=
  Program.execute_standardPart_sound h (accepted_result h)

/-- The observed equality is actual eventual equality in the same completion,
not merely a comparison between names. -/
theorem observation_holds (choice : Bool) (remaining : List Bool)
    (completion : Completion (branchExecution choice remaining).commitments) :
    ∀ᶠ n in (completion.ultrafilter : Filter ℕ),
      if choice then stepExpr.denote n = 0 else stepExpr.denote n ≠ 0 := by
  have hmem := completion.contains
    (show (zeroObservation choice).denote ∈ (branchExecution choice remaining).commitments from
      ⟨zeroObservation choice, by simp [branchExecution], rfl⟩)
  change ∀ᶠ n in (completion.ultrafilter : Filter ℕ), n ∈ (zeroObservation choice).denote at hmem
  cases choice <;>
    simpa [zeroObservation, Observation.denote, comparisonSet, compareReal,
      Expr.denote, Expr.eval] using hmem

/-- Both branches have a nonzero selected step. On the zero branch this requires
replacement by ε². The original h remains zero in that branch's completion. -/
theorem selected_step_nonzero (choice : Bool) (remaining : List Bool)
    (completion : Completion (branchExecution choice remaining).commitments) :
    ∀ᶠ n in (completion.ultrafilter : Filter ℕ), (selectedStep choice).denote n ≠ 0 := by
  cases choice
  · simpa [selectedStep] using observation_holds false remaining completion
  · simpa [selectedStep, fallback_step_denote] using
      (epsilon_nonzero completion).mono (fun n hn => pow_ne_zero 2 hn)

/-- Replacement preserves infinitesimality. Each branch therefore uses a
nonzero infinitesimal, with the nonzero condition established separately. -/
theorem selected_step_infinitesimal (choice : Bool) {Γ : Commitments}
    (completion : Completion Γ) :
    NearStandardAt completion.ultrafilter (selectedStep choice).denote 0 := by
  have hzero : standardPart Support.universe (selectedStep choice) = some 0 := by
    cases choice <;> decide +kernel
  simpa using nearStandardAt_of_cofiniteLimit completion
    (standardPart_universe_cofinite hzero)

/-- The actual returned expression denotes the literal quotient with its
branch's selected nonzero step, in every completion of that same branch. -/
theorem accepted_quotient_correspondence {choice : Bool} {remaining : List Bool}
    {execution : Execution}
    (h : domainProgram.execute (choice :: remaining) = some execution)
    (completion : Completion execution.commitments) :
    ∀ᶠ n in (completion.ultrafilter : Filter ℕ),
      execution.expression.denote n =
        ((2 + (selectedStep choice).denote n) ^ 3 - 2 ^ 3) /
          (selectedStep choice).denote n := by
  rw [execute_cons, Option.some.injEq] at h
  subst execution
  cases choice
  · exact (selected_step_nonzero false remaining completion).mono
      (fun n hn => guarded_quotient_denote n hn)
  · exact Filter.Eventually.of_forall fallback_quotient_denote

/-- The represented reciprocal is an inverse precisely where the negative
observation supplies the missing nonzero condition. -/
theorem negative_branch_inverse (remaining : List Bool)
    (completion : Completion (branchExecution false remaining).commitments) :
    ∀ᶠ n in (completion.ultrafilter : Filter ℕ),
      stepExpr.denote n * reciprocalExpr.denote n = 1 :=
  (selected_step_nonzero false remaining completion).mono
    (fun n hn => represented_inverse_of_nonzero n hn)

/-- The zero answer rules out any inverse to the original step in that
completion. A fallback cannot be replaced by cancellation of h. -/
theorem zero_branch_has_no_inverse (remaining : List Bool)
    (completion : Completion (branchExecution true remaining).commitments)
    (candidate : Sequence) :
    ¬ (∀ᶠ n in (completion.ultrafilter : Filter ℕ), stepExpr.denote n * candidate n = 1) := by
  intro hinverse
  have hzero := observation_holds true remaining completion
  have hfalse : ∀ᶠ n in (completion.ultrafilter : Filter ℕ), False := by
    filter_upwards [hzero, hinverse] with n hn hi
    simp only [↓reduceIte] at hn
    simp [hn] at hi
  obtain ⟨_, impossible⟩ := Filter.Eventually.exists hfalse
  exact impossible

/-- Before refinement the represented quotient has limits 0 and 12 in
compatible completions. Its failed extraction reflects a real disagreement,
not merely insufficient search by the extractor. -/
theorem guarded_no_common_standardPart :
    ¬ ∃ r : ℝ, ∀ completion : Completion (∅ : Commitments),
      NearStandardAt completion.ultrafilter guardedQuotientExpr.denote r := by
  rintro ⟨r, hcommon⟩
  have hrun (choice : Bool) :
      run Support.universe [zeroObservation choice] = some (branchSupport choice) := by
    simp only [run, branch_commit, Option.bind_some]
  obtain ⟨evenCompletion, _⟩ := accepted_realized (execute_cons true [])
  obtain ⟨oddCompletion, _⟩ := accepted_realized (execute_cons false [])
  let evenFree : Completion (∅ : Commitments) :=
    ⟨evenCompletion.ultrafilter, evenCompletion.extendsCofinite, by simp⟩
  let oddFree : Completion (∅ : Commitments) :=
    ⟨oddCompletion.ultrafilter, oddCompletion.extendsCofinite, by simp⟩
  have heven : NearStandardAt evenCompletion.ultrafilter guardedQuotientExpr.denote 0 := by
    simpa using run_standardPart_sound (hrun true) guarded_even_extracts_zero evenCompletion
  have hodd : NearStandardAt oddCompletion.ultrafilter guardedQuotientExpr.denote 12 := by
    simpa [branchExecution, branchExpression] using
      accepted_standardPart (execute_cons false []) oddCompletion
  have hrzero : r = 0 := nearStandardAt_unique (hcommon evenFree) heven
  have hrtwelve : r = 12 := nearStandardAt_unique (hcommon oddFree) hodd
  linarith

#print axioms Hyperreals.DomainInfinitesimal.step_extracts_zero
#print axioms Hyperreals.DomainInfinitesimal.step_infinitesimal
#print axioms Hyperreals.DomainInfinitesimal.represented_inverse_of_nonzero
#print axioms Hyperreals.DomainInfinitesimal.guarded_quotient_denote
#print axioms Hyperreals.DomainInfinitesimal.fallback_quotient_denote
#print axioms Hyperreals.DomainInfinitesimal.branch_commit
#print axioms Hyperreals.DomainInfinitesimal.branch_extracts
#print axioms Hyperreals.DomainInfinitesimal.guarded_even_extracts_zero
#print axioms Hyperreals.DomainInfinitesimal.guarded_unrefined_unknown
#print axioms Hyperreals.DomainInfinitesimal.execute_cons
#print axioms Hyperreals.DomainInfinitesimal.accepted_result
#print axioms Hyperreals.DomainInfinitesimal.accepted_realized
#print axioms Hyperreals.DomainInfinitesimal.accepted_standardPart
#print axioms Hyperreals.DomainInfinitesimal.observation_holds
#print axioms Hyperreals.DomainInfinitesimal.selected_step_infinitesimal
#print axioms Hyperreals.DomainInfinitesimal.selected_step_nonzero
#print axioms Hyperreals.DomainInfinitesimal.accepted_quotient_correspondence
#print axioms Hyperreals.DomainInfinitesimal.negative_branch_inverse
#print axioms Hyperreals.DomainInfinitesimal.zero_branch_has_no_inverse
#print axioms Hyperreals.DomainInfinitesimal.guarded_no_common_standardPart

end Hyperreals.DomainInfinitesimal
