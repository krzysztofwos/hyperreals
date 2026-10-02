import Hyperreals.ObservationPrograms
import Hyperreals.InfinitesimalCase

/-!
# An adaptive infinitesimal quotient with a common ordinary output

The program queries whether (-1)^n is negative. A positive answer selects the
literal backward difference quotient for x³ at 2. A negative answer selects the
literal forward quotient. The two quotients lie on opposite sides of 12, yet
every accepted execution computes standard part 12. The proof covers every
proposed choice stream and retains the actual recorded branch observation.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.AdaptiveInfinitesimal

open Residue ObservationPrograms InfinitesimalCase

noncomputable def backwardQuotient : Sequence :=
  fun n => (2 ^ 3 - (2 - epsilon n) ^ 3) / epsilon n

/-- The backward numerator is kept as a literal difference of two cubes. -/
def backwardQuotientExpr : Expr :=
  .divMonomial
    (.sub (cubeExpr (.constant 2)) (cubeExpr (.sub (.constant 2) .reciprocalIndex)))
    1 (.negSucc 0)

theorem backwardQuotientExpr_denote (n : ℕ) :
    backwardQuotientExpr.denote n = backwardQuotient n := by
  dsimp only [backwardQuotientExpr, cubeExpr, Expr.denote, Expr.eval,
    Laurent.monomial, backwardQuotient, epsilon, reciprocalIndex]
  norm_num
  ring_nf
  simp

theorem backward_identity (n : ℕ) (hn : 0 < n) :
    backwardQuotient n = 12 - 6 * epsilon n + epsilon n ^ 2 := by
  have hnonzero : epsilon n ≠ 0 := ne_of_gt (epsilon_pos n hn)
  unfold backwardQuotient
  field_simp
  ring

theorem backward_strictly_below_derivative (n : ℕ) (hn : 0 < n) :
    backwardQuotient n < 12 := by
  rw [backward_identity n hn]
  have hpositive := epsilon_pos n hn
  have hsmall : epsilon n ≤ 1 := by
    unfold epsilon reciprocalIndex
    apply inv_le_one_of_one_le₀
    exact_mod_cast hn
  nlinarith [mul_nonneg hpositive.le (sub_nonneg.mpr hsmall)]

private theorem eventually_index_pos : ∀ᶠ n : ℕ in Filter.cofinite, 0 < n := by
  rw [Nat.cofinite_eq_atTop]
  exact eventually_gt_atTop 0

theorem backward_cofiniteLimit : CofiniteLimit backwardQuotient 12 := by
  have hexpanded : CofiniteLimit (fun n => 12 - 6 * epsilon n + epsilon n ^ 2) 12 := by
    have h := ((CofiniteLimit.constant 12).sub
      ((CofiniteLimit.constant 6).mul reciprocalIndex_cofiniteLimit)).add
      (reciprocalIndex_cofiniteLimit.pow 2)
    simpa [Sequence.constant, epsilon] using h
  have heq : backwardQuotient =ᶠ[Filter.cofinite]
      (fun n => 12 - 6 * epsilon n + epsilon n ^ 2) :=
    eventually_index_pos.mono fun n hn => backward_identity n hn
  exact hexpanded.congr' heq.symm

theorem backward_standardPart {Γ : Commitments} (completion : Completion Γ) :
    NearStandardAt completion.ultrafilter backwardQuotient 12 :=
  nearStandardAt_of_cofiniteLimit completion backward_cofiniteLimit

/-- The backward error is negative in every free completion, although its
standard part vanishes. -/
theorem backward_below_in_completion {Γ : Commitments} (completion : Completion Γ) :
    ∀ᶠ n in (completion.ultrafilter : Filter ℕ), backwardQuotient n < 12 :=
  completion.extendsCofinite (eventually_index_pos.mono
    fun n hn => backward_strictly_below_derivative n hn)

theorem forward_above_in_completion {Γ : Commitments} (completion : Completion Γ) :
    ∀ᶠ n in (completion.ultrafilter : Filter ℕ), 12 < quotient n :=
  completion.extendsCofinite (eventually_index_pos.mono
    fun n hn => quotient_strictly_above_derivative n hn)

theorem quotients_distinct (n : ℕ) (hn : 0 < n) :
    backwardQuotient n < quotient n :=
  (backward_strictly_below_derivative n hn).trans
    (quotient_strictly_above_derivative n hn)

/-- True means the odd residue was selected, and selects the backward quotient. -/
def signObservation (choice : Bool) : Observation :=
  ⟨.lt, .periodic [1, -1], .constant 0, choice⟩

def branchSupport (choice : Bool) : Support :=
  if choice then [false, true] else [true, false]

def branchExpression (choice : Bool) : Expr :=
  if choice then backwardQuotientExpr else quotientExpr

/-- The completion-dependent answer chooses which literal quotient to compute. -/
def adaptiveProgram : Program :=
  .query .lt (.periodic [1, -1]) (.constant 0)
    (.returnExpr backwardQuotientExpr) (.returnExpr quotientExpr)

theorem branch_commit (choice : Bool) :
    commit Support.universe (signObservation choice) = some (branchSupport choice) := by
  cases choice <;> decide +kernel

theorem branch_valid (choice : Bool) : (branchExpression choice).valid = true := by
  cases choice <;> decide +kernel

theorem branch_nonempty (choice : Bool) : (branchSupport choice).nonempty = true := by
  cases choice <;> decide +kernel

/-- Both original ASTs are evaluated by the executable extractor on their
actual branch support. No expansion or limit assertion is supplied. -/
theorem branch_extracts (choice : Bool) :
    standardPart (branchSupport choice) (branchExpression choice) = some 12 := by
  cases choice <;> decide +kernel

def branchExecution (choice : Bool) (remaining : List Bool) : Execution :=
  { observations := [signObservation choice]
    support := branchSupport choice
    expression := branchExpression choice
    result := some 12
    unusedChoices := remaining }

/-- One proposed answer is consumed. Every suffix is left unused. This equality
is proved from the executable commit and extraction checks above. -/
theorem execute_cons (choice : Bool) (remaining : List Bool) :
    adaptiveProgram.execute (choice :: remaining) = some (branchExecution choice remaining) := by
  have hc := branch_commit choice
  have hv := branch_valid choice
  have hn := branch_nonempty choice
  have he := branch_extracts choice
  cases choice <;>
    simp only [adaptiveProgram, Program.execute, Program.executeFrom,
      signObservation, branchSupport, branchExpression, Bool.false_eq_true,
      ↓reduceIte] at hc hv hn he ⊢ <;>
    simp only [hc, Option.bind_some, hv, hn, Bool.and_self, ↓reduceIte, he,
      Option.map_some, branchExecution, signObservation, branchSupport,
      branchExpression, Bool.false_eq_true, ↓reduceIte]

theorem empty_stream_rejected : adaptiveProgram.execute [] = none := rfl

/-- Uniform computational result for every accepted proposed choice stream,
rather than a claim restricted to two selected example snapshots. -/
theorem accepted_result {choices : List Bool} {execution : Execution}
    (h : adaptiveProgram.execute choices = some execution) : execution.result = some 12 := by
  cases choices with
  | nil => simp [empty_stream_rejected] at h
  | cons choice remaining =>
      rw [execute_cons, Option.some.injEq] at h
      subst execution
      rfl

/-- Each accepted adaptive run has one fixed compatible free completion that
reproduces its entire selected trace and literal leaf expression. -/
theorem accepted_realized {choices : List Bool} {execution : Execution}
    (h : adaptiveProgram.execute choices = some execution) :
    ∃ completion : Completion execution.commitments,
      adaptiveProgram.interpret completion.ultrafilter =
        (execution.observations, execution.expression) :=
  Program.execute_realized h

/-- Every accepted execution returns 12 in every completion of its own trace.
The positive and negative runs need not share one completion. -/
theorem accepted_standardPart {choices : List Bool} {execution : Execution}
    (h : adaptiveProgram.execute choices = some execution) :
    ∀ completion : Completion execution.commitments,
      NearStandardAt completion.ultrafilter execution.expression.denote 12 :=
  Program.execute_standardPart_sound h (accepted_result h)

/-- Even after choosing a branch, the literal quotient lies strictly on the
specified side of 12 in every compatible completion. -/
theorem accepted_branch_side {choice : Bool} {remaining : List Bool} {execution : Execution}
    (h : adaptiveProgram.execute (choice :: remaining) = some execution)
    (completion : Completion execution.commitments) :
    if choice then
      ∀ᶠ n in (completion.ultrafilter : Filter ℕ), execution.expression.denote n < 12
    else
      ∀ᶠ n in (completion.ultrafilter : Filter ℕ), 12 < execution.expression.denote n := by
  rw [execute_cons, Option.some.injEq] at h
  subst execution
  cases choice
  · simpa only [Bool.false_eq_true, ↓reduceIte, branchExecution, branchExpression,
      quotientExpr_denote] using forward_above_in_completion completion
  · simpa only [↓reduceIte, branchExecution, branchExpression,
      backwardQuotientExpr_denote] using backward_below_in_completion completion

#print axioms Hyperreals.AdaptiveInfinitesimal.backwardQuotientExpr_denote
#print axioms Hyperreals.AdaptiveInfinitesimal.backward_identity
#print axioms Hyperreals.AdaptiveInfinitesimal.backward_strictly_below_derivative
#print axioms Hyperreals.AdaptiveInfinitesimal.backward_standardPart
#print axioms Hyperreals.AdaptiveInfinitesimal.backward_below_in_completion
#print axioms Hyperreals.AdaptiveInfinitesimal.forward_above_in_completion
#print axioms Hyperreals.AdaptiveInfinitesimal.quotients_distinct
#print axioms Hyperreals.AdaptiveInfinitesimal.branch_commit
#print axioms Hyperreals.AdaptiveInfinitesimal.branch_valid
#print axioms Hyperreals.AdaptiveInfinitesimal.branch_nonempty
#print axioms Hyperreals.AdaptiveInfinitesimal.branch_extracts
#print axioms Hyperreals.AdaptiveInfinitesimal.execute_cons
#print axioms Hyperreals.AdaptiveInfinitesimal.empty_stream_rejected
#print axioms Hyperreals.AdaptiveInfinitesimal.accepted_result
#print axioms Hyperreals.AdaptiveInfinitesimal.accepted_realized
#print axioms Hyperreals.AdaptiveInfinitesimal.accepted_standardPart
#print axioms Hyperreals.AdaptiveInfinitesimal.accepted_branch_side

end Hyperreals.AdaptiveInfinitesimal
