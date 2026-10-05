import Hyperreals.DifferentiableInfinitesimal
import Hyperreals.DomainInfinitesimal
import Mathlib.Tactic.FinCases

/-! # Elementary vector functions and an observation-dependent infinitesimal step -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

open Filter Topology

namespace Hyperreals.DifferentiableExamples

open Hyperreals.Differentiable

private def x : Expr 2 := .var 0
private def y : Expr 2 := .var 1
private def one : Expr 2 := .constant 1

/-- A vector example whose derivative contains irrational symbolic values. -/
def vectorExample : Program 2 3 := ![
  .mul (.sin x) (.exp y),
  .log (.add (.add one (.mul x x)) (.mul y y)),
  .div (.sqrt (.add one (.mul x x))) (.add one (.mul y y))]

def vectorDirection : Fin 2 → Expr 2 := ![.constant 1, .constant 2]

theorem vectorExample_domain : vectorExample.Domain ![1, 0] := by
  intro i
  fin_cases i <;> norm_num [vectorExample, Program.Domain, Expr.Domain, Expr.eval, x, y, one]

theorem vectorExample_derivative :
    (vectorExample.jvp vectorDirection).eval ![1, 0] =
      ![Real.cos 1 + 2 * Real.sin 1, 1, 1 / Real.sqrt 2] := by
  funext i
  fin_cases i <;>
    norm_num [vectorExample, vectorDirection, Program.jvp, Program.eval,
      Expr.jvp, Expr.eval, x, y, one] <;> ring

/-- Reuse the checked observation's nonzero step for any expression in the new language.
The trace still belongs to the residue observation engine. The quotient is interpreted
by the new language's semantics, not by Laurent normalization. -/
theorem quotient_after_step_observation {n : Nat} (e : Expr n)
    (direction : Fin n → Expr n) (point : Fin n → ℝ) (hdomain : e.Domain point)
    (choice : Bool) (remaining : List Bool)
    (C : Completion (DomainInfinitesimal.branchExecution choice remaining).commitments) :
    NearStandardAt C.ultrafilter
      (e.quotient direction point (DomainInfinitesimal.selectedStep choice).denote)
      ((e.jvp direction).eval point) :=
  e.quotient_standardPart direction point hdomain C _
    (DomainInfinitesimal.selected_step_infinitesimal choice C)
    (DomainInfinitesimal.selected_step_nonzero choice remaining C)

/-- Both accepted step choices give the same elementary vector derivative. -/
theorem vectorExample_after_observation (choice : Bool) (remaining : List Bool)
    (C : Completion (DomainInfinitesimal.branchExecution choice remaining).commitments)
    (i : Fin 3) :
    NearStandardAt C.ultrafilter
      ((vectorExample i).quotient vectorDirection ![1, 0]
        (DomainInfinitesimal.selectedStep choice).denote)
      (![Real.cos 1 + 2 * Real.sin 1, 1, 1 / Real.sqrt 2] i) := by
  have h := quotient_after_step_observation (vectorExample i) vectorDirection ![1, 0]
    (vectorExample_domain i) choice remaining C
  have heq := congrFun vectorExample_derivative i
  exact heq ▸ h

#print axioms vectorExample_domain
#print axioms vectorExample_derivative
#print axioms quotient_after_step_observation
#print axioms vectorExample_after_observation

end Hyperreals.DifferentiableExamples
