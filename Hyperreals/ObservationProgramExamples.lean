import Hyperreals.ObservationPrograms

/-!
# Kernel-checked examples of adaptive observation programs

The two branches of `adaptive` ask different second queries and return different
expressions. Closed executable checks use kernel reduction. They cover accepted
paths, preserved unused choices, inconsistent choices, an exhausted stream,
unknown extraction, and invalid expressions.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.ObservationPrograms.Examples

open Residue

/-- The first answer determines both the next comparison and the returned leaf. -/
def adaptive : Program :=
  .query .lt (.constant 0) (.periodic [1, -1])
    (.query .lt (.constant 7) .index
      (.returnExpr (.add (.constant 7) .reciprocalIndex))
      (.returnExpr (.constant 99)))
    (.query .eq (.periodic [0, 1, 2]) (.constant 2)
      (.returnExpr (.constant 42))
      (.returnExpr (.constant 0)))

theorem odd_accepts : (adaptive.execute [false, true, false]).isSome = true := by
  decide +kernel

def oddExecution : Execution :=
  (adaptive.execute [false, true, false]).get odd_accepts

theorem odd_execution : adaptive.execute [false, true, false] = some oddExecution :=
  Option.eq_some_of_isSome odd_accepts

/-- The odd branch introduces period three and retains residue five modulo six. -/
theorem odd_details :
    oddExecution.support = [false, false, false, false, false, true] ∧
      oddExecution.observations.map Observation.comparison = [.lt, .eq] ∧
      oddExecution.observations.map Observation.choice = [false, true] ∧
      oddExecution.result = some 42 ∧ oddExecution.unusedChoices = [false] := by
  decide +kernel

theorem odd_realized :
    ∃ completion : Completion oddExecution.commitments,
      adaptive.interpret completion.ultrafilter =
        (oddExecution.observations, oddExecution.expression) :=
  Program.execute_realized odd_execution

theorem odd_standardPart : ∀ completion : Completion oddExecution.commitments,
    NearStandardAt completion.ultrafilter oddExecution.expression.denote 42 :=
  Program.execute_standardPart_sound odd_execution odd_details.2.2.2.1

theorem even_accepts : (adaptive.execute [true, true]).isSome = true := by
  decide +kernel

def evenExecution : Execution :=
  (adaptive.execute [true, true]).get even_accepts

theorem even_execution : adaptive.execute [true, true] = some evenExecution :=
  Option.eq_some_of_isSome even_accepts

/-- The even branch asks the eventual growth query and returns seven plus 1/n. -/
theorem even_details :
    evenExecution.support = [true, false] ∧
      evenExecution.observations.map Observation.comparison = [.lt, .lt] ∧
      evenExecution.observations.map Observation.choice = [true, true] ∧
      evenExecution.result = some 7 ∧ evenExecution.unusedChoices = [] := by
  decide +kernel

theorem even_realized :
    ∃ completion : Completion evenExecution.commitments,
      adaptive.interpret completion.ultrafilter =
        (evenExecution.observations, evenExecution.expression) :=
  Program.execute_realized even_execution

theorem even_standardPart : ∀ completion : Completion evenExecution.commitments,
    NearStandardAt completion.ultrafilter evenExecution.expression.denote 7 :=
  Program.execute_standardPart_sound even_execution even_details.2.2.2.1

/-- A later answer cannot reverse an earlier answer to the same actual query. -/
def repeatedQuery : Program :=
  .query .lt (.constant 0) (.periodic [1, -1])
    (.query .lt (.constant 0) (.periodic [1, -1])
      (.returnExpr (.constant 1)) (.returnExpr (.constant 2)))
    (.returnExpr (.constant 3))

theorem contradiction_rejected : (repeatedQuery.execute [true, false]).isNone = true := by
  decide +kernel

theorem cofinite_false_rejected : (adaptive.execute [true, false]).isNone = true := by
  decide +kernel

theorem exhausted_stream_rejected :
    (adaptive.execute []).isNone = true ∧ (adaptive.execute [false]).isNone = true := by
  decide +kernel

/-- An unknown standard part is distinct from a rejected program execution. -/
theorem unknown_succeeds :
    ((Program.returnExpr (.periodic [1, -1])).execute []).map
      (fun execution => execution.result) = some none := by
  decide +kernel

theorem invalid_leaf_rejected :
    ((Program.returnExpr (.periodic [])).execute []).isNone = true := by
  decide +kernel

theorem invalid_query_rejected :
    ((Program.query .eq (.periodic []) (.constant 0)
      (.returnExpr (.constant 1)) (.returnExpr (.constant 2))).execute [true]).isNone = true := by
  decide +kernel

#print axioms Hyperreals.ObservationPrograms.Examples.odd_execution
#print axioms Hyperreals.ObservationPrograms.Examples.odd_details
#print axioms Hyperreals.ObservationPrograms.Examples.odd_realized
#print axioms Hyperreals.ObservationPrograms.Examples.odd_standardPart
#print axioms Hyperreals.ObservationPrograms.Examples.even_execution
#print axioms Hyperreals.ObservationPrograms.Examples.even_details
#print axioms Hyperreals.ObservationPrograms.Examples.even_realized
#print axioms Hyperreals.ObservationPrograms.Examples.even_standardPart
#print axioms Hyperreals.ObservationPrograms.Examples.contradiction_rejected
#print axioms Hyperreals.ObservationPrograms.Examples.cofinite_false_rejected
#print axioms Hyperreals.ObservationPrograms.Examples.exhausted_stream_rejected
#print axioms Hyperreals.ObservationPrograms.Examples.unknown_succeeds
#print axioms Hyperreals.ObservationPrograms.Examples.invalid_leaf_rejected
#print axioms Hyperreals.ObservationPrograms.Examples.invalid_query_rejected

end Hyperreals.ObservationPrograms.Examples
