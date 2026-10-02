import Hyperreals.ResidueTrace
import Mathlib.Data.Set.Finite.Lattice

/-!
# Finite adaptive programs with certified observations

A program is a finite decision tree. Its executable interpreter consumes proposed
Boolean choices and checks each choice using the residue runtime. Later queries
and the returned expression may depend on earlier choices. Invalid queried or
returned expressions, exhausted choice streams, and inconsistent choices cause
failure. Unvisited branches are not checked. A successful execution can still
have an unknown standard part.

The classical interpretation follows the membership decisions of one ultrafilter.
The soundness theorem derives agreement with this interpretation from the actual
checked execution. No ultrafilter or convergence proof is supplied to execution.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter

namespace Hyperreals.ObservationPrograms

open Residue

/-- Finite syntax whose remaining computation can depend on an observed answer. -/
inductive Program where
  | returnExpr (expression : Expr)
  | query (comparison : Comparison) (left right : Expr) (onTrue onFalse : Program)

/-- Executable output data. The trace contains only the selected query branches. -/
structure Execution where
  observations : List Observation
  support : Support
  expression : Expr
  result : Option Rat
  unusedChoices : List Bool
  deriving Repr

/-- Interpret proposed choices through checked commits. Unknown extraction is
represented by a successful execution whose `result` is `none`. -/
def Program.executeFrom : Program → Support → List Bool → Option Execution
  | .returnExpr expression, support, choices =>
      if expression.valid && support.nonempty then
        some ⟨[], support, expression, standardPart support expression, choices⟩
      else none
  | .query _ _ _ _ _, _, [] => none
  | .query comparison left right onTrue onFalse, support, choice :: choices =>
      let observation : Observation := ⟨comparison, left, right, choice⟩
      (commit support observation).bind fun next =>
        (if choice then onTrue.executeFrom next choices
          else onFalse.executeFrom next choices).map fun execution =>
          { execution with observations := observation :: execution.observations }

/-- Start with no commitments. The choice stream proposes answers, not proofs. -/
def Program.execute (program : Program) (choices : List Bool) : Option Execution :=
  program.executeFrom Support.universe choices

/-- Syntactic path relation recording every branch selected in a program. -/
inductive Program.Path : Program → List Observation → Expr → Prop where
  | returnExpr (expression : Expr) : Path (.returnExpr expression) [] expression
  | queryTrue {comparison : Comparison} {left right : Expr} {onTrue onFalse : Program}
      {observations : List Observation} {expression : Expr}
      (tail : Path onTrue observations expression) :
      Path (.query comparison left right onTrue onFalse)
        (⟨comparison, left, right, true⟩ :: observations) expression
  | queryFalse {comparison : Comparison} {left right : Expr} {onTrue onFalse : Program}
      {observations : List Observation} {expression : Expr}
      (tail : Path onFalse observations expression) :
      Path (.query comparison left right onTrue onFalse)
        (⟨comparison, left, right, false⟩ :: observations) expression

/-- The mathematical observation family recorded by an execution. -/
def Execution.commitments (execution : Execution) : Commitments :=
  {A | ∃ observation ∈ execution.observations, A = observation.denote}

/-- Every accepted program execution replays through the existing checked
runtime, follows its source tree, and records the actual extraction result. -/
theorem Program.executeFrom_spec {program : Program} {support : Support}
    {choices : List Bool} {execution : Execution}
    (h : program.executeFrom support choices = some execution) :
    run support execution.observations = some execution.support ∧
      program.Path execution.observations execution.expression ∧
      execution.expression.valid = true ∧
      standardPart execution.support execution.expression = execution.result := by
  induction program generalizing support choices execution with
  | returnExpr expression =>
      simp only [executeFrom] at h
      split at h
      next hvalid =>
        cases h
        have hparts : expression.valid = true ∧ support.nonempty = true := by
          simpa only [Bool.and_eq_true] using hvalid
        exact ⟨rfl, .returnExpr expression, hparts.1, rfl⟩
      next => contradiction
  | query comparison left right onTrue onFalse trueHypothesis falseHypothesis =>
      cases choices with
      | nil => simp [executeFrom] at h
      | cons choice choices =>
          cases hcommit : commit support ⟨comparison, left, right, choice⟩ with
          | none => simp [executeFrom, hcommit] at h
          | some next =>
              cases choice <;> simp only [executeFrom, hcommit, Option.bind_some,
                Bool.false_eq_true, ↓reduceIte] at h
              · cases htail : onFalse.executeFrom next choices with
                | none => simp [htail] at h
                | some tail =>
                    simp only [htail, Option.map_some, Option.some.injEq] at h
                    subst execution
                    obtain ⟨hrun, hpath, hvalid, hresult⟩ := falseHypothesis htail
                    exact ⟨by simpa only [run, hcommit, Option.bind_some] using hrun,
                      .queryFalse hpath, hvalid, hresult⟩
              · cases htail : onTrue.executeFrom next choices with
                | none => simp [htail] at h
                | some tail =>
                    simp only [htail, Option.map_some, Option.some.injEq] at h
                    subst execution
                    obtain ⟨hrun, hpath, hvalid, hresult⟩ := trueHypothesis htail
                    exact ⟨by simpa only [run, hcommit, Option.bind_some] using hrun,
                      .queryTrue hpath, hvalid, hresult⟩

/-- Classical execution consults one fixed ultrafilter at every adaptive query.
It returns the full branch trace as well as the selected leaf expression. -/
noncomputable def Program.interpret (program : Program) (U : Ultrafilter ℕ) :
    List Observation × Expr := by
  classical
  exact match program with
  | .returnExpr expression => ([], expression)
  | .query comparison left right onTrue onFalse =>
      if comparisonSet comparison left right ∈ U then
        let tail := onTrue.interpret U
        (⟨comparison, left, right, true⟩ :: tail.1, tail.2)
      else
        let tail := onFalse.interpret U
        (⟨comparison, left, right, false⟩ :: tail.1, tail.2)

/-- A completion containing the actual selected observations reproduces every
branch of the syntactic path, including queries chosen by earlier answers. -/
theorem Program.Path.interpret_eq {program : Program} {observations : List Observation}
    {expression : Expr} (hpath : program.Path observations expression)
    (U : Ultrafilter ℕ) (hmem : ∀ observation ∈ observations, observation.denote ∈ U) :
    program.interpret U = (observations, expression) := by
  induction hpath with
  | returnExpr => rfl
  | @queryTrue comparison left right onTrue onFalse observations expression _ ih =>
      have hselected : comparisonSet comparison left right ∈ U := by
        simpa [Observation.denote] using hmem ⟨comparison, left, right, true⟩ (by simp)
      have htail := ih (fun observation h => hmem observation (by simp [h]))
      simp only [Program.interpret, hselected, ↓reduceIte, htail]
  | @queryFalse comparison left right onTrue onFalse observations expression _ ih =>
      have hselected : comparisonSet comparison left right ∉ U := by
        apply Ultrafilter.compl_mem_iff_notMem.mp
        simpa [Observation.denote] using hmem ⟨comparison, left, right, false⟩ (by simp)
      have htail := ih (fun observation h => hmem observation (by simp [h]))
      simp only [Program.interpret, hselected, ↓reduceIte, htail]

/-- Every completion of an accepted execution reproduces the entire adaptive
branch trace under a single fixed ultrafilter. -/
theorem Program.execute_interpret {program : Program} {choices : List Bool}
    {execution : Execution} (h : program.execute choices = some execution)
    (completion : Completion execution.commitments) :
    program.interpret completion.ultrafilter =
      (execution.observations, execution.expression) := by
  apply (Program.executeFrom_spec h).2.1.interpret_eq
  intro observation hmem
  exact completion.contains ⟨observation, hmem, rfl⟩

/-- An accepted execution has at least one single free completion realizing all
its adaptive decisions. The theorem does not compute that completion. -/
theorem Program.execute_realized {program : Program} {choices : List Bool}
    {execution : Execution} (h : program.execute choices = some execution) :
    ∃ completion : Completion execution.commitments,
      program.interpret completion.ultrafilter =
        (execution.observations, execution.expression) := by
  obtain ⟨completion⟩ := run_trace_extendible (Program.executeFrom_spec h).1
  exact ⟨completion, Program.execute_interpret h completion⟩

/-- The reported rational is valid in every completion of the actual program
trace. Extraction supplies no new branch choice. -/
theorem Program.execute_standardPart_sound {program : Program} {choices : List Bool}
    {execution : Execution} {r : Rat} (h : program.execute choices = some execution)
    (hresult : execution.result = some r) :
    ∀ completion : Completion execution.commitments,
      NearStandardAt completion.ultrafilter execution.expression.denote (r : ℝ) :=
  run_standardPart_sound (Program.executeFrom_spec h).1
    ((Program.executeFrom_spec h).2.2.2.trans hresult)

/-- Unknown records the actual extractor outcome, without asserting that every
individual completion lacks a finite standard part. -/
theorem Program.execute_unknown {program : Program} {choices : List Bool}
    {execution : Execution} (h : program.execute choices = some execution)
    (hresult : execution.result = none) :
    standardPart execution.support execution.expression = none :=
  (Program.executeFrom_spec h).2.2.2.trans hresult

#print axioms Hyperreals.ObservationPrograms.Program.executeFrom_spec
#print axioms Hyperreals.ObservationPrograms.Program.Path.interpret_eq
#print axioms Hyperreals.ObservationPrograms.Program.execute_interpret
#print axioms Hyperreals.ObservationPrograms.Program.execute_realized
#print axioms Hyperreals.ObservationPrograms.Program.execute_standardPart_sound
#print axioms Hyperreals.ObservationPrograms.Program.execute_unknown

end Hyperreals.ObservationPrograms

namespace Hyperreals

/-- A monotone run whose every prefix is extendible has one free completion of
the entire union. This classical compactness consequence does not compute a
witness, require a common index in all committed sets, or certify unexamined
future transitions from a finite replay. -/
theorem extendible_iUnion_of_monotone (commitments : ℕ → Commitments)
    (hmono : Monotone commitments) (hprefix : ∀ n, Extendible (commitments n)) :
    Extendible (⋃ n, commitments n) := by
  apply extendible_of_hasFreeFIP
  intro finite hfinite
  have hdirected : Directed (· ⊆ ·) (fun n => completionBasis (commitments n)) := by
    intro i j
    refine ⟨max i j, ?_, ?_⟩
    · exact Set.union_subset_union_left _ (hmono (Nat.le_max_left i j))
    · exact Set.union_subset_union_left _ (hmono (Nat.le_max_right i j))
  have hinUnion : (↑finite : Set (Set ℕ)) ⊆ ⋃ n, completionBasis (commitments n) := by
    intro A hA
    rcases hfinite hA with hcommitted | hcofinite
    · obtain ⟨n, hn⟩ := Set.mem_iUnion.mp hcommitted
      exact Set.mem_iUnion.mpr ⟨n, Or.inl hn⟩
    · exact Set.mem_iUnion.mpr ⟨0, Or.inr hcofinite⟩
  obtain ⟨n, hn⟩ := hdirected.exists_mem_subset_of_finset_subset_biUnion hinUnion
  exact (hasFreeFIP_iff_extendible.mpr (hprefix n)) finite hn

#print axioms Hyperreals.extendible_iUnion_of_monotone

end Hyperreals
