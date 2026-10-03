import Hyperreals.ResidueTrace
import Hyperreals.ResidueLimitDiagnostic

/-!
# Replay snapshots for concrete finite-period sessions

A snapshot contains only the recorded observations, final support, query, and
claimed extraction result. Its executable check replays the observations from
the universe and recomputes extraction. An equality proof for this check can be
produced by kernel reduction of a closed snapshot, independently of the native
checker's reported answer. Capturing the intended external session remains a
separate provenance obligation.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter

namespace Hyperreals.Residue.Replay

/-- Data exported from one session and one extraction query. It has no proof fields. -/
structure Snapshot where
  observations : List Observation
  support : Support
  expression : Expr
  result : Option Rat
  deriving Repr

/-- Replay and extraction must agree exactly with the exported data. The
validity guard also rejects invalid queries claiming an unknown result. -/
def Snapshot.check (snapshot : Snapshot) : Bool :=
  snapshot.expression.valid && snapshot.support.nonempty &&
    decide (run Support.universe snapshot.observations = some snapshot.support) &&
    decide (standardPart snapshot.support snapshot.expression = snapshot.result)

/-- The actual sequence observations recorded in the exported session. -/
def Snapshot.commitments (snapshot : Snapshot) : Commitments :=
  {A | ∃ observation ∈ snapshot.observations, A = observation.denote}

theorem Snapshot.check_iff (snapshot : Snapshot) :
    snapshot.check = true ↔
      snapshot.expression.valid = true ∧ snapshot.support.nonempty = true ∧
      run Support.universe snapshot.observations = some snapshot.support ∧
      standardPart snapshot.support snapshot.expression = snapshot.result := by
  simp only [Snapshot.check, Bool.and_eq_true, decide_eq_true_eq, and_assoc]

theorem Snapshot.checked_run {snapshot : Snapshot} (hcheck : snapshot.check = true) :
    run Support.universe snapshot.observations = some snapshot.support :=
  (snapshot.check_iff.mp hcheck).2.2.1

theorem Snapshot.checked_result {snapshot : Snapshot} (hcheck : snapshot.check = true) :
    standardPart snapshot.support snapshot.expression = snapshot.result :=
  (snapshot.check_iff.mp hcheck).2.2.2

/-- A successful replay has at least one free completion of its actual trace. -/
theorem Snapshot.trace_extendible {snapshot : Snapshot} (hcheck : snapshot.check = true) :
    Extendible snapshot.commitments :=
  run_trace_extendible (Snapshot.checked_run hcheck)

/-- The replayed final support describes exactly the free ultrafilters
containing every actual observation in the exported trace. -/
theorem Snapshot.support_mem_iff {snapshot : Snapshot} (hcheck : snapshot.check = true)
    (U : Ultrafilter ℕ) (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    snapshot.support.carrier ∈ U ↔
      ∀ observation ∈ snapshot.observations, observation.denote ∈ U :=
  run_universe_mem_iff (Snapshot.checked_run hcheck) U hfree

/-- A successful rational extraction holds in every completion of the actual
exported trace. The existence theorem above prevents a vacuous application. -/
theorem Snapshot.standardPart_sound {snapshot : Snapshot} {r : Rat}
    (hcheck : snapshot.check = true) (hresult : snapshot.result = some r) :
    ∀ C : Completion snapshot.commitments,
      NearStandardAt C.ultrafilter snapshot.expression.denote (r : ℝ) :=
  run_standardPart_sound (Snapshot.checked_run hcheck)
    ((Snapshot.checked_result hcheck).trans hresult)

/-- The executable rejection recorded by a successful replay. -/
theorem Snapshot.unknown_result {snapshot : Snapshot}
    (hcheck : snapshot.check = true) (hresult : snapshot.result = none) :
    standardPart snapshot.support snapshot.expression = none :=
  (Snapshot.checked_result hcheck).trans hresult

/-- Rejection rules out a common finite real standard part across all completions
of the actual trace. Individual completions or later refinements can still have one. -/
theorem Snapshot.no_common_standardPart {snapshot : Snapshot}
    (hcheck : snapshot.check = true) (hresult : snapshot.result = none) (r : ℝ) :
    ¬ ∀ C : Completion snapshot.commitments,
      NearStandardAt C.ultrafilter snapshot.expression.denote r := by
  intro hall
  apply standardPart_none_no_common_real (snapshot.check_iff.mp hcheck).1
    (snapshot.check_iff.mp hcheck).2.1 (Snapshot.unknown_result hcheck hresult) r
  intro U hfree hsupport
  apply hall ⟨U, hfree, ?_⟩
  rintro A ⟨observation, hmem, rfl⟩
  exact (Snapshot.support_mem_iff hcheck U hfree).mp hsupport observation hmem

/-- Every replayed rejection is classified by a proved mathematical reason. -/
theorem Snapshot.failure_classified {snapshot : Snapshot}
    (hcheck : snapshot.check = true) (hresult : snapshot.result = none) :
    (∃ residue, diagnoseStandardPart snapshot.support snapshot.expression = .divergent residue) ∨
      ∃ r q s v, diagnoseStandardPart snapshot.support snapshot.expression = .disagreement r q s v :=
  (standardPart_none_iff_diagnostic_failure (snapshot.check_iff.mp hcheck).1
    (snapshot.check_iff.mp hcheck).2.1).mp (Snapshot.unknown_result hcheck hresult)

#print axioms Hyperreals.Residue.Replay.Snapshot.check_iff
#print axioms Hyperreals.Residue.Replay.Snapshot.checked_run
#print axioms Hyperreals.Residue.Replay.Snapshot.checked_result
#print axioms Hyperreals.Residue.Replay.Snapshot.trace_extendible
#print axioms Hyperreals.Residue.Replay.Snapshot.support_mem_iff
#print axioms Hyperreals.Residue.Replay.Snapshot.standardPart_sound
#print axioms Hyperreals.Residue.Replay.Snapshot.unknown_result
#print axioms Hyperreals.Residue.Replay.Snapshot.no_common_standardPart
#print axioms Hyperreals.Residue.Replay.Snapshot.failure_classified

end Hyperreals.Residue.Replay
