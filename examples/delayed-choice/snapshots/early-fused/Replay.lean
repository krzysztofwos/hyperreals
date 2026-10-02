import Hyperreals.ResidueReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false
set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

open Hyperreals Hyperreals.Residue Hyperreals.Residue.Replay

namespace Hyperreals.GeneratedReplay

def snapshot : Snapshot := {
  observations := [⟨.eq, (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat))]), (.constant ((1 : Rat) / (1 : Rat))), true⟩]
  support := [false, true, false, true]
  expression := (.divMonomial (.add (.divMonomial (.sub (.add (.add (.mul (.mul (.constant ((1 : Rat) / (1 : Rat))) (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex)) (.mul (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex) (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex))) (.mul (.periodic [((3 : Rat) / (1 : Rat)), ((-2 : Rat) / (1 : Rat)), ((5 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((-4 : Rat) / (1 : Rat)), ((7 : Rat) / (1 : Rat))]) (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex))) (.periodic [((1 : Rat) / (3 : Rat)), ((-2 : Rat) / (5 : Rat)), ((7 : Rat) / (4 : Rat)), ((0 : Rat) / (1 : Rat))])) (.add (.add (.mul (.mul (.constant ((1 : Rat) / (1 : Rat))) (.constant ((2 : Rat) / (1 : Rat)))) (.mul (.constant ((2 : Rat) / (1 : Rat))) (.constant ((2 : Rat) / (1 : Rat))))) (.mul (.periodic [((3 : Rat) / (1 : Rat)), ((-2 : Rat) / (1 : Rat)), ((5 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((-4 : Rat) / (1 : Rat)), ((7 : Rat) / (1 : Rat))]) (.constant ((2 : Rat) / (1 : Rat))))) (.periodic [((1 : Rat) / (3 : Rat)), ((-2 : Rat) / (5 : Rat)), ((7 : Rat) / (4 : Rat)), ((0 : Rat) / (1 : Rat))]))) ((1 : Rat) / (1 : Rat)) (-1 : Int)) (.divMonomial (.sub (.add (.sub (.mul (.mul (.constant ((1 : Rat) / (1 : Rat))) (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex)) (.mul (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex) (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex))) (.mul (.periodic [((3 : Rat) / (1 : Rat)), ((-2 : Rat) / (1 : Rat)), ((5 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((-4 : Rat) / (1 : Rat)), ((7 : Rat) / (1 : Rat))]) (.add (.constant ((2 : Rat) / (1 : Rat))) .reciprocalIndex))) (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (7 : Rat)), ((-2 : Rat) / (3 : Rat)), ((5 : Rat) / (2 : Rat)), ((-1 : Rat) / (4 : Rat))])) (.add (.sub (.mul (.mul (.constant ((1 : Rat) / (1 : Rat))) (.constant ((2 : Rat) / (1 : Rat)))) (.mul (.constant ((2 : Rat) / (1 : Rat))) (.constant ((2 : Rat) / (1 : Rat))))) (.mul (.periodic [((3 : Rat) / (1 : Rat)), ((-2 : Rat) / (1 : Rat)), ((5 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((-4 : Rat) / (1 : Rat)), ((7 : Rat) / (1 : Rat))]) (.constant ((2 : Rat) / (1 : Rat))))) (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (7 : Rat)), ((-2 : Rat) / (3 : Rat)), ((5 : Rat) / (2 : Rat)), ((-1 : Rat) / (4 : Rat))]))) ((1 : Rat) / (1 : Rat)) (-1 : Int))) ((2 : Rat) / (1 : Rat)) (0 : Int))
  result := some ((12 : Rat) / (1 : Rat))
}

theorem replay_check : snapshot.check = true := by decide +kernel

theorem trace_consistent : Extendible snapshot.commitments :=
  Snapshot.trace_extendible replay_check

theorem extraction_matches : standardPart snapshot.support snapshot.expression = snapshot.result :=
  Snapshot.checked_result replay_check

theorem support_correspondence (U : Ultrafilter ℕ)
    (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    snapshot.support.carrier ∈ U ↔
      ∀ observation ∈ snapshot.observations, observation.denote ∈ U :=
  Snapshot.support_mem_iff replay_check U hfree

theorem replayed_standard_part :
    ∀ C : Completion snapshot.commitments,
      NearStandardAt C.ultrafilter snapshot.expression.denote ((((12 : Rat) / (1 : Rat)) : Rat) : ℝ) :=
  Snapshot.standardPart_sound replay_check rfl

#print axioms Hyperreals.GeneratedReplay.replay_check
#print axioms Hyperreals.GeneratedReplay.trace_consistent
#print axioms Hyperreals.GeneratedReplay.extraction_matches
#print axioms Hyperreals.GeneratedReplay.support_correspondence
#print axioms Hyperreals.GeneratedReplay.replayed_standard_part

end Hyperreals.GeneratedReplay
