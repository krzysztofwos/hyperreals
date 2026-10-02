import Hyperreals.ResidueReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false
set_option maxRecDepth 8192
set_option maxHeartbeats 2000000

open Hyperreals Hyperreals.Residue Hyperreals.Residue.Replay

namespace Hyperreals.GeneratedReplay

def snapshot : Snapshot := {
  observations := [⟨.eq, (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat))]), (.constant ((1 : Rat) / (1 : Rat))), true⟩,
    ⟨.eq, (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((2 : Rat) / (1 : Rat)), ((3 : Rat) / (1 : Rat)), ((4 : Rat) / (1 : Rat)), ((5 : Rat) / (1 : Rat))]), (.constant ((5 : Rat) / (1 : Rat))), true⟩,
    ⟨.eq, (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((2 : Rat) / (1 : Rat)), ((3 : Rat) / (1 : Rat)), ((4 : Rat) / (1 : Rat))]), (.constant ((3 : Rat) / (1 : Rat))), true⟩,
    ⟨.eq, (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((2 : Rat) / (1 : Rat)), ((3 : Rat) / (1 : Rat))]), (.constant ((3 : Rat) / (1 : Rat))), true⟩]
  support := [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]
  expression := (.periodic [((0 : Rat) / (1 : Rat)), ((1 : Rat) / (1 : Rat)), ((2 : Rat) / (1 : Rat)), ((3 : Rat) / (1 : Rat)), ((4 : Rat) / (1 : Rat)), ((5 : Rat) / (1 : Rat)), ((6 : Rat) / (1 : Rat)), ((7 : Rat) / (1 : Rat)), ((8 : Rat) / (1 : Rat)), ((9 : Rat) / (1 : Rat)), ((10 : Rat) / (1 : Rat)), ((11 : Rat) / (1 : Rat)), ((12 : Rat) / (1 : Rat)), ((13 : Rat) / (1 : Rat)), ((14 : Rat) / (1 : Rat)), ((15 : Rat) / (1 : Rat)), ((16 : Rat) / (1 : Rat)), ((17 : Rat) / (1 : Rat)), ((18 : Rat) / (1 : Rat)), ((19 : Rat) / (1 : Rat)), ((20 : Rat) / (1 : Rat)), ((21 : Rat) / (1 : Rat)), ((22 : Rat) / (1 : Rat)), ((23 : Rat) / (1 : Rat)), ((24 : Rat) / (1 : Rat)), ((25 : Rat) / (1 : Rat)), ((26 : Rat) / (1 : Rat)), ((27 : Rat) / (1 : Rat)), ((28 : Rat) / (1 : Rat)), ((29 : Rat) / (1 : Rat)), ((30 : Rat) / (1 : Rat)), ((31 : Rat) / (1 : Rat)), ((32 : Rat) / (1 : Rat)), ((33 : Rat) / (1 : Rat)), ((34 : Rat) / (1 : Rat)), ((35 : Rat) / (1 : Rat)), ((36 : Rat) / (1 : Rat)), ((37 : Rat) / (1 : Rat)), ((38 : Rat) / (1 : Rat)), ((39 : Rat) / (1 : Rat)), ((40 : Rat) / (1 : Rat)), ((41 : Rat) / (1 : Rat)), ((42 : Rat) / (1 : Rat)), ((43 : Rat) / (1 : Rat)), ((44 : Rat) / (1 : Rat)), ((45 : Rat) / (1 : Rat)), ((46 : Rat) / (1 : Rat)), ((47 : Rat) / (1 : Rat)), ((48 : Rat) / (1 : Rat)), ((49 : Rat) / (1 : Rat)), ((50 : Rat) / (1 : Rat)), ((51 : Rat) / (1 : Rat)), ((52 : Rat) / (1 : Rat)), ((53 : Rat) / (1 : Rat)), ((54 : Rat) / (1 : Rat)), ((55 : Rat) / (1 : Rat)), ((56 : Rat) / (1 : Rat)), ((57 : Rat) / (1 : Rat)), ((58 : Rat) / (1 : Rat)), ((59 : Rat) / (1 : Rat))])
  result := some ((23 : Rat) / (1 : Rat))
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
      NearStandardAt C.ultrafilter snapshot.expression.denote ((((23 : Rat) / (1 : Rat)) : Rat) : ℝ) :=
  Snapshot.standardPart_sound replay_check rfl

#print axioms Hyperreals.GeneratedReplay.replay_check
#print axioms Hyperreals.GeneratedReplay.trace_consistent
#print axioms Hyperreals.GeneratedReplay.extraction_matches
#print axioms Hyperreals.GeneratedReplay.support_correspondence
#print axioms Hyperreals.GeneratedReplay.replayed_standard_part

end Hyperreals.GeneratedReplay
