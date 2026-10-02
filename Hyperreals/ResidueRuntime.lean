import Hyperreals.ResidueRuntimeCore
import Hyperreals.ResidueComparison
import Hyperreals.ResidueLimit

/-!
# Eventual comparisons and finite traces for arbitrary finite periods

Each accepted step intersects two finite-period supports at their computed LCM.
Its mask agrees with the actual observation outside the proved finite cutoff.
Nonempty final support supplies one free completion of all recorded observations.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter

namespace Hyperreals.Residue

theorem Observation.mask_length_pos (observation : Observation)
    (hvalid : observation.valid = true) : 0 < observation.mask.length := by
  have hv : observation.left.valid = true ∧ observation.right.valid = true := by
    simpa [Observation.valid] using hvalid
  have hlength := compile_mask_pos observation.comparison observation.left observation.right hv.1 hv.2
  cases hchoice : observation.choice <;> simp [Observation.mask, hchoice, hlength]

noncomputable def Observation.denote (observation : Observation) : Set ℕ :=
  let selected := comparisonSet observation.comparison observation.left observation.right
  if observation.choice then selected else selectedᶜ

theorem Observation.mask_correct (observation : Observation)
    (hvalid : observation.valid = true) :
    ∀ᶠ n in Filter.cofinite, n ∈ observation.mask.carrier ↔ n ∈ observation.denote := by
  have hv : observation.left.valid = true ∧ observation.right.valid = true := by
    simpa [Observation.valid] using hvalid
  have hlength := compile_mask_pos observation.comparison observation.left observation.right hv.1 hv.2
  rw [Nat.cofinite_eq_atTop]
  filter_upwards [eventually_ge_atTop
    (compile observation.comparison observation.left observation.right).cutoff] with n hn
  have hc := compile_correct observation.comparison observation.left observation.right
    hv.1 hv.2 n hn
  cases hchoice : observation.choice
  · simp only [Observation.mask, Observation.denote, hchoice, Bool.false_eq_true,
      ↓reduceIte, Support.carrier_compl _ hlength, Set.mem_compl_iff]
    exact not_congr hc
  · simpa only [Observation.mask, Observation.denote, hchoice, ↓reduceIte,
      Support.carrier, Set.mem_ofPred_eq] using hc

/-- Each recorded observation is respected on a cofinite part of the support. -/
def Supports (support : Support) (commitments : Commitments) : Prop :=
  ∀ A ∈ commitments, ∀ᶠ n in Filter.cofinite, n ∈ support.carrier → n ∈ A

theorem supports_empty : Supports Support.universe ∅ := by
  simp [Supports]

theorem commit_sound {support next : Support} {observation : Observation}
    (h : commit support observation = some next) :
    support.nonempty = true ∧ observation.valid = true ∧ next.nonempty = true ∧
      next = support.inter observation.mask := by
  simp only [commit] at h
  split at h
  · cases h
    rename_i hcheck
    have hparts : support.nonempty = true ∧ observation.valid = true ∧
        (support.inter observation.mask).nonempty = true := by
      simpa only [Bool.and_eq_true, and_assoc] using hcheck
    exact ⟨hparts.1, hparts.2.1, hparts.2.2, rfl⟩
  · contradiction

theorem commit_preserves_support {support next : Support} {observation : Observation}
    {commitments : Commitments} (hsupport : Supports support commitments)
    (h : commit support observation = some next) :
    Supports next (insert observation.denote commitments) := by
  have hs := commit_sound h
  have hlength := Support.length_pos_of_nonempty hs.1
  have hmasklength := observation.mask_length_pos hs.2.1
  intro A hA
  rcases hA with rfl | hA
  · filter_upwards [observation.mask_correct hs.2.1] with n hn
    intro hnext
    rw [hs.2.2.2, Support.carrier_inter _ _ hlength hmasklength] at hnext
    exact hn.mp hnext.2
  · filter_upwards [hsupport A hA] with n hn
    intro hnext
    rw [hs.2.2.2, Support.carrier_inter _ _ hlength hmasklength] at hnext
    exact hn hnext.1

theorem extendible_of_support {support : Support} {commitments : Commitments}
    (hne : support.nonempty = true) (hsupport : Supports support commitments) :
    Extendible commitments := by
  have hNeBot : Filter.NeBot (Filter.cofinite ⊓ Filter.principal support.carrier) :=
    (Support.carrier_infinite hne).cofinite_inf_principal_neBot
  rcases Filter.exists_ultrafilter_iff.mpr hNeBot with ⟨U, hU⟩
  have hcof : (U : Filter ℕ) ≤ Filter.cofinite := hU.trans inf_le_left
  have hmem : support.carrier ∈ U :=
    (hU.trans inf_le_right) (Filter.mem_principal_self support.carrier)
  refine ⟨⟨U, hcof, ?_⟩⟩
  intro A hA
  exact Filter.mem_of_superset (Filter.inter_mem hmem (hcof (hsupport A hA)))
    (fun _ hn => hn.2 hn.1)

theorem run_sound {support next : Support} {observations : List Observation}
    {commitments : Commitments} (hne : support.nonempty = true)
    (hsupport : Supports support commitments)
    (h : run support observations = some next) :
    next.nonempty = true ∧ Supports next commitments ∧
      ∀ observation ∈ observations,
        ∀ᶠ n in Filter.cofinite, n ∈ next.carrier → n ∈ observation.denote := by
  induction observations generalizing support commitments with
  | nil =>
      simp only [run, Option.some.injEq] at h
      subst next
      exact ⟨hne, hsupport, by simp⟩
  | cons observation rest ih =>
      simp only [run] at h
      cases hc : commit support observation with
      | none => simp [hc] at h
      | some intermediate =>
          simp only [hc, Option.bind_some] at h
          have htail := ih (commit_sound hc).2.2.1 (commit_preserves_support hsupport hc) h
          refine ⟨htail.1, ?_, ?_⟩
          · intro A hA
            exact htail.2.1 A (Set.mem_insert_of_mem _ hA)
          · intro selected hselected
            rcases List.mem_cons.mp hselected with rfl | hselected
            · exact htail.2.1 selected.denote (Set.mem_insert _ _)
            · exact htail.2.2 selected hselected

/-- One free completion satisfies every actual observation in an accepted run. -/
theorem run_trace_extendible {observations : List Observation} {next : Support}
    (h : run Support.universe observations = some next) :
    Extendible {A | ∃ observation ∈ observations, A = observation.denote} := by
  have hs := run_sound (by rfl : Support.universe.nonempty = true) supports_empty h
  apply extendible_of_support hs.1
  rintro A ⟨observation, hmem, rfl⟩
  exact hs.2.2 observation hmem

#print axioms Hyperreals.Residue.Observation.mask_correct
#print axioms Hyperreals.Residue.commit_sound
#print axioms Hyperreals.Residue.commit_preserves_support
#print axioms Hyperreals.Residue.run_trace_extendible

end Hyperreals.Residue
