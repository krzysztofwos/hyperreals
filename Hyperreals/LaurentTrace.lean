import Hyperreals.LaurentRuntime
import Hyperreals.LaurentStandardPart

/-!
# Exact correspondence between Laurent traces and free completions

The runtime's periodic masks agree with actual observations outside finite
prefixes. Every free ultrafilter therefore contains a mask exactly when it
contains that observation. Replaying a successful trace characterizes all its
compatible completions, not only the existence of one completion. This closes
the bridge from support-relative standard parts to every completion consistent
with the recorded observations.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter

namespace Hyperreals.Laurent

/-- Finite prefix differences cannot change membership in a free ultrafilter. -/
theorem Observation.mask_mem_iff (observation : Observation)
    (hvalid : observation.valid = true) (U : Ultrafilter ℕ)
    (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    observation.mask.carrier ∈ U ↔ observation.denote ∈ U :=
  Filter.eventually_congr (hfree (observation.mask_correct hvalid))

/-- An accepted transition imposes exactly the new observation on completions. -/
theorem commit_mem_iff {support next : Support} {observation : Observation}
    (hcommit : commit support observation = some next)
    (U : Ultrafilter ℕ) (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    next.carrier ∈ U ↔ support.carrier ∈ U ∧ observation.denote ∈ U := by
  have hsound := commit_sound hcommit
  change next.carrier ∈ (U : Filter ℕ) ↔
    support.carrier ∈ (U : Filter ℕ) ∧ observation.denote ∈ (U : Filter ℕ)
  rw [hsound.2.2, Periodic.Support.carrier_inter, Filter.inter_mem_iff]
  exact and_congr Iff.rfl (observation.mask_mem_iff hsound.1 U hfree)

/-- Final support membership is equivalent to initial support membership and
every actual observation in the successful trace. -/
theorem run_mem_iff {support next : Support} {observations : List Observation}
    (hrun : run support observations = some next)
    (U : Ultrafilter ℕ) (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    next.carrier ∈ U ↔
      support.carrier ∈ U ∧ ∀ observation ∈ observations, observation.denote ∈ U := by
  induction observations generalizing support with
  | nil =>
      simp only [run, Option.some.injEq] at hrun
      subst next
      simp
  | cons observation rest inductionHypothesis =>
      simp only [run] at hrun
      cases hcommit : commit support observation with
      | none => simp [hcommit] at hrun
      | some intermediate =>
          simp only [hcommit, Option.bind_some] at hrun
          rw [inductionHypothesis hrun, commit_mem_iff hcommit U hfree]
          simp only [List.forall_mem_cons]
          tauto

/-- Starting from universe, the final support describes precisely all free
ultrafilters consistent with the actual accepted observations. -/
theorem run_universe_mem_iff {next : Support} {observations : List Observation}
    (hrun : run Periodic.Support.universe observations = some next)
    (U : Ultrafilter ℕ) (hfree : (U : Filter ℕ) ≤ Filter.cofinite) :
    next.carrier ∈ U ↔ ∀ observation ∈ observations, observation.denote ∈ U := by
  have huniverse : Periodic.Support.universe.carrier ∈ U := by
    change Periodic.Support.universe.carrier ∈ (U : Filter ℕ)
    simp [Periodic.Support.carrier, Periodic.Support.universe, Periodic.Support.at]
  simpa only [huniverse, true_and] using run_mem_iff hrun U hfree

/-- The runtime's extracted rational is the standard part in every completion
consistent with the actual recorded trace. -/
theorem run_standardPart_sound {next : Support} {observations : List Observation}
    {expression : Expr} {r : Rat}
    (hrun : run Periodic.Support.universe observations = some next)
    (hresult : standardPart next expression = some r) :
    ∀ C : Completion {A | ∃ observation ∈ observations, A = observation.denote},
      NearStandardAt C.ultrafilter expression.denote (r : ℝ) := by
  intro completion
  apply standardPart_sound hresult completion.ultrafilter completion.extendsCofinite
  apply (run_universe_mem_iff hrun completion.ultrafilter completion.extendsCofinite).mpr
  intro observation hmem
  exact completion.contains ⟨observation, hmem, rfl⟩

#print axioms Hyperreals.Laurent.Observation.mask_mem_iff
#print axioms Hyperreals.Laurent.commit_mem_iff
#print axioms Hyperreals.Laurent.run_mem_iff
#print axioms Hyperreals.Laurent.run_universe_mem_iff
#print axioms Hyperreals.Laurent.run_standardPart_sound

end Hyperreals.Laurent
