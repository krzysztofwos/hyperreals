import Hyperreals.ResidueLimitCore
import Hyperreals.ResidueSupport
import Hyperreals.ResidueExpr
import Hyperreals.LaurentLimit
import Hyperreals.StandardPart

/-!
# Correctness of arbitrary-period standard-part extraction

The exact common-period enumeration checks every active residue. Scalar Laurent
limit correctness supplies its branch limits. Finiteness of the residue family
allows their eventual neighborhood conditions to hold simultaneously. Selecting
the actual residue then proves convergence along every free filter containing
the remaining support, without assuming any input convergence certificate.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.Residue

/-- A common result can only arise from a nonempty list of matching successes. -/
theorem commonValue_success {values : List (Option Rat)} {r : Rat}
    (hresult : commonValue values = some r) :
    values ≠ [] ∧ ∀ value ∈ values, value = some r := by
  cases values with
  | nil => simp [commonValue] at hresult
  | cons first rest =>
      cases first with
      | none => simp [commonValue] at hresult
      | some candidate =>
          simp only [commonValue] at hresult
          split at hresult
          next hsame =>
            cases hresult
            refine ⟨by simp, ?_⟩
            intro value hmem
            rcases List.mem_cons.mp hmem with rfl | hmem
            · rfl
            · exact of_decide_eq_true ((List.all_eq_true.mp hsame) value hmem)
          next => contradiction

/-- Success checks validity, support nonemptiness, and the same Laurent limit
at every active residue of the computed common period. -/
theorem standardPart_success {support : Support} {expression : Expr} {r : Rat}
    (hresult : standardPart support expression = some r) :
    expression.valid = true ∧ support.nonempty = true ∧
      ∀ residue < Nat.lcm support.length expression.period,
        support.at residue = true →
          (expression.normalizeAt residue).standardPart? = some r := by
  simp only [standardPart] at hresult
  split at hresult
  next hcheck =>
    have hparts : expression.valid = true ∧ support.nonempty = true := by
      simpa only [Bool.and_eq_true] using hcheck
    refine ⟨hparts.1, hparts.2, ?_⟩
    intro residue hresidue hactive
    apply (commonValue_success hresult).2
    apply List.mem_map.mpr
    refine ⟨residue, ?_, rfl⟩
    exact List.mem_filter.mpr ⟨List.mem_range.mpr hresidue, hactive⟩
  next => contradiction

noncomputable section

/-- Every free filter containing the current support receives the exact limit
returned by the executable common-period extraction. -/
theorem standardPart_tendsto {support : Support} {expression : Expr} {r : Rat}
    (hresult : standardPart support expression = some r)
    (filter : Filter ℕ) (hfree : filter ≤ Filter.cofinite)
    (hsupport : support.carrier ∈ filter) :
    Tendsto expression.denote filter (𝓝 (r : ℝ)) := by
  have hsuccess := standardPart_success hresult
  let period := Nat.lcm support.length expression.period
  have hperiod : 0 < period :=
    Nat.lcm_pos (Support.length_pos_of_nonempty hsuccess.2.1)
      (expression.period_pos hsuccess.1)
  let branch := fun (residue : Fin period) (n : ℕ) =>
    if support.at residue then (expression.normalizeAt residue).eval n else (r : ℝ)
  have hbranch (residue : Fin period) :
      Tendsto (branch residue) filter (𝓝 (r : ℝ)) := by
    by_cases hactive : support.at residue = true
    · have hscalar := (expression.normalizeAt residue).standardPart?_cofinite r
        (hsuccess.2.2 residue residue.isLt hactive)
      simpa [branch, hactive] using hscalar.mono_left hfree
    · simp only [branch, hactive]
      exact tendsto_const_nhds
  have hpositive : ∀ᶠ n in filter, 0 < n := by
    apply hfree
    rw [Nat.cofinite_eq_atTop]
    exact eventually_gt_atTop 0
  intro target htarget
  change ∀ᶠ n in filter, expression.denote n ∈ target
  have hbranches : ∀ᶠ n in filter, ∀ residue : Fin period, branch residue n ∈ target :=
    Filter.eventually_all.mpr (fun residue => (hbranch residue).eventually htarget)
  filter_upwards [hbranches, hpositive, hsupport] with n hall hn hs
  change support.at n = true at hs
  have hactive : support.at (n % period) = true := by
    rw [support.at_mod period n (Nat.dvd_lcm_left _ _)]
    exact hs
  have hselected := hall ⟨n % period, Nat.mod_lt n hperiod⟩
  simp only [branch, hactive, ite_true] at hselected
  rw [expression.normalizeAt_residue_correct n period
    (Nat.dvd_lcm_right _ _) hn hsuccess.1] at hselected
  exact hselected

/-- Successful extraction yields the same standard part in every free
ultrafilter that contains the remaining support. -/
theorem standardPart_sound {support : Support} {expression : Expr} {r : Rat}
    (hresult : standardPart support expression = some r)
    (U : Ultrafilter ℕ) (hfree : (U : Filter ℕ) ≤ Filter.cofinite)
    (hsupport : support.carrier ∈ U) :
    NearStandardAt U expression.denote (r : ℝ) :=
  standardPart_tendsto hresult (U : Filter ℕ) hfree hsupport

/-- Extraction from full support entails ordinary cofinite convergence. -/
theorem standardPart_universe_cofinite {expression : Expr} {r : Rat}
    (hresult : standardPart Support.universe expression = some r) :
    CofiniteLimit expression.denote (r : ℝ) := by
  apply standardPart_tendsto hresult Filter.cofinite le_rfl
  simp

#print axioms Hyperreals.Residue.commonValue_success
#print axioms Hyperreals.Residue.standardPart_success
#print axioms Hyperreals.Residue.standardPart_tendsto
#print axioms Hyperreals.Residue.standardPart_sound
#print axioms Hyperreals.Residue.standardPart_universe_cofinite

end

end Hyperreals.Residue
