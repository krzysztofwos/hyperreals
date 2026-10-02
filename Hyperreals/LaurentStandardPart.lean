import Hyperreals.LaurentRuntimeCore
import Hyperreals.LaurentLimit
import Hyperreals.Periodic
import Hyperreals.StandardPart

/-!
# Standard parts extracted by the periodic Laurent runtime

Successful extraction checks every active parity and returns a rational only
when their finite limits agree. The resulting limit holds in every free
ultrafilter containing the current support. No convergence certificate is an
input: the proof derives the needed limits from the executable coefficient
checks and the normalization refinement theorem.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.Laurent

/-- A successful runtime extraction checked validity, nonempty support, and
the same finite limit on every active parity. -/
theorem standardPart_success {support : Support} {expression : Expr} {r : Rat}
    (hresult : standardPart support expression = some r) :
    expression.valid = true ∧ support.nonempty = true ∧
      (∀ odd : Bool, support.at odd = true →
        (expression.normalize odd).standardPart? = some r) := by
  cases hvalid : expression.valid with
  | false => simp [standardPart, hvalid] at hresult
  | true =>
      refine ⟨rfl, ?_⟩
      cases support with
      | mk even odd =>
          cases even <;> cases odd <;>
            cases heven : (expression.normalize false).standardPart? <;>
            cases hodd : (expression.normalize true).standardPart? <;>
            simp_all [standardPart, Periodic.Support.nonempty, Periodic.Support.at]

noncomputable section

/-- The result remains a limit along any filter extending the cofinite filter
and containing the support. The filter need not already be an ultrafilter. -/
theorem standardPart_tendsto {support : Support} {expression : Expr} {r : Rat}
    (hresult : standardPart support expression = some r)
    (filter : Filter ℕ) (hfree : filter ≤ Filter.cofinite)
    (hsupport : support.carrier ∈ filter) :
    Tendsto expression.denote filter (𝓝 (r : ℝ)) := by
  have hsuccess := standardPart_success hresult
  let branch := fun (odd : Bool) (n : ℕ) =>
    if support.at odd then (expression.normalize odd).eval n else (r : ℝ)
  have hbranch (odd : Bool) : Tendsto (branch odd) Filter.cofinite (𝓝 (r : ℝ)) := by
    by_cases hactive : support.at odd = true
    · simpa [branch, hactive] using
        (expression.normalize odd).standardPart?_cofinite r (hsuccess.2.2 odd hactive)
    · simp [branch, hactive]
  have hcombined :
      Tendsto (fun n => if Periodic.parity n then branch true n else branch false n)
        Filter.cofinite (𝓝 (r : ℝ)) :=
    (hbranch true).if' (hbranch false)
  have hpositive : ∀ᶠ n in filter, 0 < n := by
    apply hfree
    rw [Nat.cofinite_eq_atTop]
    exact eventually_gt_atTop 0
  have hactive : ∀ᶠ n in filter, support.at (Periodic.parity n) = true := hsupport
  refine (hcombined.mono_left hfree).congr' ?_
  filter_upwards [hpositive, hactive] with n hn hparity
  have hnormalize := expression.normalize_sequence_correct n hn hsuccess.1
  change (expression.normalize (Periodic.parity n)).eval n = expression.denote n at hnormalize
  cases hp : Periodic.parity n <;>
    simp only [hp] at hparity hnormalize <;>
    simpa [hp, branch, hparity] using hnormalize

/-- Actual successful extraction supplies the standard part in every free
ultrafilter completion that contains the remaining support. -/
theorem standardPart_sound {support : Support} {expression : Expr} {r : Rat}
    (hresult : standardPart support expression = some r)
    (U : Ultrafilter ℕ) (hfree : (U : Filter ℕ) ≤ Filter.cofinite)
    (hsupport : support.carrier ∈ U) :
    NearStandardAt U expression.denote (r : ℝ) :=
  standardPart_tendsto hresult (U : Filter ℕ) hfree hsupport

/-- With both parities still active, the recognized limit is ordinary cofinite
convergence and is therefore independent of any later completion choices. -/
theorem standardPart_universe_cofinite {expression : Expr} {r : Rat}
    (hresult : standardPart Periodic.Support.universe expression = some r) :
    CofiniteLimit expression.denote (r : ℝ) := by
  apply standardPart_tendsto hresult Filter.cofinite le_rfl
  simp [Periodic.Support.carrier, Periodic.Support.universe, Periodic.Support.at]

#print axioms Hyperreals.Laurent.standardPart_success
#print axioms Hyperreals.Laurent.standardPart_tendsto
#print axioms Hyperreals.Laurent.standardPart_sound
#print axioms Hyperreals.Laurent.standardPart_universe_cofinite

end

end Hyperreals.Laurent
