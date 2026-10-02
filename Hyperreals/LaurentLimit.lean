import Hyperreals.LaurentLimitCore
import Hyperreals.LaurentExpr
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Correctness of exact Laurent standard-part extraction

Successful execution of `Form.standardPart?` entails ordinary convergence to
the returned rational. The proof derives convergence from the coefficient
checks, using the vanishing of positive powers of the reciprocal index.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.Laurent

noncomputable section

theorem Poly.eval_eq_zero_of_all_zero (polynomial : Poly)
    (hzero : polynomial.all (fun value => decide (value = 0)) = true) (x : ℝ) :
    polynomial.eval x = 0 := by
  induction polynomial with
  | nil => rfl
  | cons coefficient tail inductionHypothesis =>
      simp only [List.all_cons, Bool.and_eq_true, decide_eq_true_eq] at hzero
      simp [Poly.eval, hzero.1, inductionHypothesis hzero.2]

/-- The executable coefficient check produces an ordinary sequence limit. -/
theorem Poly.standardPartAt?_tendsto (polynomial : Poly) (shift : Nat) (r : Rat)
    (hresult : polynomial.standardPartAt? shift = some r) :
    Tendsto (fun n : ℕ => polynomial.eval (n : ℝ) / (n : ℝ) ^ shift)
      atTop (𝓝 (r : ℝ)) := by
  induction shift generalizing polynomial with
  | zero =>
      cases polynomial with
      | nil =>
          simp only [Poly.standardPartAt?, Option.some.injEq] at hresult
          subst r
          simp [Poly.eval]
      | cons coefficient tail =>
          simp only [Poly.standardPartAt?] at hresult
          split at hresult
          next hzero =>
            simp only [Option.some.injEq] at hresult
            subst r
            simp [Poly.eval, Poly.eval_eq_zero_of_all_zero tail hzero]
          next => contradiction
  | succ shift inductionHypothesis =>
      cases polynomial with
      | nil =>
          simp only [Poly.standardPartAt?, Option.some.injEq] at hresult
          subst r
          simp [Poly.eval]
      | cons coefficient tail =>
          have htail := inductionHypothesis tail hresult
          have hvanishing :
              Tendsto (fun n : ℕ => (coefficient : ℝ) * ((n : ℝ)⁻¹) ^ (shift + 1))
                atTop (𝓝 (0 : ℝ)) := by
            simpa using (tendsto_const_nhds (x := (coefficient : ℝ))).mul
              ((tendsto_inv_atTop_nhds_zero_nat (𝕜 := ℝ)).pow (shift + 1))
          have hsum := hvanishing.add htail
          simp only [zero_add] at hsum
          refine hsum.congr' ?_
          filter_upwards [eventually_ne_atTop (0 : ℕ)] with n hn
          have hnonzero : (n : ℝ) ≠ 0 := by exact_mod_cast hn
          simp only [Poly.eval, inv_pow, pow_succ]
          field_simp

/-- A returned exact rational is the finite limit of the represented form. -/
theorem Form.standardPart?_sound (form : Form) (r : Rat)
    (hresult : form.standardPart? = some r) :
    Tendsto (fun n : ℕ => form.eval (n : ℝ)) atTop (𝓝 (r : ℝ)) :=
  Poly.standardPartAt?_tendsto form.num form.shift r hresult

/-- Equivalent cofinite formulation for use by ultrafilter completions. -/
theorem Form.standardPart?_cofinite (form : Form) (r : Rat)
    (hresult : form.standardPart? = some r) :
    Tendsto (fun n : ℕ => form.eval (n : ℝ)) Filter.cofinite (𝓝 (r : ℝ)) := by
  rw [Nat.cofinite_eq_atTop]
  exact form.standardPart?_sound r hresult

#print axioms Hyperreals.Laurent.Poly.standardPartAt?_tendsto
#print axioms Hyperreals.Laurent.Form.standardPart?_sound
#print axioms Hyperreals.Laurent.Form.standardPart?_cofinite

end

end Hyperreals.Laurent
