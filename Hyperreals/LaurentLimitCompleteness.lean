import Hyperreals.LaurentLimit

/-!
# Completeness of exact Laurent limit extraction

A finite limit along even one nontrivial free filter forces the coefficient
checks made by the executable extractor. Thus Laurent growth cannot masquerade
as a finite limit by selecting an ultrafilter or an infinite subsequence.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.Laurent

noncomputable section

private theorem Poly.all_zero_of_standardPartAt_zero {polynomial : Poly}
    (h : polynomial.standardPartAt? 0 = some 0) :
    polynomial.all (fun value => decide (value = 0)) = true := by
  cases polynomial with
  | nil => rfl
  | cons coefficient tail =>
      simp only [Poly.standardPartAt?] at h
      split at h
      next htail =>
        have hc : coefficient = 0 := Option.some.inj h
        simp [hc, htail]
      next => contradiction

/-- Any finite limit on any nontrivial free filter is rational and is returned
by the exact coefficient algorithm. The real-valued formulation rules out
irrational finite limits as well. -/
theorem Poly.standardPartAt?_complete_real (polynomial : Poly) (shift : Nat)
    (filter : Filter ℕ) [NeBot filter] (hfree : filter ≤ Filter.cofinite)
    {r : ℝ}
    (hlimit : Tendsto (fun n : ℕ => polynomial.eval (n : ℝ) / (n : ℝ) ^ shift)
      filter (𝓝 r)) :
    ∃ q : Rat, polynomial.standardPartAt? shift = some q ∧ (q : ℝ) = r := by
  have hinv : Tendsto (fun n : ℕ => (n : ℝ)⁻¹) filter (𝓝 (0 : ℝ)) := by
    apply (tendsto_inv_atTop_nhds_zero_nat (𝕜 := ℝ)).mono_left
    simpa only [Nat.cofinite_eq_atTop] using hfree
  have hpositive : ∀ᶠ n in filter, 0 < n := by
    apply hfree
    rw [Nat.cofinite_eq_atTop]
    exact eventually_gt_atTop 0
  induction polynomial generalizing shift r with
  | nil =>
      refine ⟨0, rfl, ?_⟩
      simpa [Poly.eval] using hlimit
  | cons coefficient tail ih =>
      cases shift with
      | zero =>
          have htail : Tendsto (fun n : ℕ => Poly.eval tail (n : ℝ) / (n : ℝ) ^ 0)
              filter (𝓝 (0 : ℝ)) := by
            have h := (hlimit.sub (tendsto_const_nhds (x := (coefficient : ℝ)))).mul hinv
            simp only [mul_zero] at h
            refine h.congr' ?_
            filter_upwards [hpositive] with n hn
            have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
            simp only [Poly.eval, pow_zero, div_one]
            field_simp [hn0]
            ring
          obtain ⟨q, hq, hqzero⟩ := ih 0 htail
          have hq0 : q = 0 := by exact_mod_cast hqzero
          subst q
          have hzero := Poly.all_zero_of_standardPartAt_zero hq
          refine ⟨coefficient, by simp [Poly.standardPartAt?, hzero], ?_⟩
          simpa [Poly.eval, Poly.eval_eq_zero_of_all_zero tail hzero] using hlimit
      | succ shift =>
          have hvanishing : Tendsto
              (fun n : ℕ => (coefficient : ℝ) * ((n : ℝ)⁻¹) ^ (shift + 1))
              filter (𝓝 (0 : ℝ)) := by
            simpa using (tendsto_const_nhds (x := (coefficient : ℝ))).mul
              (hinv.pow (shift + 1))
          have htail : Tendsto (fun n : ℕ => Poly.eval tail (n : ℝ) / (n : ℝ) ^ shift)
              filter (𝓝 r) := by
            have h := hlimit.sub hvanishing
            simp only [sub_zero] at h
            refine h.congr' ?_
            filter_upwards [hpositive] with n hn
            have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
            simp only [Poly.eval, inv_pow, pow_succ]
            field_simp
            ring
          exact ih shift htail

/-- A rational candidate is extracted whenever it is a free-filter limit. -/
theorem Form.standardPart?_complete (form : Form) (r : Rat)
    (filter : Filter ℕ) [NeBot filter] (hfree : filter ≤ Filter.cofinite)
    (hlimit : Tendsto (fun n : ℕ => form.eval (n : ℝ)) filter (𝓝 (r : ℝ))) :
    form.standardPart? = some r := by
  obtain ⟨q, hq, heq⟩ := Poly.standardPartAt?_complete_real
    form.num form.shift filter hfree hlimit
  have hqr : q = r := by exact_mod_cast heq
  simpa [Form.standardPart?, hqr] using hq

/-- Rejected forms have no finite real limit along any nontrivial free filter. -/
theorem Form.no_finite_limit_of_standardPart?_none (form : Form)
    (hreject : form.standardPart? = none)
    (filter : Filter ℕ) [NeBot filter] (hfree : filter ≤ Filter.cofinite) (r : ℝ) :
    ¬ Tendsto (fun n : ℕ => form.eval (n : ℝ)) filter (𝓝 r) := by
  intro hlimit
  obtain ⟨q, hq, _⟩ := Poly.standardPartAt?_complete_real
    form.num form.shift filter hfree hlimit
  change form.standardPart? = some q at hq
  rw [hreject] at hq
  contradiction

/-- Every rejected form grows without bound in absolute value. Otherwise a
bounded infinite subsequence would have a free-ultrafilter limit in a compact
interval, contradicting scalar completeness. -/
theorem Form.abs_tendsto_atTop_of_standardPart?_none (form : Form)
    (hreject : form.standardPart? = none) :
    Tendsto (fun n : ℕ => |form.eval (n : ℝ)|) Filter.cofinite atTop := by
  apply tendsto_atTop.mpr
  intro bound
  by_contra hbounded
  let indices : Set ℕ := {n | |form.eval (n : ℝ)| < bound}
  have hinfinite : indices.Infinite := by
    intro hfinite
    apply hbounded
    apply Filter.mem_of_superset hfinite.compl_mem_cofinite
    intro n hn
    change ¬ |form.eval (n : ℝ)| < bound at hn
    exact le_of_not_gt hn
  have hnebot : NeBot (Filter.cofinite ⊓ Filter.principal indices) :=
    hinfinite.cofinite_inf_principal_neBot
  obtain ⟨U, hU⟩ := Filter.exists_ultrafilter_iff.mpr hnebot
  have hfree : (U : Filter ℕ) ≤ Filter.cofinite := hU.trans inf_le_left
  have hindices : indices ∈ U :=
    (hU.trans inf_le_right) (Filter.mem_principal_self indices)
  have hinterval : Set.Icc (-bound) bound ∈
      U.map (fun n : ℕ => form.eval (n : ℝ)) := by
    change {n : ℕ | form.eval (n : ℝ) ∈ Set.Icc (-bound) bound} ∈ U
    apply Filter.mem_of_superset hindices
    intro n hn
    exact ⟨(abs_lt.mp hn).1.le, (abs_lt.mp hn).2.le⟩
  obtain ⟨r, _, hlimit⟩ := isCompact_Icc.ultrafilter_le_nhds' _ hinterval
  exact form.no_finite_limit_of_standardPart?_none hreject (U : Filter ℕ) hfree r hlimit

#print axioms Hyperreals.Laurent.Form.abs_tendsto_atTop_of_standardPart?_none
#print axioms Hyperreals.Laurent.Poly.standardPartAt?_complete_real
#print axioms Hyperreals.Laurent.Form.standardPart?_complete
#print axioms Hyperreals.Laurent.Form.no_finite_limit_of_standardPart?_none

end

end Hyperreals.Laurent
