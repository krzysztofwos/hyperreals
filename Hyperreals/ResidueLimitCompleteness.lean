import Hyperreals.ResidueLimit
import Hyperreals.LaurentLimitCompleteness

/-!
# Completeness across every compatible completion

Every active residue of the support/expression common period supports a free
ultrafilter. Applying scalar Laurent completeness to these completions proves
that exact extraction finds every completion-independent finite real value.
The common value is necessarily rational.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.Residue

/-- A nonempty family of matching successful branches is accepted. -/
theorem commonValue_complete {values : List (Option Rat)} {r : Rat}
    (hnonempty : values ≠ []) (hall : ∀ value ∈ values, value = some r) :
    commonValue values = some r := by
  cases values with
  | nil => contradiction
  | cons first rest =>
      have hfirst := hall first (by simp)
      subst first
      simp only [commonValue]
      have hrest : rest.all (fun value => decide (value = some r)) = true := by
        apply List.all_eq_true.mpr
        intro value hmem
        exact decide_eq_true (hall value (by simp [hmem]))
      simp [hrest]

/-- Nonempty support remains nonempty after common-period refinement. -/
theorem exists_active_residue {support : Support} {expression : Expr}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true) :
    ∃ residue < Nat.lcm support.length expression.period, support.at residue = true := by
  let period := Nat.lcm support.length expression.period
  have hp : 0 < period := Nat.lcm_pos (Support.length_pos_of_nonempty hnonempty)
    (expression.period_pos hvalid)
  obtain ⟨n, hn⟩ := (Support.carrier_infinite hnonempty).nonempty
  refine ⟨n % period, Nat.mod_lt n hp, ?_⟩
  rw [support.at_mod period n (Nat.dvd_lcm_left _ _)]
  exact hn

/-- The branch coefficient checks completely characterize executable success. -/
theorem standardPart_complete_of_branches {support : Support} {expression : Expr} {r : Rat}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true)
    (hbranches : ∀ residue < Nat.lcm support.length expression.period,
      support.at residue = true →
        (expression.normalizeAt residue).standardPart? = some r) :
    standardPart support expression = some r := by
  simp only [standardPart, hvalid, hnonempty, Bool.and_self, ↓reduceIte]
  apply commonValue_complete
  · obtain ⟨residue, hr, ha⟩ := exists_active_residue hvalid hnonempty
    have hmem : residue ∈ (List.range (Nat.lcm support.length expression.period)).filter
        (fun n => support.at n) := List.mem_filter.mpr ⟨List.mem_range.mpr hr, ha⟩
    intro hnil
    have : (expression.normalizeAt residue).standardPart? ∈
        ((List.range (Nat.lcm support.length expression.period)).filter
          (fun n => support.at n)).map (fun n => (expression.normalizeAt n).standardPart?) :=
      List.mem_map.mpr ⟨residue, hmem, rfl⟩
    simp [hnil] at this
  · intro value hmem
    obtain ⟨residue, hresmem, rfl⟩ := List.mem_map.mp hmem
    obtain ⟨hr, ha⟩ := List.mem_filter.mp hresmem
    exact hbranches residue (List.mem_range.mp hr) ha

noncomputable section

/-- Each residue of a positive period has a free completion concentrated on it. -/
theorem exists_free_ultrafilter_on_residue (period residue : ℕ)
    (hperiod : 0 < period) (hresidue : residue < period) :
    ∃ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite ∧
      {n : ℕ | n % period = residue} ∈ U := by
  let indices : Set ℕ := {n | n % period = residue}
  have hinfinite : indices.Infinite := by
    let index : ℕ → ℕ := fun k => residue + period * k
    have hinj : Function.Injective index := by
      intro a b hab
      exact Nat.eq_of_mul_eq_mul_left hperiod (Nat.add_left_cancel hab)
    apply (Set.infinite_range_of_injective hinj).mono
    rintro n ⟨k, rfl⟩
    simp [indices, index, Nat.mod_eq_of_lt hresidue]
  have hnebot : NeBot (Filter.cofinite ⊓ Filter.principal indices) :=
    hinfinite.cofinite_inf_principal_neBot
  obtain ⟨U, hU⟩ := Filter.exists_ultrafilter_iff.mpr hnebot
  exact ⟨U, hU.trans inf_le_left,
    (hU.trans inf_le_right) (Filter.mem_principal_self indices)⟩

/-- Every active common-period residue has a compatible completion in which
the source expression agrees eventually with that exact Laurent branch. -/
theorem exists_compatible_completion_on_residue {support : Support} {expression : Expr}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true)
    {residue : ℕ} (hresidue : residue < Nat.lcm support.length expression.period)
    (hactive : support.at residue = true) :
    ∃ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite ∧ support.carrier ∈ U ∧
      expression.denote =ᶠ[(U : Filter ℕ)] (fun n => (expression.normalizeAt residue).eval n) := by
  let period := Nat.lcm support.length expression.period
  have hperiod : 0 < period := Nat.lcm_pos (Support.length_pos_of_nonempty hnonempty)
    (expression.period_pos hvalid)
  obtain ⟨U, hfree, hclass⟩ := exists_free_ultrafilter_on_residue period residue hperiod hresidue
  have hsupport : support.carrier ∈ U := by
    apply Filter.mem_of_superset hclass
    intro n hn
    change support.at n = true
    rw [← support.at_mod period n (Nat.dvd_lcm_left _ _), hn]
    exact hactive
  have hpositive : ∀ᶠ n in (U : Filter ℕ), 0 < n := by
    apply hfree
    rw [Nat.cofinite_eq_atTop]
    exact eventually_gt_atTop 0
  refine ⟨U, hfree, hsupport, ?_⟩
  filter_upwards [hpositive, hclass] with n hn hmod
  rw [← hmod]
  exact (expression.normalizeAt_residue_correct n period
    (Nat.dvd_lcm_right _ _) hn hvalid).symm

/-- A common real completion limit forces every active branch to return that
same real value as an exact rational. -/
theorem active_branch_complete_real {support : Support} {expression : Expr} {r : ℝ}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true)
    (hlimit : ∀ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite →
      support.carrier ∈ U → NearStandardAt U expression.denote r)
    {residue : ℕ} (hresidue : residue < Nat.lcm support.length expression.period)
    (hactive : support.at residue = true) :
    ∃ q : Rat, (expression.normalizeAt residue).standardPart? = some q ∧ (q : ℝ) = r := by
  obtain ⟨U, hfree, hsupport, heq⟩ :=
    exists_compatible_completion_on_residue hvalid hnonempty hresidue hactive
  have hbranch := (hlimit U hfree hsupport).congr' heq
  exact Laurent.Poly.standardPartAt?_complete_real
    (expression.normalizeAt residue).num (expression.normalizeAt residue).shift
    (U : Filter ℕ) hfree hbranch

/-- Exact extraction is complete even for common limits initially specified as
arbitrary reals: the computed value is necessarily rational. -/
theorem standardPart_complete_real {support : Support} {expression : Expr} {r : ℝ}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true)
    (hlimit : ∀ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite →
      support.carrier ∈ U → NearStandardAt U expression.denote r) :
    ∃ q : Rat, standardPart support expression = some q ∧ (q : ℝ) = r := by
  obtain ⟨residue, hr, ha⟩ := exists_active_residue hvalid hnonempty
  obtain ⟨q, hq, hqr⟩ := active_branch_complete_real hvalid hnonempty hlimit hr ha
  refine ⟨q, standardPart_complete_of_branches hvalid hnonempty ?_, hqr⟩
  intro other ho hoa
  obtain ⟨p, hp, hpr⟩ := active_branch_complete_real hvalid hnonempty hlimit ho hoa
  have hpq : p = q := by exact_mod_cast (hpr.trans hqr.symm)
  simpa [hpq] using hp

/-- Complete real-valued semantic specification: shared finite completion
limits are exactly the rational results returned by extraction. -/
theorem standardPart_real_iff_all_completions {support : Support} {expression : Expr} {r : ℝ}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true) :
    (∃ q : Rat, standardPart support expression = some q ∧ (q : ℝ) = r) ↔
      ∀ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite →
        support.carrier ∈ U → NearStandardAt U expression.denote r := by
  constructor
  · rintro ⟨q, hq, rfl⟩ U hfree hs
    exact standardPart_sound hq U hfree hs
  · exact standardPart_complete_real hvalid hnonempty

/-- The returned rational exists exactly when all compatible completions agree
on that standard part. Both validity and nonempty support are essential. -/
theorem standardPart_iff_all_completions {support : Support} {expression : Expr} {r : Rat}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true) :
    standardPart support expression = some r ↔
      ∀ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite →
        support.carrier ∈ U → NearStandardAt U expression.denote (r : ℝ) := by
  constructor
  · intro h U hfree hs
    exact standardPart_sound h U hfree hs
  · intro h
    obtain ⟨q, hq, heq⟩ := standardPart_complete_real hvalid hnonempty h
    have hqr : q = r := by exact_mod_cast heq
    simpa [hqr] using hq

/-- Ordinary convergence supplies executable extraction on any valid nonempty
support. This is useful when compiling generic mathematical constructions. -/
theorem standardPart_complete_of_cofinite {support : Support} {expression : Expr} {r : Rat}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true)
    (hlimit : CofiniteLimit expression.denote (r : ℝ)) :
    standardPart support expression = some r := by
  apply (standardPart_iff_all_completions hvalid hnonempty).mpr
  intro U hfree _
  exact hlimit.mono_left hfree

/-- A valid nonempty rejection rules out every completion-independent finite
real standard part, including irrational candidates. -/
theorem standardPart_none_no_common_real {support : Support} {expression : Expr}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true)
    (hreject : standardPart support expression = none) (r : ℝ) :
    ¬ ∀ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite →
      support.carrier ∈ U → NearStandardAt U expression.denote r := by
  intro h
  obtain ⟨q, hq, _⟩ := standardPart_complete_real hvalid hnonempty h
  rw [hreject] at hq
  contradiction

#print axioms Hyperreals.Residue.commonValue_complete
#print axioms Hyperreals.Residue.standardPart_complete_of_branches
#print axioms Hyperreals.Residue.exists_free_ultrafilter_on_residue
#print axioms Hyperreals.Residue.exists_compatible_completion_on_residue
#print axioms Hyperreals.Residue.standardPart_complete_real
#print axioms Hyperreals.Residue.standardPart_real_iff_all_completions
#print axioms Hyperreals.Residue.standardPart_iff_all_completions
#print axioms Hyperreals.Residue.standardPart_complete_of_cofinite
#print axioms Hyperreals.Residue.standardPart_none_no_common_real

end

end Hyperreals.Residue
