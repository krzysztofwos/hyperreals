import Hyperreals.StandardPart

/-!
# Limit side conditions and exact Laurent terms

A nonzero denominator limit permits division of convergent sequences. It does
not imply that the quotient stays away from zero: that conclusion requires a
nonzero quotient limit. The reciprocal-index counterexample records the missing
hypothesis in a rule that would infer a positive lower bound from denominator
convergence alone.

The final theorem records why truncating a positive power before Laurent
division can change the constant coefficient. This explains why vanishing terms
must be retained when later operations can divide by powers of an infinitesimal.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals

noncomputable section

/-- A positive eventual lower bound on the absolute value of a sequence. -/
def EventuallyAbsLowerBound (x : Sequence) : Prop :=
  ∃ bound : ℝ, 0 < bound ∧ ∀ᶠ n in Filter.cofinite, bound ≤ |x n|

/-- For a convergent sequence, staying away from zero is equivalent to a
nonzero limit. The forward direction rules out spurious lower-bound facts. -/
theorem eventuallyAbsLowerBound_iff_limit_ne_zero {x : Sequence} {a : ℝ}
    (hx : CofiniteLimit x a) : EventuallyAbsLowerBound x ↔ a ≠ 0 := by
  constructor
  · rintro ⟨bound, hbound, heventual⟩ rfl
    have habs : Tendsto (fun n ↦ |x n|) Filter.cofinite (𝓝 0) := by
      simpa using hx.abs
    have hsmall := habs.eventually_lt_const hbound
    obtain ⟨n, hlower, hupper⟩ := (heventual.and hsmall).exists
    exact (not_lt_of_ge hlower) hupper
  · intro ha
    have hpositive : 0 < |a| := abs_pos.mpr ha
    refine ⟨|a| / 2, half_pos hpositive, ?_⟩
    exact hx.abs.eventually_const_le (half_lt_self hpositive)

/-- The correct division rule: the denominator limit must be nonzero to
compute the quotient limit. The quotient limit itself must be nonzero to
deduce a positive eventual lower bound. -/
theorem quotient_limit_and_lower_bound_iff {x y : Sequence} {a b : ℝ}
    (hx : CofiniteLimit x a) (hy : CofiniteLimit y b) (hb : b ≠ 0) :
    CofiniteLimit (fun n ↦ x n / y n) (a / b) ∧
      (EventuallyAbsLowerBound (fun n ↦ x n / y n) ↔ a / b ≠ 0) := by
  have hquotient := hx.div hy hb
  exact ⟨hquotient, eventuallyAbsLowerBound_iff_limit_ne_zero hquotient⟩

/-- Concrete counterexample to assigning a positive absolute lower bound to
every quotient whose denominator has a nonzero limit. -/
theorem reciprocal_div_one_no_positive_lower_bound :
    CofiniteLimit (fun n ↦ reciprocalIndex n / 1) 0 ∧
      ¬ EventuallyAbsLowerBound (fun n ↦ reciprocalIndex n / 1) := by
  have hlimit : CofiniteLimit (fun n ↦ reciprocalIndex n / 1) 0 := by
    simpa using reciprocalIndex_cofiniteLimit
  refine ⟨hlimit, ?_⟩
  rw [eventuallyAbsLowerBound_iff_limit_ne_zero hlimit]
  simp

/-- An omitted `c * δ^k` term has limit zero when `k > 0`, but division by
`δ^k` gives limit `c`. A series truncation before that division cannot use
vanishing of the omitted term as evidence that the constant term is preserved. -/
theorem discarded_reciprocal_power_changes_shifted_limit (k : ℕ) (hk : 0 < k)
    (c : ℝ) :
    CofiniteLimit (fun n ↦ c * reciprocalIndex n ^ k) 0 ∧
      CofiniteLimit (fun n ↦ (c * reciprocalIndex n ^ k) / reciprocalIndex n ^ k) c := by
  constructor
  · have hpower := reciprocalIndex_cofiniteLimit.pow k
    simpa [Sequence.constant, zero_pow (Nat.ne_of_gt hk)] using
      (CofiniteLimit.constant c).mul hpower
  · have hnonzero : ∀ᶠ n : ℕ in Filter.cofinite, n ≠ 0 := by
      rw [Nat.cofinite_eq_atTop]
      exact eventually_ne_atTop 0
    have heq : (fun n ↦ (c * reciprocalIndex n ^ k) / reciprocalIndex n ^ k)
        =ᶠ[Filter.cofinite] Sequence.constant c := by
      filter_upwards [hnonzero] with n hn
      have hdelta : reciprocalIndex n ≠ 0 := by
        simp [reciprocalIndex, hn]
      simp [Sequence.constant, pow_ne_zero k hdelta]
    exact (CofiniteLimit.constant c).congr' heq.symm

#print axioms Hyperreals.eventuallyAbsLowerBound_iff_limit_ne_zero
#print axioms Hyperreals.quotient_limit_and_lower_bound_iff
#print axioms Hyperreals.reciprocal_div_one_no_positive_lower_bound
#print axioms Hyperreals.discarded_reciprocal_power_changes_shifted_limit

end

end Hyperreals
