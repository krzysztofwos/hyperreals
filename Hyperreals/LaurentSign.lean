import Hyperreals.LaurentSignCore
import Hyperreals.LaurentExpr
import Mathlib.Data.Rat.Floor
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# Sound computed sign bounds for exact Laurent expressions

The executable Horner recursion supplies both a sign and a concrete natural
cutoff. The proofs establish its agreement with real evaluation on the entire
tail, including zero polynomials represented with arbitrary trailing zeros.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Laurent

private theorem rat_abs_cast (q : Rat) : ((q.abs : Rat) : ℝ) = |(q : ℝ)| := by
  rw [Rat.abs]
  split_ifs with h
  · exact (abs_of_nonneg (Rat.cast_nonneg.mpr h)).symm
  · simp only [Rat.cast_neg]
    exact (abs_of_neg (Rat.cast_lt_zero.mpr (lt_of_not_ge h))).symm

private theorem rat_le_nat_ceil (q : Rat) : (q : ℝ) ≤ (q.ceil.toNat : ℝ) := by
  have hceil : (q : ℝ) ≤ (q.ceil : ℝ) := by
    exact_mod_cast (Rat.le_ceil (x := q))
  have hnat : (q.ceil : ℝ) ≤ (q.ceil.toNat : ℝ) := by
    exact_mod_cast Int.self_le_toNat q.ceil
  exact hceil.trans hnat

/-- Every computed cutoff is positive, so Laurent denominators are positive. -/
theorem Poly.one_le_cutoff (polynomial : Poly) : 1 ≤ polynomial.cutoff := by
  induction polynomial with
  | nil => simp [Poly.cutoff, Poly.tailSign]
  | cons head tail ih =>
      simp only [Poly.cutoff, Poly.tailSign]
      split_ifs with h
      · simp
      · exact ih.trans (Nat.le_max_left _ _)

/-- The zero case is an identity, not a sample-based decision. -/
theorem Poly.eval_eq_zero_of_leading_eq_zero (polynomial : Poly)
    (hzero : polynomial.tailSign.leading = 0) (x : ℝ) : polynomial.eval x = 0 := by
  induction polynomial with
  | nil => simp [Poly.eval]
  | cons head tail ih =>
      by_cases htail : (Poly.tailSign tail).leading = 0
      · have hhead : head = 0 := by simpa [Poly.tailSign, htail] using hzero
        simp [Poly.eval, hhead, ih htail]
      · simp [Poly.tailSign, htail] at hzero

/-- Beyond the computed cutoff, Horner evaluation has at least the magnitude
of its leading coefficient, with the same sign. -/
theorem Poly.leading_bounds (polynomial : Poly) (x : ℝ)
    (hx : (polynomial.cutoff : ℝ) ≤ x) :
    (0 < (polynomial.tailSign.leading : ℝ) →
      (polynomial.tailSign.leading : ℝ) ≤ polynomial.eval x) ∧
    ((polynomial.tailSign.leading : ℝ) < 0 →
      polynomial.eval x ≤ (polynomial.tailSign.leading : ℝ)) := by
  induction polynomial generalizing x with
  | nil => simp [Poly.tailSign]
  | cons head tail ih =>
      by_cases htail : (Poly.tailSign tail).leading = 0
      · have heval := Poly.eval_eq_zero_of_leading_eq_zero tail htail x
        simp [Poly.tailSign, htail, Poly.eval, heval]
      · have htail_cutoff : (Poly.cutoff tail : ℝ) ≤ x := by
          apply le_trans _ hx
          exact_mod_cast (show Poly.cutoff tail ≤ Poly.cutoff (head :: tail) from by
            simp only [Poly.cutoff, Poly.tailSign, htail, ↓reduceIte]
            exact Nat.le_max_left _ _)
        have hstep :
            (((head.abs / (Poly.tailSign tail).leading.abs).ceil.toNat + 1 : ℕ) : ℝ) ≤ x := by
          apply le_trans _ hx
          exact_mod_cast (show
            (head.abs / (Poly.tailSign tail).leading.abs).ceil.toNat + 1 ≤
                Poly.cutoff (head :: tail) from by
              simp only [Poly.cutoff, Poly.tailSign, htail, ↓reduceIte]
              exact Nat.le_max_right _ _)
        have hratio : |(head : ℝ)| / |((Poly.tailSign tail).leading : ℝ)| + 1 ≤ x := by
          have h := rat_le_nat_ceil (head.abs / (Poly.tailSign tail).leading.abs)
          simp only [Rat.cast_div, rat_abs_cast] at h
          push_cast at hstep
          linarith
        have hxnonneg : 0 ≤ x := by
          have hcutoff : (1 : ℝ) ≤ Poly.cutoff tail := by
            exact_mod_cast Poly.one_le_cutoff tail
          linarith
        obtain ⟨hpositive, hnegative⟩ := ih x htail_cutoff
        simp only [Poly.tailSign, htail, ↓reduceIte, Poly.eval]
        constructor
        · intro hlead
          have hbound : |(head : ℝ)| ≤ (x - 1) * (Poly.tailSign tail).leading := by
            apply (div_le_iff₀ hlead).mp
            rw [abs_of_pos hlead] at hratio
            linarith
          have hmul := mul_le_mul_of_nonneg_left (hpositive hlead) hxnonneg
          have habs := neg_abs_le (head : ℝ)
          nlinarith
        · intro hlead
          have hbound : |(head : ℝ)| ≤ (x - 1) * (-(Poly.tailSign tail).leading : ℝ) := by
            apply (div_le_iff₀ (neg_pos.mpr hlead)).mp
            rw [abs_of_neg hlead] at hratio
            linarith
          have hmul := mul_le_mul_of_nonneg_left (hnegative hlead) hxnonneg
          have habs := le_abs_self (head : ℝ)
          nlinarith

/-- The executable polynomial sign classifies real evaluation on its whole
computed tail, for all three outcomes. -/
theorem Poly.sign_correct (polynomial : Poly) (x : ℝ)
    (hx : (polynomial.cutoff : ℝ) ≤ x) :
    (polynomial.sign = -1 ↔ polynomial.eval x < 0) ∧
    (polynomial.sign = 0 ↔ polynomial.eval x = 0) ∧
    (polynomial.sign = 1 ↔ 0 < polynomial.eval x) := by
  have hbounds := polynomial.leading_bounds x hx
  rcases lt_trichotomy polynomial.tailSign.leading 0 with hnegative | hzero | hpositive
  · have hlead : (polynomial.tailSign.leading : ℝ) < 0 := Rat.cast_lt_zero.mpr hnegative
    have heval : polynomial.eval x < 0 := (hbounds.2 hlead).trans_lt hlead
    simp [Poly.sign, hnegative, heval, ne_of_lt heval, not_lt_of_ge heval.le]
  · have heval := polynomial.eval_eq_zero_of_leading_eq_zero hzero x
    simp [Poly.sign, hzero, heval]
  · have hlead : (0 : ℝ) < polynomial.tailSign.leading := Rat.cast_pos.mpr hpositive
    have heval : 0 < polynomial.eval x := hlead.trans_le (hbounds.1 hlead)
    simp [Poly.sign, not_lt_of_ge hpositive.le, ne_of_gt hpositive, heval,
      ne_of_gt heval, not_lt_of_ge heval.le]

theorem Form.one_le_cutoff (form : Form) : 1 ≤ form.cutoff :=
  form.num.one_le_cutoff

/-- Dividing by a positive power of the index preserves the certified sign. -/
theorem Form.sign_correct_real (form : Form) (x : ℝ)
    (hx : (form.cutoff : ℝ) ≤ x) :
    (form.sign = -1 ↔ form.eval x < 0) ∧
    (form.sign = 0 ↔ form.eval x = 0) ∧
    (form.sign = 1 ↔ 0 < form.eval x) := by
  have hcutoff : (1 : ℝ) ≤ form.cutoff := by exact_mod_cast form.one_le_cutoff
  have hxpositive : 0 < x := by linarith
  have hdenominator : 0 < x ^ form.shift := pow_pos hxpositive _
  simpa only [Form.sign, Form.eval, div_lt_iff₀ hdenominator, zero_mul,
    div_pos_iff_of_pos_right hdenominator, div_eq_zero_iff, ne_of_gt hdenominator,
    or_false] using form.num.sign_correct x hx

/-- Runtime bridge: every natural index at or beyond the returned cutoff
agrees with the returned integer sign. -/
theorem Form.sign_correct (form : Form) {n : ℕ} (hn : form.cutoff ≤ n) :
    (form.sign = -1 ↔ form.eval (n : ℝ) < 0) ∧
    (form.sign = 0 ↔ form.eval (n : ℝ) = 0) ∧
    (form.sign = 1 ↔ 0 < form.eval (n : ℝ)) :=
  form.sign_correct_real n (by exact_mod_cast hn)

#print axioms Hyperreals.Laurent.Poly.eval_eq_zero_of_leading_eq_zero
#print axioms Hyperreals.Laurent.Poly.leading_bounds
#print axioms Hyperreals.Laurent.Poly.sign_correct
#print axioms Hyperreals.Laurent.Form.sign_correct

end Hyperreals.Laurent
