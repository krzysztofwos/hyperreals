import Hyperreals.PolynomialDifferentiationCore
import Hyperreals.ResidueExpr
import Hyperreals.ResidueLimitCompleteness
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Algebra.Polynomial.Inductions
import Mathlib.Algebra.Polynomial.Eval.Defs
import Mathlib.Tactic.LinearCombination

/-!
# Generic executable infinitesimal polynomial differentiation

The compiler accepts arbitrary finite rational coefficient lists. Exact
finite-index identities connect generated syntax to literal difference
quotients. Ordinary differentiation and cofinite convergence are proved from
the coefficients, rather than accepted as certificates from the caller.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.PolynomialDifferentiation

open Residue

private theorem denote_constant (q : Rat) :
    (Expr.constant q).denote = Sequence.constant (q : ℝ) := rfl

private theorem denote_add (left right : Expr) :
    (Expr.add left right).denote = (fun n => left.denote n + right.denote n) := rfl

private theorem denote_mul (left right : Expr) :
    (Expr.mul left right).denote = (fun n => left.denote n * right.denote n) := rfl

@[simp] theorem polynomialExpr_valid (coefficients : List Rat) (argument : Expr)
    (hvalid : argument.valid = true) : (polynomialExpr coefficients argument).valid = true := by
  induction coefficients with
  | nil => rfl
  | cons coefficient rest ih => simp [polynomialExpr, Expr.valid, hvalid, ih]

@[simp] theorem polynomialExpr_eval (coefficients : List Rat) (argument : Expr)
    (residue : Nat) (x : ℝ) :
    (polynomialExpr coefficients argument).eval residue x =
      Laurent.Poly.eval coefficients (argument.eval residue x) := by
  induction coefficients with
  | nil => simp [polynomialExpr, Expr.eval, Laurent.Poly.eval]
  | cons coefficient rest ih => simp [polynomialExpr, Expr.eval, Laurent.Poly.eval, ih]

@[simp] theorem value_cast (coefficients : List Rat) (a : Rat) :
    (value coefficients a : ℝ) = Laurent.Poly.eval coefficients a := by
  induction coefficients with
  | nil => simp [value, Laurent.Poly.eval]
  | cons coefficient rest ih => simp [value, Laurent.Poly.eval, ih]

/-- Dense coefficient lists cover every rational polynomial, including zero.
The executable interface therefore restricts coefficients, not polynomial degree. -/
theorem rational_polynomial_representable (p : Polynomial Rat) :
    ∃ coefficients : List Rat, ∀ x : ℝ,
      Laurent.Poly.eval coefficients x = p.eval₂ (Rat.castHom ℝ) x := by
  induction p using Polynomial.induction_on' with
  | add p q hp hq =>
      obtain ⟨left, hl⟩ := hp
      obtain ⟨right, hr⟩ := hq
      exact ⟨Laurent.Poly.add left right, fun x => by simp [hl, hr]⟩
  | monomial n a =>
      exact ⟨Laurent.Poly.shift n [a], fun x => by
        simp [Laurent.Poly.eval, Polynomial.eval₂_monomial, mul_comm]⟩

/-- Agreement with the ordinary real derivative for every coefficient list. -/
theorem polynomial_hasDerivAt (coefficients : List Rat) (a : Rat) :
    HasDerivAt (Laurent.Poly.eval coefficients) (derivativeValue coefficients a : ℝ) a := by
  induction coefficients with
  | nil => simpa [Laurent.Poly.eval, derivativeValue] using hasDerivAt_const (a : ℝ) (0 : ℝ)
  | cons coefficient rest ih =>
      have h := (hasDerivAt_const (a : ℝ) (coefficient : ℝ)).add
        ((hasDerivAt_id (a : ℝ)).mul ih)
      simpa only [Laurent.Poly.eval, derivativeValue, Rat.cast_add, Rat.cast_mul,
        value_cast, zero_add, one_mul, id_eq] using! h

@[simp] theorem dividedDifference_valid (coefficients : List Rat) (a : Rat) (increment : Expr)
    (hvalid : increment.valid = true) :
    (dividedDifference coefficients a increment).valid = true := by
  induction coefficients with
  | nil => rfl
  | cons coefficient rest ih =>
      have hp := polynomialExpr_valid rest (.add (.constant a) increment)
        (by simp [Expr.valid, hvalid])
      simp only [dividedDifference, Expr.valid, hp, ih, Bool.and_self]

/-- Exact polynomial identity. Zero increments require no cancellation. -/
theorem dividedDifference_identity (coefficients : List Rat) (a : Rat) (increment : Expr)
    (n : Nat) :
    increment.denote n * (dividedDifference coefficients a increment).denote n =
      Laurent.Poly.eval coefficients ((a : ℝ) + increment.denote n) -
        Laurent.Poly.eval coefficients a := by
  induction coefficients with
  | nil => simp [dividedDifference, Expr.denote, Expr.eval, Laurent.Poly.eval]
  | cons coefficient rest ih =>
      simp only [dividedDifference, Expr.denote, Expr.eval, polynomialExpr_eval,
        Laurent.Poly.eval] at *
      linear_combination (a : ℝ) * ih

/-- The compiled polynomial is the genuine quotient wherever division is legal. -/
theorem dividedDifference_eq_quotient (coefficients : List Rat) (a : Rat) (increment : Expr)
    (n : Nat) (hnonzero : increment.denote n ≠ 0) :
    (dividedDifference coefficients a increment).denote n =
      (Laurent.Poly.eval coefficients ((a : ℝ) + increment.denote n) -
        Laurent.Poly.eval coefficients a) / increment.denote n := by
  apply (eq_div_iff hnonzero).2
  simpa [mul_comm] using dividedDifference_identity coefficients a increment n

/-- Polynomial compilation preserves limits along any filter. -/
theorem polynomialExpr_tendsto (coefficients : List Rat) (argument : Expr) (a : ℝ)
    (F : Filter Nat) (hlimit : Tendsto argument.denote F (𝓝 a)) :
    Tendsto (polynomialExpr coefficients argument).denote F
      (𝓝 (Laurent.Poly.eval coefficients a)) := by
  induction coefficients with
  | nil => simpa [polynomialExpr, Laurent.Poly.eval, denote_constant, Sequence.constant]
      using! (tendsto_const_nhds : Tendsto (fun _ : Nat => (0 : ℝ)) F (𝓝 0))
  | cons coefficient rest ih =>
      exact tendsto_const_nhds.add (hlimit.mul ih)

/-- Evaluating polynomial syntax preserves any cofinite input limit. -/
theorem polynomialExpr_cofiniteLimit (coefficients : List Rat) (argument : Expr) (a : ℝ)
    (hlimit : CofiniteLimit argument.denote a) :
    CofiniteLimit (polynomialExpr coefficients argument).denote
      (Laurent.Poly.eval coefficients a) :=
  polynomialExpr_tendsto coefficients argument a Filter.cofinite hlimit

/-- The divided-difference limit holds along any filter, so an observation may
establish infinitesimality only on the currently retained completions. -/
theorem dividedDifference_tendsto (coefficients : List Rat) (a : Rat) (increment : Expr)
    (F : Filter Nat) (hlimit : Tendsto increment.denote F (𝓝 0)) :
    Tendsto (dividedDifference coefficients a increment).denote F
      (𝓝 (derivativeValue coefficients a : ℝ)) := by
  induction coefficients with
  | nil => simpa [dividedDifference, derivativeValue, denote_constant, Sequence.constant]
      using! (tendsto_const_nhds : Tendsto (fun _ : Nat => (0 : ℝ)) F (𝓝 0))
  | cons coefficient rest ih =>
      have ha : Tendsto (.add (.constant a) increment : Expr).denote F (𝓝 (a : ℝ)) := by
        simpa only [denote_add, denote_constant, add_zero, Sequence.constant] using!
          (tendsto_const_nhds : Tendsto (Sequence.constant (a : ℝ)) F (𝓝 (a : ℝ))).add hlimit
      have h := (polynomialExpr_tendsto rest _ a F ha).add
        ((tendsto_const_nhds : Tendsto (Sequence.constant (a : ℝ)) F (𝓝 (a : ℝ))).mul ih)
      simpa only [dividedDifference, derivativeValue, Rat.cast_add, Rat.cast_mul,
        value_cast, denote_add, denote_mul, denote_constant, Sequence.constant] using! h

/-- Divided differences converge to the derivative for any representable
infinitesimal increment. Nonzeroness is needed only to identify the quotient. -/
theorem dividedDifference_cofiniteLimit (coefficients : List Rat) (a : Rat) (increment : Expr)
    (hlimit : CofiniteLimit increment.denote 0) :
    CofiniteLimit (dividedDifference coefficients a increment).denote
      (derivativeValue coefficients a : ℝ) :=
  dividedDifference_tendsto coefficients a increment Filter.cofinite hlimit

@[simp] theorem incrementExpr_valid (c : Rat) (k : Nat) :
    (incrementExpr c k).valid = true := by simp [incrementExpr, Expr.valid]

@[simp] theorem incrementExpr_denote (c : Rat) (k n : Nat) :
    (incrementExpr c k).denote n = (c : ℝ) / (n : ℝ) ^ k := by
  simp [incrementExpr, Expr.denote, Expr.eval, Laurent.monomial]

@[simp] theorem quotientExpr_valid (coefficients : List Rat) (a c : Rat) (k : Nat)
    (hc : c ≠ 0) : (quotientExpr coefficients a c k).valid = true := by
  have hp := polynomialExpr_valid coefficients (.add (.constant a) (incrementExpr c (k + 1)))
    (by simp [Expr.valid])
  have hq := polynomialExpr_valid coefficients (.constant a) rfl
  simp only [quotientExpr, Expr.valid, hp, hq, Bool.true_and, decide_eq_true hc]

/-- Denotation of the unexpanded syntax is the literal difference quotient. -/
theorem quotientExpr_denote (coefficients : List Rat) (a c : Rat) (k n : Nat) :
    (quotientExpr coefficients a c k).denote n =
      (Laurent.Poly.eval coefficients ((a : ℝ) + (c : ℝ) / (n : ℝ) ^ (k + 1)) -
        Laurent.Poly.eval coefficients a) / ((c : ℝ) / (n : ℝ) ^ (k + 1)) := by
  simp only [quotientExpr, Expr.denote, Expr.eval, polynomialExpr_eval,
    incrementExpr, Laurent.monomial, Rat.cast_one, one_mul]

private theorem eventually_index_pos : ∀ᶠ n : Nat in Filter.cofinite, 0 < n := by
  rw [Nat.cofinite_eq_atTop]
  exact eventually_gt_atTop 0

set_option maxHeartbeats 600000 in
theorem incrementExpr_cofiniteLimit (c : Rat) (k : Nat) :
    CofiniteLimit (incrementExpr c (k + 1)).denote 0 := by
  have h := (CofiniteLimit.constant (c : ℝ)).mul (reciprocalIndex_cofiniteLimit.pow (k + 1))
  have heq : (incrementExpr c (k + 1)).denote =
      (fun n : Nat => (c : ℝ) * reciprocalIndex n ^ (k + 1)) := by
    funext n
    rw [incrementExpr_denote]
    simp [reciprocalIndex, div_eq_mul_inv, inv_pow]
  rw [heq]
  simpa [Sequence.constant] using h

/-- Every nonzero rational scale and every positive integer power gives the
ordinary derivative, for the literal quotient syntax before normalization. -/
theorem quotientExpr_cofiniteLimit (coefficients : List Rat) (a c : Rat) (k : Nat)
    (hc : c ≠ 0) :
    CofiniteLimit (quotientExpr coefficients a c k).denote
      (derivativeValue coefficients a : ℝ) := by
  have h := dividedDifference_cofiniteLimit coefficients a (incrementExpr c (k + 1))
    (incrementExpr_cofiniteLimit c k)
  apply h.congr'
  filter_upwards [eventually_index_pos] with n hn
  have hcn : (c : ℝ) ≠ 0 := by exact_mod_cast hc
  have hnonzero : (incrementExpr c (k + 1)).denote n ≠ 0 := by
    rw [incrementExpr_denote]
    exact div_ne_zero hcn (pow_ne_zero _ (by exact_mod_cast hn.ne'))
  rw [dividedDifference_eq_quotient _ _ _ _ hnonzero, quotientExpr_denote,
    incrementExpr_denote]

/-- Completeness closes the executable loop: normalization and extraction find
this value on every nonempty support, without requiring the value as a certificate. -/
theorem dividedDifference_standardPart (coefficients : List Rat) (a : Rat)
    (increment : Expr) (support : Support) (hvalid : increment.valid = true)
    (hnonempty : support.nonempty = true) (hlimit : CofiniteLimit increment.denote 0) :
    standardPart support (dividedDifference coefficients a increment) =
      some (derivativeValue coefficients a) :=
  standardPart_complete_of_cofinite (dividedDifference_valid _ _ _ hvalid) hnonempty
    (dividedDifference_cofiniteLimit _ _ _ hlimit)

/-- A checked zero standard part on the current support suffices. Global
cofinite convergence is unnecessary, so observations can establish the premise. -/
theorem dividedDifference_standardPart_of_standardPart (coefficients : List Rat) (a : Rat)
    (increment : Expr) (support : Support)
    (hzero : standardPart support increment = some 0) :
    standardPart support (dividedDifference coefficients a increment) =
      some (derivativeValue coefficients a) := by
  have hsuccess := standardPart_success hzero
  apply (standardPart_iff_all_completions
    (dividedDifference_valid _ _ _ hsuccess.1) hsuccess.2.1).mpr
  intro U hfree hsupport
  apply dividedDifference_tendsto coefficients a increment (U : Filter Nat)
  simpa [NearStandardAt] using! standardPart_sound hzero U hfree hsupport

/-- Generic executable differentiation: every polynomial, rational point,
nonzero rational scale and positive integer power is handled by the existing
extractor on any nonempty residue support. The source is the literal quotient. -/
theorem quotientExpr_standardPart (coefficients : List Rat) (a c : Rat) (k : Nat)
    (support : Support) (hc : c ≠ 0) (hnonempty : support.nonempty = true) :
    standardPart support (quotientExpr coefficients a c k) =
      some (derivativeValue coefficients a) :=
  standardPart_complete_of_cofinite (quotientExpr_valid _ _ _ _ hc) hnonempty
    (quotientExpr_cofiniteLimit _ _ _ _ hc)

/-- The returned exact rational is the usual real derivative, not only a limit
assigned to a specially constructed expression. -/
theorem quotientExpr_computes_derivative (coefficients : List Rat) (a c : Rat) (k : Nat)
    (support : Support) (hc : c ≠ 0) (hnonempty : support.nonempty = true) :
    standardPart support (quotientExpr coefficients a c k) =
        some (derivativeValue coefficients a) ∧
      deriv (Laurent.Poly.eval coefficients) (a : ℝ) = (derivativeValue coefficients a : ℝ) :=
  ⟨quotientExpr_standardPart _ _ _ _ _ hc hnonempty,
    (polynomial_hasDerivAt coefficients a).deriv⟩

#print axioms Hyperreals.PolynomialDifferentiation.rational_polynomial_representable
#print axioms Hyperreals.PolynomialDifferentiation.polynomial_hasDerivAt
#print axioms Hyperreals.PolynomialDifferentiation.dividedDifference_identity
#print axioms Hyperreals.PolynomialDifferentiation.dividedDifference_cofiniteLimit
#print axioms Hyperreals.PolynomialDifferentiation.quotientExpr_denote
#print axioms Hyperreals.PolynomialDifferentiation.quotientExpr_cofiniteLimit
#print axioms Hyperreals.PolynomialDifferentiation.dividedDifference_standardPart
#print axioms Hyperreals.PolynomialDifferentiation.dividedDifference_standardPart_of_standardPart
#print axioms Hyperreals.PolynomialDifferentiation.quotientExpr_standardPart
#print axioms Hyperreals.PolynomialDifferentiation.quotientExpr_computes_derivative

end Hyperreals.PolynomialDifferentiation
