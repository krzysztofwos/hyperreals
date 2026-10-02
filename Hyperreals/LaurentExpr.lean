import Hyperreals.LaurentCore
import Hyperreals.Semantics
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Algebra.Ring.Commute
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Refinement of exact Laurent normalization to real sequences

The normalization rules preserve pointwise real denotation at every positive
index. No polynomial coefficients are discarded, including coefficients whose
powers can later be shifted into the constant term by division.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Laurent

noncomputable def Poly.eval : Poly → ℝ → ℝ
  | [], _ => 0
  | coefficient :: rest, x => (coefficient : ℝ) + x * Poly.eval rest x

noncomputable def Form.eval (form : Form) (x : ℝ) : ℝ :=
  form.num.eval x / x ^ form.shift

noncomputable def monomial (coefficient : Rat) (power : Int) (x : ℝ) : ℝ :=
  match power with
  | .ofNat k => (coefficient : ℝ) * x ^ k
  | .negSucc k => (coefficient : ℝ) / x ^ (k + 1)

noncomputable def Expr.eval : Expr → Bool → ℝ → ℝ
  | .constant value, _, _ => value
  | .index, _, x => x
  | .reciprocalIndex, _, x => x⁻¹
  | .alternating, odd, _ => if odd then -1 else 1
  | .add left right, odd, x => left.eval odd x + right.eval odd x
  | .sub left right, odd, x => left.eval odd x - right.eval odd x
  | .mul left right, odd, x => left.eval odd x * right.eval odd x
  | .divMonomial argument coefficient power, odd, x =>
      argument.eval odd x / monomial coefficient power x

noncomputable def Expr.denote : Expr → Sequence
  | .constant value => fun _ => value
  | .index => fun n => n
  | .reciprocalIndex => fun n => (n : ℝ)⁻¹
  | .alternating => fun n => (-1 : ℝ) ^ n
  | .add left right => fun n => left.denote n + right.denote n
  | .sub left right => fun n => left.denote n - right.denote n
  | .mul left right => fun n => left.denote n * right.denote n
  | .divMonomial argument coefficient power =>
      fun n => argument.denote n / monomial coefficient power n

@[simp] theorem Poly.eval_add (left right : Poly) (x : ℝ) :
    (left.add right).eval x = left.eval x + right.eval x := by
  induction left generalizing right with
  | nil => simp [Poly.add, Poly.eval]
  | cons a rest ih =>
      cases right with
      | nil => simp [Poly.add, Poly.eval]
      | cons b tail => simp [Poly.add, Poly.eval, ih]; ring

@[simp] theorem Poly.eval_scale (coefficient : Rat) (polynomial : Poly) (x : ℝ) :
    (polynomial.scale coefficient).eval x = (coefficient : ℝ) * polynomial.eval x := by
  induction polynomial with
  | nil => simp [Poly.scale, Poly.eval]
  | cons a rest ih => simp [Poly.scale, Poly.eval, ih]; ring

@[simp] theorem Poly.eval_neg (polynomial : Poly) (x : ℝ) :
    polynomial.neg.eval x = -polynomial.eval x := by
  simp [Poly.neg]

@[simp] theorem Poly.eval_shift (k : ℕ) (polynomial : Poly) (x : ℝ) :
    (polynomial.shift k).eval x = x ^ k * polynomial.eval x := by
  induction k with
  | zero => simp [Poly.shift]
  | succ k ih => simp [Poly.shift, Poly.eval, ih, pow_succ]; ring

@[simp] theorem Poly.eval_mul (left right : Poly) (x : ℝ) :
    (left.mul right).eval x = left.eval x * right.eval x := by
  induction left with
  | nil => simp [Poly.mul, Poly.eval]
  | cons a rest ih => simp [Poly.mul, Poly.eval, ih]; ring

@[simp] theorem Form.eval_constant (value : Rat) (x : ℝ) :
    (Form.constant value).eval x = value := by
  simp [Form.constant, Form.eval, Poly.eval]

@[simp] theorem Form.eval_index (x : ℝ) : Form.index.eval x = x := by
  simp [Form.index, Form.eval, Poly.eval]

@[simp] theorem Form.eval_reciprocalIndex (x : ℝ) :
    Form.reciprocalIndex.eval x = x⁻¹ := by
  simp [Form.reciprocalIndex, Form.eval, Poly.eval, one_div]

theorem Form.eval_add (left right : Form) (x : ℝ) (hx : x ≠ 0) :
    (left.add right).eval x = left.eval x + right.eval x := by
  simp only [Form.add, Form.eval, Poly.eval_add, Poly.eval_shift, pow_add]
  field_simp

@[simp] theorem Form.eval_neg (form : Form) (x : ℝ) :
    form.neg.eval x = -form.eval x := by
  simp [Form.neg, Form.eval, neg_div]

theorem Form.eval_sub (left right : Form) (x : ℝ) (hx : x ≠ 0) :
    (left.sub right).eval x = left.eval x - right.eval x := by
  simp [Form.sub, Form.eval_add _ _ _ hx, sub_eq_add_neg]

@[simp] theorem Form.eval_mul (left right : Form) (x : ℝ) :
    (left.mul right).eval x = left.eval x * right.eval x := by
  simp [Form.mul, Form.eval, pow_add, div_mul_div_comm]

theorem Form.eval_divMonomial (form : Form) (coefficient : Rat) (power : Int)
    (x : ℝ) (hx : x ≠ 0) (hc : coefficient ≠ 0) :
    (form.divMonomial coefficient power).eval x = form.eval x / monomial coefficient power x := by
  have hcReal : (coefficient : ℝ) ≠ 0 := by exact_mod_cast hc
  cases power with
  | ofNat k =>
      simp [Form.divMonomial, Form.eval, monomial, Poly.eval_scale, pow_add]
      field_simp
  | negSucc k =>
      simp [Form.divMonomial, Form.eval, monomial, Poly.eval_shift, Poly.eval_scale]
      field_simp

theorem Expr.normalize_correct (expression : Expr) (odd : Bool) (x : ℝ)
    (hx : 0 < x) (hvalid : expression.valid = true) :
    (expression.normalize odd).eval x = expression.eval odd x := by
  induction expression with
  | constant value => simp [Expr.normalize, Expr.eval]
  | index => simp [Expr.normalize, Expr.eval]
  | reciprocalIndex => simp [Expr.normalize, Expr.eval]
  | alternating => cases odd <;> simp [Expr.normalize, Expr.eval]
  | add left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      simpa [Expr.normalize, Expr.eval, Form.eval_add _ _ _ hx.ne'] using
        congrArg₂ (· + ·) (ihl hparts.1) (ihr hparts.2)
  | sub left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      simpa [Expr.normalize, Expr.eval, Form.eval_sub _ _ _ hx.ne'] using
        congrArg₂ (· - ·) (ihl hparts.1) (ihr hparts.2)
  | mul left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      simpa [Expr.normalize, Expr.eval] using
        congrArg₂ (· * ·) (ihl hparts.1) (ihr hparts.2)
  | divMonomial argument coefficient power ih =>
      have hparts : argument.valid = true ∧ decide (coefficient ≠ 0) = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      have hc : coefficient ≠ 0 := of_decide_eq_true hparts.2
      simpa [Expr.normalize, Expr.eval, Form.eval_divMonomial _ _ _ _ hx.ne' hc] using
        congrArg (· / monomial coefficient power x) (ih hparts.1)

theorem Expr.eval_parity_correct (expression : Expr) (n : ℕ) :
    expression.eval (decide (n % 2 = 1)) n = expression.denote n := by
  induction expression with
  | constant _ | index | reciprocalIndex => rfl
  | alternating =>
      simp only [Expr.eval, Expr.denote]
      rw [neg_one_pow_eq_pow_mod_two]
      by_cases h : n % 2 = 1
      · simp [h]
      · have hz : n % 2 = 0 := by omega
        simp [hz]
  | add _ _ ihl ihr => simp [Expr.eval, Expr.denote, ihl, ihr]
  | sub _ _ ihl ihr => simp [Expr.eval, Expr.denote, ihl, ihr]
  | mul _ _ ihl ihr => simp [Expr.eval, Expr.denote, ihl, ihr]
  | divMonomial _ _ _ ih => simp [Expr.eval, Expr.denote, ih]

theorem Expr.normalize_sequence_correct (expression : Expr) (n : ℕ)
    (hn : 0 < n) (hvalid : expression.valid = true) :
    (expression.normalize (decide (n % 2 = 1))).eval n = expression.denote n := by
  rw [expression.normalize_correct _ _ (by exact_mod_cast hn) hvalid,
    expression.eval_parity_correct]

#print axioms Hyperreals.Laurent.Poly.eval_mul
#print axioms Hyperreals.Laurent.Expr.normalize_correct
#print axioms Hyperreals.Laurent.Expr.normalize_sequence_correct

end Hyperreals.Laurent
