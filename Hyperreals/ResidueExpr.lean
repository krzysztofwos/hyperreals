import Hyperreals.ResidueExprCore
import Hyperreals.LaurentExpr
import Mathlib.Data.Nat.GCD.Basic

/-! Refinement of finite-table expression normalization to exact real sequence
semantics. Common-period reduction is proved from divisibility. It does not
assume that an input table or expression already has a certified denotation. -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

noncomputable def Expr.eval : Expr → Nat → ℝ → ℝ
  | .constant value, _, _ => value
  | .index, _, x => x
  | .reciprocalIndex, _, x => x⁻¹
  | .periodic values, residue, _ => (values[residue % values.length]?.getD 0 : Rat)
  | .add left right, residue, x => left.eval residue x + right.eval residue x
  | .sub left right, residue, x => left.eval residue x - right.eval residue x
  | .mul left right, residue, x => left.eval residue x * right.eval residue x
  | .divMonomial argument coefficient power, residue, x =>
      argument.eval residue x / Laurent.monomial coefficient power x

/-- Source semantics: each table is sampled at the index modulo its own length. -/
noncomputable def Expr.denote (expression : Expr) : Sequence :=
  fun n => expression.eval n n

theorem Expr.period_pos (expression : Expr) (hvalid : expression.valid = true) :
    0 < expression.period := by
  induction expression with
  | constant _ | index | reciprocalIndex => simp [Expr.period]
  | periodic values => cases values <;> simp_all [Expr.valid, Expr.period]
  | add left right ihl ihr | sub left right ihl ihr | mul left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      exact Nat.lcm_pos (ihl hparts.1) (ihr hparts.2)
  | divMonomial argument coefficient power ih =>
      have hparts : argument.valid = true ∧ decide (coefficient ≠ 0) = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      exact ih hparts.1

/-- A common-period reduction leaves the exact normalized form unchanged. -/
theorem Expr.normalizeAt_mod (expression : Expr) (n period : ℕ)
    (hperiod : expression.period ∣ period) :
    expression.normalizeAt (n % period) = expression.normalizeAt n := by
  induction expression with
  | constant _ | index | reciprocalIndex => rfl
  | periodic values =>
      change values.length ∣ period at hperiod
      simp only [Expr.normalizeAt]
      rw [Nat.mod_mod_of_dvd n hperiod]
  | add left right ihl ihr | sub left right ihl ihr | mul left right ihl ihr =>
      have hl : left.period ∣ period := (Nat.dvd_lcm_left _ _).trans hperiod
      have hr : right.period ∣ period := (Nat.dvd_lcm_right _ _).trans hperiod
      simp only [Expr.normalizeAt, ihl hl, ihr hr]
  | divMonomial argument coefficient power ih =>
      simp only [Expr.normalizeAt, ih hperiod]

/-- Every valid expression normalizes without changing its real evaluation. -/
theorem Expr.normalizeAt_correct (expression : Expr) (residue : ℕ) (x : ℝ)
    (hx : 0 < x) (hvalid : expression.valid = true) :
    (expression.normalizeAt residue).eval x = expression.eval residue x := by
  induction expression with
  | constant value => simp [Expr.normalizeAt, Expr.eval]
  | index => simp [Expr.normalizeAt, Expr.eval]
  | reciprocalIndex => simp [Expr.normalizeAt, Expr.eval]
  | periodic values => simp [Expr.normalizeAt, Expr.eval]
  | add left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      simpa [Expr.normalizeAt, Expr.eval, Laurent.Form.eval_add _ _ _ hx.ne'] using
        congrArg₂ (· + ·) (ihl hparts.1) (ihr hparts.2)
  | sub left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      simpa [Expr.normalizeAt, Expr.eval, Laurent.Form.eval_sub _ _ _ hx.ne'] using
        congrArg₂ (· - ·) (ihl hparts.1) (ihr hparts.2)
  | mul left right ihl ihr =>
      have hparts : left.valid = true ∧ right.valid = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      simpa [Expr.normalizeAt, Expr.eval] using
        congrArg₂ (· * ·) (ihl hparts.1) (ihr hparts.2)
  | divMonomial argument coefficient power ih =>
      have hparts : argument.valid = true ∧ decide (coefficient ≠ 0) = true := by
        simpa only [Expr.valid, Bool.and_eq_true] using hvalid
      have hc : coefficient ≠ 0 := of_decide_eq_true hparts.2
      simpa [Expr.normalizeAt, Expr.eval, Laurent.Form.eval_divMonomial _ _ _ _ hx.ne' hc]
        using congrArg (· / Laurent.monomial coefficient power x) (ih hparts.1)

theorem Expr.normalize_sequence_correct (expression : Expr) (n : ℕ)
    (hn : 0 < n) (hvalid : expression.valid = true) :
    (expression.normalizeAt n).eval n = expression.denote n :=
  expression.normalizeAt_correct n n (by exact_mod_cast hn) hvalid

/-- Refinement after reducing an index modulo any common multiple of the
expression period, as used by both comparison and standard-part compilation. -/
theorem Expr.normalizeAt_residue_correct (expression : Expr) (n period : ℕ)
    (hperiod : expression.period ∣ period) (hn : 0 < n)
    (hvalid : expression.valid = true) :
    (expression.normalizeAt (n % period)).eval n = expression.denote n := by
  rw [expression.normalizeAt_mod n period hperiod]
  exact expression.normalize_sequence_correct n hn hvalid

#print axioms Hyperreals.Residue.Expr.period_pos
#print axioms Hyperreals.Residue.Expr.normalizeAt_mod
#print axioms Hyperreals.Residue.Expr.normalizeAt_correct
#print axioms Hyperreals.Residue.Expr.normalizeAt_residue_correct

end Hyperreals.Residue
