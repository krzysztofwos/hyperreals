import Hyperreals.Semantics
import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Exact expression semantics

This module defines an exact algebraic core of the Python sequence language.
Constants are mathematical real numbers, so connecting Python floating-point
values to this syntax remains an explicit refinement obligation.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals

/-- The exact, total algebraic fragment currently modeled in Lean. -/
inductive Expr where
  | constant (value : ℝ)
  | index
  | reciprocalIndex
  | alternatingSign
  | add (left right : Expr)
  | sub (left right : Expr)
  | mul (left right : Expr)
  | exp (argument : Expr)

namespace Expr

/-- Mathematical denotation of an expression as a real sequence. -/
noncomputable def denote : Expr → Sequence
  | .constant value => Sequence.constant value
  | .index => fun n ↦ n
  | .reciprocalIndex => fun n ↦ (n : ℝ)⁻¹
  | .alternatingSign => fun n ↦ (-1 : ℝ) ^ n
  | .add left right => fun n ↦ left.denote n + right.denote n
  | .sub left right => fun n ↦ left.denote n - right.denote n
  | .mul left right => fun n ↦ left.denote n * right.denote n
  | .exp argument => fun n ↦ Real.exp (argument.denote n)

/-- Exact index set for a strict comparison in the expression language. -/
def ltSet (left right : Expr) : Set ℕ :=
  comparisonLt left.denote right.denote

/-- Exact index set for equality in the expression language. -/
def eqSet (left right : Expr) : Set ℕ :=
  comparisonEq left.denote right.denote

@[simp] theorem denote_constant (value : ℝ) (n : ℕ) :
    (Expr.constant value).denote n = value := rfl

@[simp] theorem denote_index (n : ℕ) : Expr.index.denote n = n := rfl

@[simp] theorem denote_reciprocalIndex (n : ℕ) :
    Expr.reciprocalIndex.denote n = (n : ℝ)⁻¹ := rfl

@[simp] theorem denote_alternatingSign (n : ℕ) :
    Expr.alternatingSign.denote n = (-1 : ℝ) ^ n := rfl

@[simp] theorem denote_add (left right : Expr) (n : ℕ) :
    (left.add right).denote n = left.denote n + right.denote n := rfl

@[simp] theorem denote_sub (left right : Expr) (n : ℕ) :
    (left.sub right).denote n = left.denote n - right.denote n := rfl

@[simp] theorem denote_mul (left right : Expr) (n : ℕ) :
    (left.mul right).denote n = left.denote n * right.denote n := rfl

@[simp] theorem denote_exp (argument : Expr) (n : ℕ) :
    argument.exp.denote n = Real.exp (argument.denote n) := rfl

end Expr

end Hyperreals
