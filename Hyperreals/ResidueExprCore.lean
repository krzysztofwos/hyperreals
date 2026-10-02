import Hyperreals.LaurentCore
import Init.Data.Nat.Lcm

/-! Exact Laurent expressions with arbitrary finite rational periodic tables.
Every coefficient is retained. Empty tables and zero monomial divisor
coefficients are invalid and are rejected by the runtime. -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

inductive Expr where
  | constant (value : Rat)
  | index
  | reciprocalIndex
  | periodic (values : List Rat)
  | add (left right : Expr)
  | sub (left right : Expr)
  | mul (left right : Expr)
  | divMonomial (argument : Expr) (coefficient : Rat) (power : Int)
  deriving Repr

def Expr.valid : Expr → Bool
  | .constant _ | .index | .reciprocalIndex => true
  | .periodic values => !values.isEmpty
  | .add left right | .sub left right | .mul left right => left.valid && right.valid
  | .divMonomial argument coefficient _ => argument.valid && decide (coefficient ≠ 0)

def Expr.period : Expr → Nat
  | .constant _ | .index | .reciprocalIndex => 1
  | .periodic values => values.length
  | .add left right | .sub left right | .mul left right => Nat.lcm left.period right.period
  | .divMonomial argument _ _ => argument.period

/-- Substitute every periodic coefficient at a residue without approximating
the Laurent expression. Modulo the computed period gives the same form. -/
def Expr.normalizeAt : Expr → Nat → Laurent.Form
  | .constant value, _ => .constant value
  | .index, _ => .index
  | .reciprocalIndex, _ => .reciprocalIndex
  | .periodic values, residue => .constant (values[residue % values.length]?.getD 0)
  | .add left right, residue => (left.normalizeAt residue).add (right.normalizeAt residue)
  | .sub left right, residue => (left.normalizeAt residue).sub (right.normalizeAt residue)
  | .mul left right, residue => (left.normalizeAt residue).mul (right.normalizeAt residue)
  | .divMonomial argument coefficient power, residue =>
      (argument.normalizeAt residue).divMonomial coefficient power

end Hyperreals.Residue
