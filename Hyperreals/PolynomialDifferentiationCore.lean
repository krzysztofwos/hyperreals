import Hyperreals.ResidueExprCore

/-!
# Executable polynomial infinitesimal syntax

Polynomials use ascending dense rational coefficients. Neither compiler accepts
an expansion, derivative certificate, or limit certificate. The literal quotient
uses the existing monomial division constructor. The divided difference instead
computes a polynomial identity, valid even when the increment is zero.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.PolynomialDifferentiation

/-- Horner evaluation as source syntax, retaining every coefficient. -/
def polynomialExpr : List Rat → Residue.Expr → Residue.Expr
  | [], _ => .constant 0
  | coefficient :: rest, argument =>
      .add (.constant coefficient) (.mul argument (polynomialExpr rest argument))

/-- Exact rational evaluation, used only to state the computed result. -/
def value : List Rat → Rat → Rat
  | [], _ => 0
  | coefficient :: rest, a => coefficient + a * value rest a

/-- Exact derivative value computed from coefficients, never supplied as input. -/
def derivativeValue : List Rat → Rat → Rat
  | [], _ => 0
  | _ :: rest, a => value rest a + a * derivativeValue rest a

/-- The polynomial divided difference in an arbitrary representable increment.
It satisfies `h * D = P(a+h) - P(a)` without a nonzero premise. -/
def dividedDifference : List Rat → Rat → Residue.Expr → Residue.Expr
  | [], _, _ => .constant 0
  | _ :: rest, a, increment =>
      .add (polynomialExpr rest (.add (.constant a) increment))
        (.mul (.constant a) (dividedDifference rest a increment))

/-- `c / n^k`, including the constant case `k = 0`. -/
def incrementExpr (c : Rat) (k : Nat) : Residue.Expr :=
  .divMonomial (.constant c) 1 (.ofNat k)

/-- The unexpanded difference quotient. Its denominator is `c / n^(k+1)`.
Using `k+1` makes positivity of the exponent structural. -/
def quotientExpr (coefficients : List Rat) (a c : Rat) (k : Nat) : Residue.Expr :=
  .divMonomial
    (.sub (polynomialExpr coefficients (.add (.constant a) (incrementExpr c (k + 1))))
      (polynomialExpr coefficients (.constant a)))
    c (.negSucc k)

end Hyperreals.PolynomialDifferentiation
