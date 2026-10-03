import Hyperreals.ResidueLimitCore

/-! Executable explanations for standard-part extraction. Residues are measured
modulo the support/expression common period. Invalid inputs have a separate result. -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

inductive LimitDiagnostic where
  | finite (value : Rat)
  | divergent (residue : Nat)
  | disagreement (leftResidue : Nat) (leftValue : Rat) (rightResidue : Nat) (rightValue : Rat)
  | invalidInput
  deriving Repr, DecidableEq

def activeLimits (support : Support) (expression : Expr) : List (Nat × Option Rat) :=
  ((List.range (Nat.lcm support.length expression.period)).filter
    (fun residue => support.at residue)).map
    (fun residue => (residue, (expression.normalizeAt residue).standardPart?))

def diagnoseLimits (values : List (Nat × Option Rat)) : LimitDiagnostic :=
  match values.find? (fun entry => entry.2.isNone) with
  | some (residue, _) => .divergent residue
  | none =>
    match values with
    | (residue, some value) :: rest =>
      match rest.find? (fun entry => decide (entry.2 ≠ some value)) with
      | some (other, some otherValue) => .disagreement residue value other otherValue
      | some (_, none) => .invalidInput
      | none => .finite value
    | _ => .invalidInput

def diagnoseStandardPart (support : Support) (expression : Expr) : LimitDiagnostic :=
  if expression.valid && support.nonempty then diagnoseLimits (activeLimits support expression)
  else .invalidInput

end Hyperreals.Residue
