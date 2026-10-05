import Hyperreals.ResidueExprCore
import Hyperreals.ResidueSupportCore
import Hyperreals.LaurentSignCore

/-! Executable comparison compilation at the common period of both operands.
Each residue uses the exact scalar Laurent sign algorithm, and the returned
cutoff is the maximum of all branch cutoffs and one. -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

inductive Comparison where
  | lt
  | eq
  deriving Repr, DecidableEq

structure CompiledComparison where
  mask : Support
  cutoff : Nat
  deriving Repr

def compareForm (comparison : Comparison) (form : Laurent.Form) : Bool :=
  match comparison with
  | .lt => decide (form.sign = -1)
  | .eq => decide (form.sign = 0)

def comparisonForms (left right : Expr) : List Laurent.Form :=
  (List.range (Nat.lcm left.period right.period)).map
    (fun residue => (left.normalizeAt residue).sub (right.normalizeAt residue))

def formsCutoff : List Laurent.Form → Nat
  | [] => 1
  | form :: rest => max form.cutoff (formsCutoff rest)

def compile (comparison : Comparison) (left right : Expr) : CompiledComparison :=
  let forms := comparisonForms left right
  ⟨forms.map (compareForm comparison), formsCutoff forms⟩

end Hyperreals.Residue
