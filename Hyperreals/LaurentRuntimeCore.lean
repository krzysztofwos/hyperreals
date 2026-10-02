import Hyperreals.PeriodicCore
import Hyperreals.LaurentSignCore
import Hyperreals.LaurentLimitCore

/-!
# Executable periodic Laurent choices

Two residue classes retain exact Laurent forms. A comparison returns both its
tail mask and a concrete cutoff. Invalid monomial division is rejected before
any transition or standard-part request can succeed.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Laurent

abbrev Support := Periodic.Support
abbrev Comparison := Periodic.Comparison

structure CompiledComparison where
  mask : Support
  cutoff : Nat
  deriving Repr

def compareForm (comparison : Comparison) (form : Form) : Bool :=
  match comparison with
  | .lt => decide (form.sign = -1)
  | .eq => decide (form.sign = 0)

def compile (comparison : Comparison) (left right : Expr) : CompiledComparison :=
  let even := (left.normalize false).sub (right.normalize false)
  let odd := (left.normalize true).sub (right.normalize true)
  ⟨⟨compareForm comparison even, compareForm comparison odd⟩,
    max even.cutoff odd.cutoff⟩

structure Observation where
  comparison : Comparison
  left : Expr
  right : Expr
  choice : Bool
  deriving Repr

def Observation.valid (observation : Observation) : Bool :=
  observation.left.valid && observation.right.valid

def Observation.mask (observation : Observation) : Support :=
  let comparison := (compile observation.comparison observation.left observation.right).mask
  if observation.choice then comparison else comparison.compl

def commit (support : Support) (observation : Observation) : Option Support :=
  let next := support.inter observation.mask
  if observation.valid && next.nonempty then some next else none

def run (support : Support) : List Observation → Option Support
  | [] => some support
  | observation :: rest => (commit support observation).bind (fun next => run next rest)

/-- Extraction is relative to all remaining possibilities, without choosing one. -/
def standardPart (support : Support) (expression : Expr) : Option Rat := do
  if !expression.valid then none
  else if support.even then
    let even ← (expression.normalize false).standardPart?
    if support.odd then
      let odd ← (expression.normalize true).standardPart?
      if even = odd then some even else none
    else some even
  else if support.odd then (expression.normalize true).standardPart?
  else none

end Hyperreals.Laurent
