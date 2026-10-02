import Hyperreals.ResidueComparisonCore
import Hyperreals.ResidueLimitCore

/-! Executable observations and finite traces with dynamic LCM supports. -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

structure Observation where
  comparison : Comparison
  left : Expr
  right : Expr
  choice : Bool
  deriving Repr

def Observation.valid (observation : Observation) : Bool :=
  observation.left.valid && observation.right.valid

def Observation.mask (observation : Observation) : Support :=
  let mask := (compile observation.comparison observation.left observation.right).mask
  if observation.choice then mask else mask.compl

def commit (support : Support) (observation : Observation) : Option Support :=
  let next := support.inter observation.mask
  if support.nonempty && observation.valid && next.nonempty then some next else none

def run (support : Support) : List Observation → Option Support
  | [] => some support
  | observation :: rest => (commit support observation).bind (fun next => run next rest)

end Hyperreals.Residue
