import Init.Data.Rat.Basic

/-!
# Exact executable periodic arithmetic

These definitions use only Lean's exact rational runtime. Their refinement to
real sequence semantics and free-completion results are in `Periodic.lean`.
Keeping computation separate from proofs avoids linking Mathlib into the CLI.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Periodic

inductive Expr where
  | constant (value : Rat)
  | alternating
  | add (left right : Expr)
  | sub (left right : Expr)
  | mul (left right : Expr)
  deriving Repr

def Expr.eval : Expr → Bool → Rat
  | .constant value, _ => value
  | .alternating, odd => if odd then -1 else 1
  | .add left right, odd => left.eval odd + right.eval odd
  | .sub left right, odd => left.eval odd - right.eval odd
  | .mul left right, odd => left.eval odd * right.eval odd

structure Support where
  even : Bool
  odd : Bool
  deriving Repr, DecidableEq

def Support.universe : Support := ⟨true, true⟩

def Support.at (support : Support) (odd : Bool) : Bool :=
  if odd then support.odd else support.even

def Support.nonempty (support : Support) : Bool := support.even || support.odd

def Support.inter (left right : Support) : Support :=
  ⟨left.even && right.even, left.odd && right.odd⟩

def Support.compl (support : Support) : Support := ⟨!support.even, !support.odd⟩

inductive Comparison where
  | lt
  | eq
  deriving Repr, DecidableEq

def Comparison.test (comparison : Comparison) (left right : Rat) : Bool :=
  match comparison with
  | .lt => decide (left < right)
  | .eq => decide (left = right)

def Comparison.compile (comparison : Comparison) (left right : Expr) : Support :=
  ⟨comparison.test (left.eval false) (right.eval false),
    comparison.test (left.eval true) (right.eval true)⟩

structure Observation where
  comparison : Comparison
  left : Expr
  right : Expr
  choice : Bool
  deriving Repr

def Observation.mask (observation : Observation) : Support :=
  let comparison := observation.comparison.compile observation.left observation.right
  if observation.choice then comparison else comparison.compl

def restrict (support : Support) (observation : Observation) : Support :=
  support.inter observation.mask

def commit (support : Support) (observation : Observation) : Option Support :=
  let next := restrict support observation
  if next.nonempty then some next else none

/-- Replay a finite observation trace, rejecting the first empty intersection. -/
def run (support : Support) : List Observation → Option Support
  | [] => some support
  | observation :: rest => (commit support observation).bind (fun next => run next rest)

end Hyperreals.Periodic
