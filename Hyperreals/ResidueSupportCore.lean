import Init.Data.List.Range
import Init.Data.Nat.Lcm

/-!
# Executable supports for arbitrary finite periods

The list length is the period, and position `r` records membership of residue
`r`. Intersection enumerates a common least-common-multiple period. Empty or
all-false supports fail the executable nonemptiness check.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

abbrev Support := List Bool

def Support.at (support : Support) (n : Nat) : Bool :=
  support.getD (n % support.length) false

def Support.universe : Support := [true]

def Support.nonempty (support : Support) : Bool := support.any id

def Support.lift (support : Support) (period : Nat) : Support :=
  (List.range period).map (fun residue => support.at residue)

def Support.inter (left right : Support) : Support :=
  (List.range (Nat.lcm left.length right.length)).map
    (fun residue => left.at residue && right.at residue)

def Support.compl (support : Support) : Support := support.map (! ·)

end Hyperreals.Residue
