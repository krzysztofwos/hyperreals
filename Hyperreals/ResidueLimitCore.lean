import Hyperreals.ResidueSupportCore
import Hyperreals.ResidueExprCore
import Hyperreals.LaurentLimitCore

/-!
# Exact standard-part extraction over arbitrary finite periods

Extraction enumerates a common period of the current support and expression.
Every active residue must have a finite Laurent limit, and all returned exact
rationals must agree. Empty support and invalid expressions are rejected before
extraction. No observation or completion choice is made by this operation.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

/-- Accept a nonempty list of successful, identical rational results. -/
def commonValue : List (Option Rat) → Option Rat
  | [] => none
  | none :: _ => none
  | some value :: rest =>
      if rest.all (fun result => decide (result = some value)) then some value else none

/-- Inspect every active residue of the support/expression common period. -/
def standardPart (support : Support) (expression : Expr) : Option Rat :=
  if expression.valid && support.nonempty then
    commonValue (((List.range (Nat.lcm support.length expression.period)).filter
      (fun residue => support.at residue)).map
      (fun residue => (expression.normalizeAt residue).standardPart?))
  else none

end Hyperreals.Residue
