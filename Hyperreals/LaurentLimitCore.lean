import Hyperreals.LaurentCore

/-!
# Executable extraction of finite Laurent limits

Coefficients are exact rationals. A numerator coefficient above the denominator
shift would produce a positive power of the index, so successful extraction
requires every such coefficient to be zero. No convergence premise is supplied
to this algorithm. Its correctness is proved in `LaurentLimit`.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Laurent

/-- Read the coefficient at `shift`, rejecting nonzero higher coefficients. -/
def Poly.standardPartAt? : Poly → Nat → Option Rat
  | [], _ => some 0
  | coefficient :: tail, 0 =>
      if tail.all (fun value => decide (value = 0)) then some coefficient else none
  | _ :: tail, shift + 1 => Poly.standardPartAt? tail shift

/-- Exact finite-limit candidate for a normalized Laurent expression. -/
def Form.standardPart? (form : Form) : Option Rat :=
  form.num.standardPartAt? form.shift

end Hyperreals.Laurent
