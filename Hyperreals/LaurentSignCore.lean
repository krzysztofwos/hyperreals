import Hyperreals.LaurentCore

/-!
# Executable signs and cutoffs for rational Laurent expressions

The recursion ignores trailing zero coefficients. At a nonzero tail, the
next Horner step `head + x * tail` preserves the leading sign and magnitude
once `x ≥ ceil(abs(head) / abs(leading)) + 1`. `LaurentSign.lean` proves this
computed rule against the real denotation. No correctness proofs are input.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Laurent

structure TailSign where
  leading : Rat
  cutoff : Nat
  deriving Repr

def Poly.tailSign : Poly → TailSign
  | [] => ⟨0, 1⟩
  | head :: tail =>
      let result := Poly.tailSign tail
      if result.leading = 0 then
        ⟨head, 1⟩
      else
        ⟨result.leading,
          max result.cutoff ((head.abs / result.leading.abs).ceil.toNat + 1)⟩

def Poly.sign (polynomial : Poly) : Int :=
  let leading := polynomial.tailSign.leading
  if leading < 0 then -1 else if leading = 0 then 0 else 1

def Poly.cutoff (polynomial : Poly) : Nat := polynomial.tailSign.cutoff

def Form.sign (form : Form) : Int := form.num.sign

def Form.cutoff (form : Form) : Nat := form.num.cutoff

end Hyperreals.Laurent
