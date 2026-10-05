import Init.Data.Rat.Basic

/-!
# Exact executable Laurent normalization

Dense ascending rational coefficients retain every term. A `Form` represents a
polynomial divided by a natural power of the index. The residue expression
normalizer uses these scalar operations on each coefficient residue. Their
real-valued arithmetic laws are proved in `LaurentExpr.lean`.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Laurent

abbrev Poly := List Rat

def Poly.add : Poly → Poly → Poly
  | [], right => right
  | left, [] => left
  | a :: left, b :: right => (a + b) :: Poly.add left right

def Poly.scale (coefficient : Rat) : Poly → Poly
  | [] => []
  | a :: rest => coefficient * a :: Poly.scale coefficient rest

def Poly.neg (polynomial : Poly) : Poly := polynomial.scale (-1)

def Poly.shift : Nat → Poly → Poly
  | 0, polynomial => polynomial
  | k + 1, polynomial => 0 :: Poly.shift k polynomial

def Poly.mul : Poly → Poly → Poly
  | [], _ => []
  | a :: rest, right => Poly.add (Poly.scale a right) (0 :: Poly.mul rest right)

structure Form where
  num : Poly
  shift : Nat
  deriving Repr, DecidableEq

def Form.constant (value : Rat) : Form := ⟨[value], 0⟩
def Form.index : Form := ⟨[0, 1], 0⟩
def Form.reciprocalIndex : Form := ⟨[1], 1⟩

def Form.add (left right : Form) : Form :=
  ⟨(left.num.shift right.shift).add (right.num.shift left.shift), left.shift + right.shift⟩

def Form.neg (form : Form) : Form := ⟨form.num.neg, form.shift⟩
def Form.sub (left right : Form) : Form := left.add right.neg
def Form.mul (left right : Form) : Form := ⟨left.num.mul right.num, left.shift + right.shift⟩

def Form.divMonomial (form : Form) (coefficient : Rat) : Int → Form
  | .ofNat power => ⟨form.num.scale coefficient⁻¹, form.shift + power⟩
  | .negSucc power => ⟨(form.num.scale coefficient⁻¹).shift (power + 1), form.shift⟩

end Hyperreals.Laurent
