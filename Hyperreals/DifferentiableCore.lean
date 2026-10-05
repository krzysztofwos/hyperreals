/-!
# Executable differentiable expressions

Finite input and output dimensions are part of the types. The compiler retains
symbolic elementary functions. It neither evaluates floating-point numbers nor
attempts to decide real domain conditions.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Differentiable

inductive Expr (n : Nat) where
  | constant (value : Rat)
  | var (index : Fin n)
  | add (left right : Expr n)
  | sub (left right : Expr n)
  | mul (left right : Expr n)
  | div (left right : Expr n)
  | neg (argument : Expr n)
  | sin (argument : Expr n)
  | cos (argument : Expr n)
  | exp (argument : Expr n)
  | log (argument : Expr n)
  | sqrt (argument : Expr n)
  deriving DecidableEq, Repr

variable {n m : Nat}

/-- Forward symbolic differentiation, with an expression for each input tangent. -/
def Expr.jvp (e : Expr n) (direction : Fin n → Expr n) : Expr n :=
  match e with
  | .constant _ => .constant 0
  | .var i => direction i
  | .add l r => .add (l.jvp direction) (r.jvp direction)
  | .sub l r => .sub (l.jvp direction) (r.jvp direction)
  | .mul l r => .add (.mul (l.jvp direction) r) (.mul l (r.jvp direction))
  | .div l r => .div (.sub (.mul (l.jvp direction) r) (.mul l (r.jvp direction)))
      (.mul r r)
  | .neg a => .neg (a.jvp direction)
  | .sin a => .mul (.cos a) (a.jvp direction)
  | .cos a => .mul (.neg (.sin a)) (a.jvp direction)
  | .exp a => .mul (.exp a) (a.jvp direction)
  | .log a => .div (a.jvp direction) a
  | .sqrt a => .div (a.jvp direction) (.mul (.constant 2) (.sqrt a))

/-- A finite vector of scalar expressions. -/
abbrev Program (n m : Nat) := Fin m → Expr n

def Program.jvp (p : Program n m) (direction : Fin n → Expr n) : Program n m :=
  fun i => (p i).jvp direction

def basis (j : Fin n) : Fin n → Expr n :=
  fun i => .constant (if i = j then 1 else 0)

/-- Rows are outputs and columns are inputs. -/
def Program.jacobian (p : Program n m) : Fin m → Fin n → Expr n :=
  fun i j => (p i).jvp (basis j)

end Hyperreals.Differentiable
