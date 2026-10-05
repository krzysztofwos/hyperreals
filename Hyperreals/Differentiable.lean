import Hyperreals.DifferentiableCore
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Correctness of the symbolic derivative compiler

Domain checks are propositions about exact reals. They restrict logarithms and
square roots to positive arguments and division to nonzero denominators. The
caller supplies these local domain facts, never a derivative or a limit claim.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

open Filter Topology

namespace Hyperreals.Differentiable

variable {n m : Nat}

def Expr.eval (e : Expr n) (x : Fin n → ℝ) : ℝ :=
  match e with
  | .constant q => q
  | .var i => x i
  | .add l r => l.eval x + r.eval x
  | .sub l r => l.eval x - r.eval x
  | .mul l r => l.eval x * r.eval x
  | .div l r => l.eval x / r.eval x
  | .neg a => -a.eval x
  | .sin a => Real.sin (a.eval x)
  | .cos a => Real.cos (a.eval x)
  | .exp a => Real.exp (a.eval x)
  | .log a => Real.log (a.eval x)
  | .sqrt a => Real.sqrt (a.eval x)

def Expr.Domain (e : Expr n) (x : Fin n → ℝ) : Prop :=
  match e with
  | .constant _ | .var _ => True
  | .add l r | .sub l r | .mul l r => l.Domain x ∧ r.Domain x
  | .div l r => l.Domain x ∧ r.Domain x ∧ r.eval x ≠ 0
  | .neg a | .sin a | .cos a | .exp a => a.Domain x
  | .log a | .sqrt a => a.Domain x ∧ 0 < a.eval x

/-- The chain rule holds for every expression and every differentiable input curve. -/
theorem Expr.hasDerivAt_eval (e : Expr n) (direction : Fin n → Expr n)
    (curve : ℝ → Fin n → ℝ) (t : ℝ)
    (hdomain : e.Domain (curve t))
    (hcurve : ∀ i, HasDerivAt (fun s => curve s i) ((direction i).eval (curve t)) t) :
    HasDerivAt (fun s => e.eval (curve s)) ((e.jvp direction).eval (curve t)) t := by
  induction e with
  | constant q => simpa [Expr.eval, Expr.jvp] using hasDerivAt_const t (q : ℝ)
  | var i => exact hcurve i
  | add l r ihl ihr => exact (ihl hdomain.1).add (ihr hdomain.2)
  | sub l r ihl ihr => exact (ihl hdomain.1).sub (ihr hdomain.2)
  | mul l r ihl ihr => exact (ihl hdomain.1).mul (ihr hdomain.2)
  | div l r ihl ihr =>
    convert! (ihl hdomain.1).div (ihr hdomain.2.1) hdomain.2.2 using 1
    simp [Expr.eval, Expr.jvp, pow_two]
  | neg a ih => exact (ih hdomain).neg
  | sin a ih => exact (ih hdomain).sin
  | cos a ih => exact (ih hdomain).cos
  | exp a ih => exact (ih hdomain).exp
  | log a ih => exact (ih hdomain.1).log (ne_of_gt hdomain.2)
  | sqrt a ih =>
    simpa only [Expr.eval, Expr.jvp, Rat.cast_ofNat] using
      (ih hdomain.1).sqrt (ne_of_gt hdomain.2)

/-- The source is Fréchet differentiable on its specified domain. -/
theorem Expr.differentiableAt (e : Expr n) (x : Fin n → ℝ)
    (hdomain : e.Domain x) : DifferentiableAt ℝ e.eval x := by
  induction e with
  | constant q => exact differentiableAt_const (q : ℝ)
  | var i => exact differentiableAt_apply i x
  | add l r ihl ihr => exact (ihl hdomain.1).add (ihr hdomain.2)
  | sub l r ihl ihr => exact (ihl hdomain.1).sub (ihr hdomain.2)
  | mul l r ihl ihr => exact (ihl hdomain.1).mul (ihr hdomain.2)
  | div l r ihl ihr =>
    convert! (ihl hdomain.1).mul ((ihr hdomain.2.1).inv hdomain.2.2) using 1
  | neg a ih => exact (ih hdomain).neg
  | sin a ih => exact (ih hdomain).sin
  | cos a ih => exact (ih hdomain).cos
  | exp a ih => exact (ih hdomain).exp
  | log a ih => exact (ih hdomain.1).log (ne_of_gt hdomain.2)
  | sqrt a ih => exact (ih hdomain.1).sqrt (ne_of_gt hdomain.2)

/-- The specified domain is open, so sufficiently small perturbations remain valid. -/
theorem Expr.domain_eventually (e : Expr n) (x : Fin n → ℝ)
    (hdomain : e.Domain x) : ∀ᶠ y in 𝓝 x, e.Domain y := by
  induction e with
  | constant q | var q => exact Filter.Eventually.of_forall (fun _ => True.intro)
  | add l r ihl ihr | sub l r ihl ihr | mul l r ihl ihr =>
    exact (ihl hdomain.1).and (ihr hdomain.2)
  | div l r ihl ihr =>
    exact (ihl hdomain.1).and ((ihr hdomain.2.1).and
      ((r.differentiableAt x hdomain.2.1).continuousAt.eventually_ne hdomain.2.2))
  | neg a ih | sin a ih | cos a ih | exp a ih => exact ih hdomain
  | log a ih | sqrt a ih =>
    exact (ih hdomain.1).and
      ((a.differentiableAt x hdomain.1).continuousAt.tendsto.eventually
        (eventually_gt_nhds hdomain.2))

/-- In particular, compiled JVPs differentiate the line through any valid input. -/
theorem Expr.hasDerivAt_line (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : e.Domain x) :
    HasDerivAt (fun t => e.eval (fun i => x i + t * (direction i).eval x))
      ((e.jvp direction).eval x) 0 := by
  have h := e.hasDerivAt_eval direction
    (fun t i => x i + t * (direction i).eval x) 0
    (by simpa using hdomain) (fun i => by
      simpa using ((hasDerivAt_id (0 : ℝ)).mul_const ((direction i).eval x)).const_add (x i))
  simpa using h

/-- Symbolic JVP evaluation is the actual Fréchet derivative applied to the direction. -/
theorem Expr.jvp_eq_fderiv (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : e.Domain x) :
    (e.jvp direction).eval x = fderiv ℝ e.eval x (fun i => (direction i).eval x) := by
  let v : Fin n → ℝ := fun i => (direction i).eval x
  have hline : HasDerivAt (fun t : ℝ => x + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add x
  have h := (e.differentiableAt x hdomain).hasFDerivAt
  have hc := h.comp_hasDerivAt_of_eq 0 hline (by simp)
  apply (e.hasDerivAt_line direction x hdomain).unique
  convert! hc using 1

/-- Compiled syntax is defined wherever both source and tangent syntax are defined. -/
theorem Expr.domain_jvp (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : e.Domain x)
    (hdirection : ∀ i, (direction i).Domain x) : (e.jvp direction).Domain x := by
  induction e with
  | constant q => trivial
  | var i => exact hdirection i
  | add l r ihl ihr | sub l r ihl ihr => exact ⟨ihl hdomain.1, ihr hdomain.2⟩
  | mul l r ihl ihr => exact ⟨⟨ihl hdomain.1, hdomain.2⟩, ⟨hdomain.1, ihr hdomain.2⟩⟩
  | div l r ihl ihr =>
    exact ⟨⟨⟨ihl hdomain.1, hdomain.2.1⟩, ⟨hdomain.1, ihr hdomain.2.1⟩⟩,
      ⟨hdomain.2.1, hdomain.2.1⟩, mul_ne_zero hdomain.2.2 hdomain.2.2⟩
  | neg a ih => exact ih hdomain
  | sin a ih | cos a ih | exp a ih => exact ⟨hdomain, ih hdomain⟩
  | log a ih => exact ⟨ih hdomain.1, hdomain.1, ne_of_gt hdomain.2⟩
  | sqrt a ih =>
    refine ⟨ih hdomain.1, ⟨True.intro, hdomain⟩, ?_⟩
    dsimp [Expr.eval]
    exact mul_ne_zero (by norm_num) (ne_of_gt (Real.sqrt_pos.2 hdomain.2))

def Program.eval (p : Program n m) (x : Fin n → ℝ) : Fin m → ℝ :=
  fun i => (p i).eval x

def Program.Domain (p : Program n m) (x : Fin n → ℝ) : Prop :=
  ∀ i, (p i).Domain x

theorem Program.hasDerivAt_line (p : Program n m) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : p.Domain x) :
    HasDerivAt (fun t => p.eval (fun i => x i + t * (direction i).eval x))
      ((p.jvp direction).eval x) 0 := by
  apply hasDerivAt_pi.2
  intro i
  exact (p i).hasDerivAt_line direction x (hdomain i)

/-- The compiled matrix entries are the Fréchet derivative on coordinate directions. -/
theorem Program.jacobian_eq_fderiv (p : Program n m) (x : Fin n → ℝ)
    (hdomain : p.Domain x) (i : Fin m) (j : Fin n) :
    (p.jacobian i j).eval x = fderiv ℝ (p i).eval x (fun k => if k = j then 1 else 0) := by
  have heval : (fun k => (basis j k).eval x) = (fun k => if k = j then (1 : ℝ) else 0) := by
    funext k
    by_cases hk : k = j <;> simp [basis, Expr.eval, hk]
  simpa only [Program.jacobian, heval] using
    (p i).jvp_eq_fderiv (basis j) x (hdomain i)

#print axioms Expr.domain_eventually
#print axioms Expr.differentiableAt
#print axioms Expr.jvp_eq_fderiv
#print axioms Expr.hasDerivAt_eval
#print axioms Expr.hasDerivAt_line
#print axioms Expr.domain_jvp
#print axioms Program.hasDerivAt_line
#print axioms Program.jacobian_eq_fderiv

end Hyperreals.Differentiable
