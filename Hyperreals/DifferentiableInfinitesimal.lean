import Hyperreals.Differentiable
import Hyperreals.StandardPart
import Mathlib.Analysis.Calculus.Deriv.Slope

/-!
# Literal infinitesimal quotients for the differentiable language

A step can depend on the completion. Its required properties are convergence to
zero and eventual nonzeroness in that same completion. No nilpotent arithmetic,
finite-difference approximation, or derivative certificate is assumed.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

open Filter Topology

namespace Hyperreals.Differentiable

variable {n m : Nat}

def Expr.quotient (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (h : Sequence) : Sequence :=
  fun k => (e.eval (fun i => x i + h k * (direction i).eval x) - e.eval x) / h k

/-- The compiled derivative is the limit of the literal quotient in any filter. -/
theorem Expr.quotient_tendsto (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : e.Domain x) (h : Sequence) (L : Filter ℕ)
    (hzero : Tendsto h L (𝓝 0)) (hnonzero : ∀ᶠ k in L, h k ≠ 0) :
    Tendsto (e.quotient direction x h) L (𝓝 ((e.jvp direction).eval x)) := by
  have hpunctured : Tendsto h L (𝓝[≠] 0) :=
    tendsto_nhdsWithin_iff.mpr ⟨hzero, hnonzero⟩
  have hlimit := (e.hasDerivAt_line direction x hdomain).tendsto_slope_zero.comp hpunctured
  convert! hlimit using 1
  funext k
  simp [Expr.quotient, smul_eq_mul, div_eq_mul_inv, mul_comm]

/-- Source operations stay inside their real domains along an infinitesimal perturbation. -/
theorem Expr.quotient_eventually_domain (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : e.Domain x) (h : Sequence) (L : Filter ℕ)
    (hzero : Tendsto h L (𝓝 0)) :
    ∀ᶠ k in L, e.Domain (fun i => x i + h k * (direction i).eval x) := by
  have hline : Tendsto (fun k i => x i + h k * (direction i).eval x) L (𝓝 x) := by
    apply tendsto_pi_nhds.2
    intro i
    simpa using tendsto_const_nhds.add (hzero.mul_const ((direction i).eval x))
  exact hline.eventually (e.domain_eventually x hdomain)

/-- Observations can establish the step premises separately in each compatible completion. -/
theorem Expr.quotient_standardPart (e : Expr n) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : e.Domain x) {Γ : Commitments} (C : Completion Γ)
    (h : Sequence) (hzero : NearStandardAt C.ultrafilter h 0)
    (hnonzero : ∀ᶠ k in (C.ultrafilter : Filter ℕ), h k ≠ 0) :
    NearStandardAt C.ultrafilter (e.quotient direction x h) ((e.jvp direction).eval x) :=
  e.quotient_tendsto direction x hdomain h _ hzero hnonzero

/-- Finite vector outputs have the same componentwise standard-part guarantee. -/
theorem Program.quotient_standardPart (p : Program n m) (direction : Fin n → Expr n)
    (x : Fin n → ℝ) (hdomain : p.Domain x) {Γ : Commitments} (C : Completion Γ)
    (h : Sequence) (hzero : NearStandardAt C.ultrafilter h 0)
    (hnonzero : ∀ᶠ k in (C.ultrafilter : Filter ℕ), h k ≠ 0) :
    ∀ i, NearStandardAt C.ultrafilter ((p i).quotient direction x h)
      (((p.jvp direction) i).eval x) :=
  fun i => (p i).quotient_standardPart direction x (hdomain i) C h hzero hnonzero

#print axioms Expr.quotient_eventually_domain
#print axioms Expr.quotient_tendsto
#print axioms Expr.quotient_standardPart
#print axioms Program.quotient_standardPart

end Hyperreals.Differentiable
