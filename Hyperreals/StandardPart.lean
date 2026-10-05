import Hyperreals.Completion
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Standard parts and completion-invariant limits

Every completion extends the cofinite filter. Ordinary convergence therefore
implies the same ultrafilter limit in every completion. These definitions and
limit laws connect exact sequence calculations to completion-relative values.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals

noncomputable section

/-- A sequence converges independently of ultrafilter choice. -/
def CofiniteLimit (x : Sequence) (c : ℝ) : Prop :=
  Tendsto x Filter.cofinite (𝓝 c)

/-- Near-standardness relative to a particular ultrafilter completion. -/
def NearStandardAt (U : Ultrafilter ℕ) (x : Sequence) (c : ℝ) : Prop :=
  Tendsto x (U : Filter ℕ) (𝓝 c)

/-- A cofinite limit is a near-standard value in every compatible completion. -/
theorem nearStandardAt_of_cofiniteLimit {Γ : Commitments} {x : Sequence} {c : ℝ}
    (C : Completion Γ) (h : CofiniteLimit x c) :
    NearStandardAt C.ultrafilter x c :=
  h.mono_left C.extendsCofinite

/-- The certified standard part is independent of which completion is chosen. -/
theorem cofiniteLimit_completion_invariant {Γ : Commitments} {x : Sequence} {c : ℝ}
    (h : CofiniteLimit x c) :
    ∀ C : Completion Γ, NearStandardAt C.ultrafilter x c :=
  fun C ↦ nearStandardAt_of_cofiniteLimit C h

/-- A sequence cannot have two real limits along the same ultrafilter. -/
theorem nearStandardAt_unique {U : Ultrafilter ℕ} {x : Sequence} {a b : ℝ}
    (ha : NearStandardAt U x a) (hb : NearStandardAt U x b) : a = b :=
  tendsto_nhds_unique ha hb

namespace CofiniteLimit

theorem constant (c : ℝ) : CofiniteLimit (Sequence.constant c) c :=
  tendsto_const_nhds

theorem add {x y : Sequence} {a b : ℝ} (hx : CofiniteLimit x a)
    (hy : CofiniteLimit y b) :
    CofiniteLimit (fun n ↦ x n + y n) (a + b) :=
  Filter.Tendsto.add hx hy

theorem sub {x y : Sequence} {a b : ℝ} (hx : CofiniteLimit x a)
    (hy : CofiniteLimit y b) :
    CofiniteLimit (fun n ↦ x n - y n) (a - b) :=
  Filter.Tendsto.sub hx hy

theorem mul {x y : Sequence} {a b : ℝ} (hx : CofiniteLimit x a)
    (hy : CofiniteLimit y b) :
    CofiniteLimit (fun n ↦ x n * y n) (a * b) :=
  Filter.Tendsto.mul hx hy

theorem div {x y : Sequence} {a b : ℝ} (hx : CofiniteLimit x a)
    (hy : CofiniteLimit y b) (hb : b ≠ 0) :
    CofiniteLimit (fun n ↦ x n / y n) (a / b) :=
  Filter.Tendsto.div hx hy hb

end CofiniteLimit

/-- The distinguished sequence `1/n` has completion-invariant standard part zero. -/
def reciprocalIndex : Sequence := fun n ↦ ((n : ℝ)⁻¹)

theorem reciprocalIndex_cofiniteLimit : CofiniteLimit reciprocalIndex 0 := by
  rw [CofiniteLimit, Nat.cofinite_eq_atTop]
  change Tendsto (fun n : ℕ ↦ ((n : ℝ)⁻¹)) atTop (𝓝 0)
  exact tendsto_inv_atTop_nhds_zero_nat

theorem reciprocalIndex_nearStandardAt {Γ : Commitments} (C : Completion Γ) :
    NearStandardAt C.ultrafilter reciprocalIndex 0 :=
  nearStandardAt_of_cofiniteLimit C reciprocalIndex_cofiniteLimit

#print axioms Hyperreals.cofiniteLimit_completion_invariant
#print axioms Hyperreals.nearStandardAt_unique
#print axioms Hyperreals.reciprocalIndex_nearStandardAt

end

end Hyperreals
