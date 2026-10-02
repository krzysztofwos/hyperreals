import Hyperreals.ResidueReplay
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Real.Hyperreal
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# An exact infinitesimal difference quotient for the cubic

For ε(n) = 1/n, the quotient ((2 + ε)^3 - 8)/ε is
12 + 6ε + ε² at every positive index. It is strictly greater than 12
there, yet has standard part 12 in every free completion. The ordinary
derivative at 2 is proved separately. No differentiation rule for arbitrary
expressions or Python-to-Lean capture refinement is assumed.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.InfinitesimalCase

noncomputable def epsilon : Sequence := reciprocalIndex

noncomputable def dy : Sequence := fun n ↦ (2 + epsilon n) ^ 3 - 2 ^ 3

noncomputable def quotient : Sequence := fun n ↦ dy n / epsilon n

noncomputable def error : Sequence := fun n ↦ quotient n - 12

theorem epsilon_pos (n : ℕ) (hn : 0 < n) : 0 < epsilon n := by
  unfold epsilon reciprocalIndex
  positivity

private theorem eventually_index_pos : ∀ᶠ n : ℕ in Filter.cofinite, 0 < n := by
  rw [Nat.cofinite_eq_atTop]
  exact eventually_gt_atTop 0

/-- The increment is positive and smaller in magnitude than every positive
real in every free completion, including completions of nonempty traces. -/
theorem epsilon_positive_infinitesimal {Γ : Commitments} (C : Completion Γ) :
    (∀ᶠ n in (C.ultrafilter : Filter ℕ), 0 < epsilon n) ∧
      ∀ bound : ℝ, 0 < bound →
        ∀ᶠ n in (C.ultrafilter : Filter ℕ), |epsilon n| < bound := by
  constructor
  · exact C.extendsCofinite (eventually_index_pos.mono fun n hn ↦ epsilon_pos n hn)
  · intro bound hbound
    have hlimit : Tendsto epsilon (C.ultrafilter : Filter ℕ) (𝓝 (0 : ℝ)) :=
      reciprocalIndex_nearStandardAt C
    have habs : Tendsto (fun n ↦ |epsilon n|) (C.ultrafilter : Filter ℕ) (𝓝 0) := by
      simpa using hlimit.abs
    exact habs.eventually_lt_const hbound

/-- Nonzero means nonzero at almost every index of the completion, not at
the exceptional index zero where totalized real inversion returns zero. -/
theorem epsilon_nonzero {Γ : Commitments} (C : Completion Γ) :
    ∀ᶠ n in (C.ultrafilter : Filter ℕ), epsilon n ≠ 0 :=
  (epsilon_positive_infinitesimal C).1.mono fun _ hn ↦ ne_of_gt hn

/-- Exact finite-index algebra, before taking any standard part. -/
theorem quotient_identity (n : ℕ) (hn : 0 < n) :
    quotient n = 12 + 6 * epsilon n + epsilon n ^ 2 := by
  have hnonzero : epsilon n ≠ 0 := ne_of_gt (epsilon_pos n hn)
  unfold quotient dy
  field_simp
  ring

/-- The difference quotient is not exactly the derivative. Its positive
error only disappears after standard-part extraction. -/
theorem quotient_strictly_above_derivative (n : ℕ) (hn : 0 < n) :
    12 < quotient n := by
  rw [quotient_identity n hn]
  have heps := epsilon_pos n hn
  nlinarith [sq_nonneg (epsilon n)]

theorem quotient_cofiniteLimit : CofiniteLimit quotient 12 := by
  have hexpanded : CofiniteLimit (fun n ↦ 12 + 6 * epsilon n + epsilon n ^ 2) 12 := by
    have h := ((CofiniteLimit.constant 12).add
      ((CofiniteLimit.constant 6).mul reciprocalIndex_cofiniteLimit)).add
      (reciprocalIndex_cofiniteLimit.pow 2)
    simpa [Sequence.constant, epsilon] using h
  have heq : quotient =ᶠ[Filter.cofinite]
      (fun n ↦ 12 + 6 * epsilon n + epsilon n ^ 2) :=
    eventually_index_pos.mono fun n hn ↦ quotient_identity n hn
  exact hexpanded.congr' heq.symm

/-- The quotient has standard part 12 regardless of later compatible choices. -/
theorem quotient_standardPart {Γ : Commitments} (C : Completion Γ) :
    NearStandardAt C.ultrafilter quotient 12 :=
  nearStandardAt_of_cofiniteLimit C quotient_cofiniteLimit

/-- The nonzero error from the ordinary derivative is infinitesimal. -/
theorem error_infinitesimal {Γ : Commitments} (C : Completion Γ) :
    NearStandardAt C.ultrafilter error 0 := by
  have h : CofiniteLimit error 0 := by
    change CofiniteLimit (fun n ↦ quotient n - 12) 0
    simpa [Sequence.constant] using
      quotient_cofiniteLimit.sub (CofiniteLimit.constant 12)
  exact nearStandardAt_of_cofiniteLimit C h

theorem quotient_ne_derivative {Γ : Commitments} (C : Completion Γ) :
    ∀ᶠ n in (C.ultrafilter : Filter ℕ), quotient n ≠ 12 := by
  exact C.extendsCofinite (eventually_index_pos.mono
    fun n hn ↦ ne_of_gt (quotient_strictly_above_derivative n hn))

/-- Independent agreement with the usual real derivative. -/
theorem cubic_hasDerivAt : HasDerivAt (fun x : ℝ ↦ x ^ 3) 12 2 := by
  have h := hasDerivAt_pow 3 (2 : ℝ)
  norm_num at h
  exact h

/-- The multiplication tree emitted by the public Python power operation
for power three. This definition contains no assumed correctness fields. -/
def cubeExpr (expression : Residue.Expr) : Residue.Expr :=
  .mul (.mul (.constant 1) expression) (.mul expression expression)

/-- The original difference quotient syntax, with no supplied expansion or limit. -/
def quotientExpr : Residue.Expr :=
  .divMonomial
    (.sub (cubeExpr (.add (.constant 2) .reciprocalIndex)) (cubeExpr (.constant 2)))
    1 (.negSucc 0)

theorem quotientExpr_denote (n : ℕ) : quotientExpr.denote n = quotient n := by
  dsimp only [quotientExpr, cubeExpr, Residue.Expr.denote, Residue.Expr.eval,
    Laurent.monomial, quotient, dy, epsilon, reciprocalIndex]
  norm_num
  ring_nf
  simp

/-- The executable extractor computes 12 on universal support without choices. -/
theorem quotient_extracts : Residue.standardPart [true] quotientExpr = some 12 := by
  decide +kernel

/-- The same fraction in Mathlib's existing hyperreal field, whose chosen
ultrafilter is one particular free completion. -/
noncomputable def hyperrealQuotient : Hyperreal :=
  ((2 + Hyperreal.epsilon) ^ 3 - 2 ^ 3) / Hyperreal.epsilon

theorem hyperreal_quotient_ofSeq : hyperrealQuotient = Hyperreal.ofSeq quotient := by
  rfl

theorem hyperreal_quotient_identity :
    hyperrealQuotient = 12 + 6 * Hyperreal.epsilon + Hyperreal.epsilon ^ 2 := by
  unfold hyperrealQuotient
  field_simp
  ring

theorem hyperreal_quotient_gt_derivative : (12 : Hyperreal) < hyperrealQuotient := by
  rw [hyperreal_quotient_identity]
  have hpositive := Hyperreal.epsilon_pos
  nlinarith [sq_nonneg Hyperreal.epsilon]

/-- Literal standard-part equality in the existing quotient field. The proof
uses the sequence limit, rather than an assumed derivative transfer rule. -/
theorem hyperreal_quotient_standardPart :
    ArchimedeanClass.stdPart
      (((2 + Hyperreal.epsilon) ^ 3 - 8) / Hyperreal.epsilon) = 12 := by
  have hlimit : (Hyperreal.ofSeq quotient).Tendsto (𝓝 (12 : ℝ)) :=
    Hyperreal.tendsto_ofSeq.mpr
      (quotient_cofiniteLimit.mono_left Filter.hyperfilter_le_cofinite)
  have h := Hyperreal.stdPart_of_tendsto hlimit
  rw [← hyperreal_quotient_ofSeq] at h
  norm_num [hyperrealQuotient] at h
  exact h

#print axioms Hyperreals.InfinitesimalCase.epsilon_positive_infinitesimal
#print axioms Hyperreals.InfinitesimalCase.epsilon_nonzero
#print axioms Hyperreals.InfinitesimalCase.quotient_identity
#print axioms Hyperreals.InfinitesimalCase.quotient_strictly_above_derivative
#print axioms Hyperreals.InfinitesimalCase.quotient_standardPart
#print axioms Hyperreals.InfinitesimalCase.error_infinitesimal
#print axioms Hyperreals.InfinitesimalCase.quotient_ne_derivative
#print axioms Hyperreals.InfinitesimalCase.cubic_hasDerivAt
#print axioms Hyperreals.InfinitesimalCase.quotientExpr_denote
#print axioms Hyperreals.InfinitesimalCase.quotient_extracts
#print axioms Hyperreals.InfinitesimalCase.hyperreal_quotient_ofSeq
#print axioms Hyperreals.InfinitesimalCase.hyperreal_quotient_identity
#print axioms Hyperreals.InfinitesimalCase.hyperreal_quotient_gt_derivative
#print axioms Hyperreals.InfinitesimalCase.hyperreal_quotient_standardPart

end Hyperreals.InfinitesimalCase
