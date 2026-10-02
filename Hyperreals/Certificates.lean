import Hyperreals.Completion
import Hyperreals.Expressions
import Mathlib.Order.Interval.Finset.Nat

/-!
# Certificates for safe comparison commitments

A certificate records one comparison polarity together with an eventual proof.
The proof is load-bearing: this module establishes the consequence checked by
the Lean kernel, but it does not claim that the Python analyzer can manufacture
such a proof from floating-point evidence.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Set

namespace Hyperreals

/-- Whether the comparison set or its complement is selected. -/
inductive ComparisonVerdict where
  | included
  | excluded
  deriving DecidableEq, Repr

/-- A proof-carrying eventual decision for one strict comparison. -/
structure ComparisonCertificate (left right : Expr) where
  verdict : ComparisonVerdict
  cutoff : ℕ
  eventually : ∀ n, cutoff ≤ n →
    match verdict with
    | .included => left.denote n < right.denote n
    | .excluded => ¬ left.denote n < right.denote n

namespace ComparisonCertificate

/-- The index set selected by a certificate. -/
def selectedSet {left right : Expr} (certificate : ComparisonCertificate left right) :
    Set ℕ :=
  match certificate.verdict with
  | .included => left.ltSet right
  | .excluded => (left.ltSet right)ᶜ

/-- An eventual membership proof makes the complement finite. -/
private theorem compl_finite_of_eventually_mem {A : Set ℕ} {cutoff : ℕ}
    (h : ∀ n, cutoff ≤ n → n ∈ A) : Aᶜ.Finite := by
  apply (Set.finite_Iio cutoff).subset
  intro n hn
  change n < cutoff
  by_contra hnot
  have hle : cutoff ≤ n := Nat.le_of_not_gt hnot
  exact hn (h n hle)

/-- Every accepted comparison certificate selects a cofinite index set. -/
theorem selectedSet_compl_finite {left right : Expr}
    (certificate : ComparisonCertificate left right) :
    certificate.selectedSetᶜ.Finite := by
  refine compl_finite_of_eventually_mem (cutoff := certificate.cutoff) ?_
  intro n hn
  cases hVerdict : certificate.verdict with
  | included =>
      simpa [selectedSet, hVerdict, Expr.ltSet, comparisonLt] using
        certificate.eventually n hn
  | excluded =>
      simpa [selectedSet, hVerdict, Expr.ltSet, comparisonLt] using
        certificate.eventually n hn

/-- A certified selection preserves any existing free-ultrafilter completion. -/
theorem preserves_extendible {Γ : Commitments} {left right : Expr}
    (hΓ : Extendible Γ) (certificate : ComparisonCertificate left right) :
    Extendible (insert certificate.selectedSet Γ) := by
  rcases hΓ with ⟨completion⟩
  exact ⟨completion.insertOfMem certificate.selectedSet
    (completion.extendsCofinite certificate.selectedSet_compl_finite)⟩

/-- A true strict comparison between constants has a direct certificate. -/
def constantsIncluded {left right : ℝ} (h : left < right) :
    ComparisonCertificate (.constant left) (.constant right) where
  verdict := .included
  cutoff := 0
  eventually := by
    intro _ _
    exact h

/-- A false strict comparison between constants has a direct certificate. -/
def constantsExcluded {left right : ℝ} (h : right ≤ left) :
    ComparisonCertificate (.constant left) (.constant right) where
  verdict := .excluded
  cutoff := 0
  eventually := by
    intro _ _
    exact not_lt_of_ge h

/-- Irreflexive strict comparison has a direct negative certificate. -/
def irreflexive (expression : Expr) :
    ComparisonCertificate expression expression where
  verdict := .excluded
  cutoff := 0
  eventually := by
    intro _ _
    exact lt_irrefl _

/-- The exponential of every real-valued expression is pointwise positive. -/
def zeroLtExp (expression : Expr) :
    ComparisonCertificate (.constant 0) expression.exp where
  verdict := .included
  cutoff := 0
  eventually := by
    intro n _
    exact Real.exp_pos (expression.denote n)

#print axioms Hyperreals.ComparisonCertificate.selectedSet_compl_finite
#print axioms Hyperreals.ComparisonCertificate.preserves_extendible

end ComparisonCertificate

end Hyperreals
