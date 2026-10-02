import Mathlib.Data.Real.Basic

/-!
# Exact semantics for sequence comparisons

This module gives the small semantic layer needed by the lazy-ultrafilter
argument. A sequence denotes an actual function `ℕ → ℝ`. A comparison denotes
the corresponding subset of indices. No syntactic identity or floating-point
approximation is used here.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals

/-- A representative sequence for a hyperreal value. -/
abbrev Sequence := ℕ → ℝ

/-- The constant real sequence. -/
def Sequence.constant (c : ℝ) : Sequence := fun _ ↦ c

/-- The set of indices where `a` is strictly less than `b`. -/
def comparisonLt (a b : Sequence) : Set ℕ := {n | a n < b n}

/-- The set of indices where `a` equals `b`. -/
def comparisonEq (a b : Sequence) : Set ℕ := {n | a n = b n}

/-- Pointwise trichotomy makes the three comparison sets a cover of `ℕ`. -/
theorem comparison_trichotomy (a b : Sequence) :
    comparisonLt a b ∪ comparisonEq a b ∪ comparisonLt b a = Set.univ := by
  ext n
  simp only [comparisonLt, comparisonEq, Set.mem_union, Set.mem_ofPred_eq,
    Set.mem_univ, iff_true]
  rcases lt_trichotomy (a n) (b n) with h | h | h
  · exact Or.inl (Or.inl h)
  · exact Or.inl (Or.inr h)
  · exact Or.inr h

/-- Strict comparison and equality for the same ordered pair are disjoint. -/
theorem comparisonLt_disjoint_comparisonEq (a b : Sequence) :
    Disjoint (comparisonLt a b) (comparisonEq a b) := by
  refine Set.disjoint_left.2 ?_
  intro n hlt heq
  exact (ne_of_lt hlt) heq

/-- Opposite strict comparisons are disjoint. -/
theorem comparisonLt_disjoint_reverse (a b : Sequence) :
    Disjoint (comparisonLt a b) (comparisonLt b a) := by
  refine Set.disjoint_left.2 ?_
  intro n hab hba
  exact (lt_asymm (show a n < b n from hab)) (show b n < a n from hba)

end Hyperreals
