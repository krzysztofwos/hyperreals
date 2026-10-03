import Hyperreals.Completion
import Hyperreals.Expressions

/-!
# Counterexample to propositional-only consistency

The comparison sets below are disjoint, so no ultrafilter completion can
contain them both. For the alternating sequence, each set is individually
compatible with a free completion. Treating the comparisons as independent
Boolean atoms loses this obstruction to their joint consistency.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Set

namespace Hyperreals

/-- Two semantically incompatible observations of the same sequence. -/
def contradictoryComparisons (a : Sequence) : Commitments :=
  {comparisonLt a (Sequence.constant 0), comparisonEq a (Sequence.constant 1)}

theorem negative_inter_one_empty (a : Sequence) :
    comparisonLt a (Sequence.constant 0) ∩
        comparisonEq a (Sequence.constant 1) = ∅ := by
  apply Set.eq_empty_iff_forall_notMem.2
  intro n hn
  rcases hn with ⟨hlt, heq⟩
  change a n < 0 at hlt
  change a n = 1 at heq
  rw [heq] at hlt
  exact (not_lt_of_ge (zero_le_one : (0 : ℝ) ≤ 1)) hlt

/-- No ultrafilter can realize both `a < 0` and `a = 1`. -/
theorem contradictoryComparisons_not_extendible (a : Sequence) :
    ¬ Extendible (contradictoryComparisons a) := by
  rintro ⟨C⟩
  have hlt : comparisonLt a (Sequence.constant 0) ∈ C.ultrafilter :=
    C.contains (by simp [contradictoryComparisons])
  have heq : comparisonEq a (Sequence.constant 1) ∈ C.ultrafilter :=
    C.contains (by simp [contradictoryComparisons])
  have hinter := Filter.inter_mem hlt heq
  rw [negative_inter_one_empty] at hinter
  exact C.ultrafilter.empty_notMem hinter

/-- Consequently, the semantic free-FIP certificate cannot exist either. -/
theorem contradictoryComparisons_not_hasFreeFIP (a : Sequence) :
    ¬ HasFreeFIP (contradictoryComparisons a) := by
  rw [hasFreeFIP_iff_extendible]
  exact contradictoryComparisons_not_extendible a

/-- The incompatible observations instantiated at the alternating sequence. -/
def alternatingContradictoryComparisons : Commitments :=
  contradictoryComparisons Expr.alternatingSign.denote

/-- The two concrete alternating-sequence observations have no common completion. -/
theorem alternatingContradictoryComparisons_not_extendible :
    ¬ Extendible alternatingContradictoryComparisons :=
  contradictoryComparisons_not_extendible Expr.alternatingSign.denote

#print axioms Hyperreals.contradictoryComparisons_not_extendible
#print axioms Hyperreals.contradictoryComparisons_not_hasFreeFIP
#print axioms Hyperreals.alternatingContradictoryComparisons_not_extendible

end Hyperreals
