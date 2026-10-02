import Hyperreals.Completion
import Hyperreals.Expressions

/-!
# Counterexample to propositional-only consistency

The current Python prototype can commit both comparison sets below for the
alternating sequence. Their intersection is empty, so no ultrafilter completion
can contain them both. This is the semantic obligation that propositional SAT
does not see when comparison atoms are treated as opaque names.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Set

namespace Hyperreals

/-- The two semantically incompatible observations reproduced in the audit. -/
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

/-- The inconsistent state instantiated at the Python DSL's alternating sequence. -/
def alternatingContradictoryComparisons : Commitments :=
  contradictoryComparisons Expr.alternatingSign.denote

/-- The two concrete observations accepted by the current Python prototype have no completion. -/
theorem alternatingContradictoryComparisons_not_extendible :
    ¬ Extendible alternatingContradictoryComparisons :=
  contradictoryComparisons_not_extendible Expr.alternatingSign.denote

#print axioms Hyperreals.contradictoryComparisons_not_extendible
#print axioms Hyperreals.contradictoryComparisons_not_hasFreeFIP
#print axioms Hyperreals.alternatingContradictoryComparisons_not_extendible

end Hyperreals
