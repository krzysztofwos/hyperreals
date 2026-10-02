import Hyperreals.ResidueComparisonCore
import Hyperreals.ResidueExpr
import Hyperreals.ResidueSupport
import Hyperreals.LaurentSign

/-! The finite-table compiler's executable mask and maximum cutoff agree with
the actual comparison of source sequences at every later natural index. -/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

def compareReal (comparison : Comparison) (left right : ℝ) : Prop :=
  match comparison with
  | .lt => left < right
  | .eq => left = right

noncomputable def comparisonSet (comparison : Comparison) (left right : Expr) : Set ℕ :=
  {n | compareReal comparison (left.denote n) (right.denote n)}

theorem formsCutoff_pos (forms : List Laurent.Form) : 1 ≤ formsCutoff forms := by
  induction forms with
  | nil => exact le_rfl
  | cons form rest ih => exact ih.trans (Nat.le_max_right _ _)

theorem form_cutoff_le_of_mem (forms : List Laurent.Form) (form : Laurent.Form)
    (hmem : form ∈ forms) : form.cutoff ≤ formsCutoff forms := by
  induction forms with
  | nil => simp at hmem
  | cons first rest ih =>
      rcases List.mem_cons.mp hmem with rfl | hmem
      · exact Nat.le_max_left _ _
      · exact (ih hmem).trans (Nat.le_max_right _ _)

theorem compile_mask_length (comparison : Comparison) (left right : Expr) :
    (compile comparison left right).mask.length = Nat.lcm left.period right.period := by
  simp [compile, comparisonForms]

theorem compile_mask_pos (comparison : Comparison) (left right : Expr)
    (hl : left.valid = true) (hr : right.valid = true) :
    0 < (compile comparison left right).mask.length := by
  rw [compile_mask_length]
  exact Nat.lcm_pos (left.period_pos hl) (right.period_pos hr)

theorem compile_one_le_cutoff (comparison : Comparison) (left right : Expr) :
    1 ≤ (compile comparison left right).cutoff :=
  formsCutoff_pos (comparisonForms left right)

private theorem compareForm_correct (comparison : Comparison) (form : Laurent.Form) (n : ℕ)
    (hn : form.cutoff ≤ n) :
    compareForm comparison form = true ↔ compareReal comparison (form.eval n) 0 := by
  have h := Laurent.Form.sign_correct form hn
  cases comparison with
  | lt => simpa [compareForm, compareReal] using h.1
  | eq => simpa [compareForm, compareReal] using h.2.1

/-- Computed comparisons are correct at every index beyond their reported
cutoff, even when their two operands have different finite periods. -/
theorem compile_correct (comparison : Comparison) (left right : Expr)
    (hl : left.valid = true) (hr : right.valid = true) (n : ℕ)
    (hn : (compile comparison left right).cutoff ≤ n) :
    (compile comparison left right).mask.at n = true ↔
      n ∈ comparisonSet comparison left right := by
  have hperiod : 0 < Nat.lcm left.period right.period :=
    Nat.lcm_pos (left.period_pos hl) (right.period_pos hr)
  have hresidue : n % Nat.lcm left.period right.period < Nat.lcm left.period right.period :=
    Nat.mod_lt n hperiod
  have hnpositive : 0 < n := lt_of_lt_of_le Nat.zero_lt_one
    ((compile_one_le_cutoff comparison left right).trans hn)
  have hnnonzero : (n : ℝ) ≠ 0 := by exact_mod_cast hnpositive.ne'
  let residue := n % Nat.lcm left.period right.period
  let form := (left.normalizeAt residue).sub (right.normalizeAt residue)
  have hform : form ∈ comparisonForms left right := by
    apply List.mem_map.mpr
    exact ⟨residue, List.mem_range.mpr hresidue, rfl⟩
  have hcutoff : form.cutoff ≤ n :=
    (form_cutoff_le_of_mem (comparisonForms left right) form hform).trans hn
  have hmask : (compile comparison left right).mask.at n = compareForm comparison form := by
    simp only [compile, comparisonForms, Support.at, List.length_map, List.length_range,
      List.map_map]
    exact getD_range_map _ _ _ _ hresidue
  have heval : form.eval n = left.denote n - right.denote n := by
    dsimp only [form, residue]
    rw [Laurent.Form.eval_sub _ _ _ hnnonzero]
    rw [left.normalizeAt_residue_correct n _ (Nat.dvd_lcm_left _ _) hnpositive hl,
      right.normalizeAt_residue_correct n _ (Nat.dvd_lcm_right _ _) hnpositive hr]
  rw [hmask, compareForm_correct comparison form n hcutoff, heval]
  cases comparison <;> simp [compareReal, comparisonSet, sub_neg, sub_eq_zero]

#print axioms Hyperreals.Residue.compile_mask_pos
#print axioms Hyperreals.Residue.compile_one_le_cutoff
#print axioms Hyperreals.Residue.compile_correct

end Hyperreals.Residue
