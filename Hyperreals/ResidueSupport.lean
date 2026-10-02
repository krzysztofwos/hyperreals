import Hyperreals.ResidueSupportCore
import Hyperreals.Completion
import Mathlib.Data.List.GetD

/-!
# Correctness of finite-period support refinement

Period lifting and LCM intersection preserve the represented subsets of the
naturals exactly. Every accepted nonempty support contains an infinite residue
class, supplying the semantic support needed for free-ultrafilter completions.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Residue

def Support.carrier (support : Support) : Set ℕ := {n | support.at n = true}

theorem getD_range_map {α : Type*} (function : ℕ → α) (default : α)
    (period index : ℕ) (hindex : index < period) :
    ((List.range period).map function).getD index default = function index := by
  rw [List.getD_eq_getElem _ _ (by simpa using hindex)]
  simp

theorem Support.length_pos_of_nonempty {support : Support}
    (h : support.nonempty = true) : 0 < support.length := by
  cases support with
  | nil => simp [Support.nonempty] at h
  | cons _ _ => simp

@[simp] theorem Support.length_lift (support : Support) (period : ℕ) :
    (support.lift period).length = period := by
  simp [Support.lift]

@[simp] theorem Support.length_inter (left right : Support) :
    (left.inter right).length = Nat.lcm left.length right.length := by
  simp [Support.inter]

@[simp] theorem Support.length_compl (support : Support) :
    support.compl.length = support.length := by
  simp [Support.compl]

theorem Support.at_mod (support : Support) (period n : ℕ)
    (hdiv : support.length ∣ period) : support.at (n % period) = support.at n := by
  simp only [Support.at, Nat.mod_mod_of_dvd n hdiv]

/-- A positive multiple of the original period represents the same membership. -/
theorem Support.at_lift (support : Support) (period : ℕ)
    (hperiod : 0 < period) (hdiv : support.length ∣ period) (n : ℕ) :
    (support.lift period).at n = support.at n := by
  simp only [Support.at, Support.lift, List.length_map, List.length_range]
  rw [getD_range_map _ _ _ _ (Nat.mod_lt n hperiod)]
  simp only [Nat.mod_mod_of_dvd n hdiv]

theorem Support.carrier_lift (support : Support) (period : ℕ)
    (hperiod : 0 < period) (hdiv : support.length ∣ period) :
    (support.lift period).carrier = support.carrier := by
  ext n
  simp only [Support.carrier, Set.mem_ofPred_eq, support.at_lift period hperiod hdiv]

theorem Support.at_inter (left right : Support)
    (hleft : 0 < left.length) (hright : 0 < right.length) (n : ℕ) :
    (left.inter right).at n = (left.at n && right.at n) := by
  simp only [Support.at, Support.inter, List.length_map, List.length_range]
  rw [getD_range_map _ _ _ _ (Nat.mod_lt n (Nat.lcm_pos hleft hright))]
  simp only [Nat.mod_mod_of_dvd n (Nat.dvd_lcm_left _ _),
    Nat.mod_mod_of_dvd n (Nat.dvd_lcm_right _ _)]

/-- Changing to a common period preserves precisely the intersection. -/
theorem Support.carrier_inter (left right : Support)
    (hleft : 0 < left.length) (hright : 0 < right.length) :
    (left.inter right).carrier = left.carrier ∩ right.carrier := by
  ext n
  simp [Support.carrier, Support.at_inter left right hleft hright]

theorem Support.at_compl (support : Support) (hperiod : 0 < support.length) (n : ℕ) :
    support.compl.at n = !support.at n := by
  simp only [Support.at, Support.compl, List.length_map]
  rw [List.getD_eq_getElem _ _ (by simpa using Nat.mod_lt n hperiod),
    List.getD_eq_getElem _ _ (Nat.mod_lt n hperiod)]
  simp

theorem Support.carrier_compl (support : Support) (hperiod : 0 < support.length) :
    support.compl.carrier = support.carrierᶜ := by
  ext n
  simp [Support.carrier, Support.at_compl support hperiod]

@[simp] theorem Support.at_universe (n : ℕ) : Support.universe.at n = true := by
  simp [Support.at, Support.universe, Nat.mod_one]

@[simp] theorem Support.carrier_universe : Support.universe.carrier = Set.univ := by
  ext n
  simp [Support.carrier]

/-- One selected residue produces infinitely many supported natural indices. -/
theorem Support.carrier_infinite {support : Support}
    (h : support.nonempty = true) : support.carrier.Infinite := by
  have hperiod := Support.length_pos_of_nonempty h
  have htrue : true ∈ support := by
    simpa [Support.nonempty] using h
  obtain ⟨residue, hresidue, hselected⟩ := List.mem_iff_getElem.mp htrue
  let index : ℕ → ℕ := fun k => residue + support.length * k
  have hinjective : Function.Injective index := by
    intro a b hab
    dsimp [index] at hab
    exact Nat.eq_of_mul_eq_mul_left hperiod (Nat.add_left_cancel hab)
  have hrange : Set.range index ⊆ support.carrier := by
    rintro n ⟨k, rfl⟩
    simp only [Support.carrier, Set.mem_ofPred_eq, Support.at, index,
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hresidue]
    rw [List.getD_eq_getElem _ _ hresidue]
    exact hselected
  exact Set.Infinite.mono hrange (Set.infinite_range_of_injective hinjective)

#print axioms Hyperreals.Residue.Support.carrier_lift
#print axioms Hyperreals.Residue.Support.carrier_inter
#print axioms Hyperreals.Residue.Support.carrier_compl
#print axioms Hyperreals.Residue.Support.carrier_infinite

end Hyperreals.Residue
