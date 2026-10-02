import Hyperreals.Completion

/-!
# Executable infinitude checks for eventual-periodic certificates

This is the first-order certificate format mirrored by the Python runtime.
A certificate records a periodic tail but deliberately says nothing about the
finite prefix, which is invisible to every free ultrafilter.

The Boolean checker is executable. Its soundness theorem proves that an accepted
certificate denotes an infinite set. Relating a Python comparison expression to
the certificate it emits remains a separate refinement obligation.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals

/-- Raw, serializable data describing a periodic tail. -/
structure EventuallyPeriodicSet where
  cutoff : ℕ
  period : ℕ
  residues : Finset ℕ
  deriving DecidableEq

namespace EventuallyPeriodicSet

/-- The concrete tail denoted by a certificate. Its unspecified prefix is omitted. -/
def carrier (certificate : EventuallyPeriodicSet) : Set ℕ :=
  {n | certificate.cutoff ≤ n ∧ n % certificate.period ∈ certificate.residues}

/-- Residues that can actually occur modulo the declared period. -/
def admissibleResidues (certificate : EventuallyPeriodicSet) : Finset ℕ :=
  certificate.residues.filter (fun residue ↦ residue < certificate.period)

/-- Executable check that the certificate contains a recurring residue. -/
def checkInfinite (certificate : EventuallyPeriodicSet) : Bool :=
  decide (certificate.period ≠ 0) &&
    decide (certificate.admissibleResidues ≠ ∅)

/-- One admissible recurring residue generates infinitely many members. -/
theorem carrier_infinite_of_residue {certificate : EventuallyPeriodicSet}
    (hperiod : 0 < certificate.period)
    {residue : ℕ} (hresidue : residue ∈ certificate.residues)
    (hresidue_lt : residue < certificate.period) :
    certificate.carrier.Infinite := by
  let index : ℕ → ℕ :=
    fun k ↦ residue + certificate.period * (certificate.cutoff + k)
  have hinjective : Function.Injective index := by
    intro left right heq
    dsimp [index] at heq
    apply Nat.add_left_cancel
    apply Nat.mul_left_cancel hperiod
    exact Nat.add_left_cancel heq
  have hrange : Set.range index ⊆ certificate.carrier := by
    rintro n ⟨k, rfl⟩
    constructor
    · dsimp [index]
      have hscale :
          certificate.cutoff ≤ certificate.period * certificate.cutoff :=
        Nat.le_mul_of_pos_left certificate.cutoff hperiod
      have hshift :
          certificate.period * certificate.cutoff ≤
            certificate.period * (certificate.cutoff + k) :=
        Nat.mul_le_mul_left certificate.period
          (Nat.le_add_right certificate.cutoff k)
      exact hscale.trans (hshift.trans
        (Nat.le_add_left
          (certificate.period * (certificate.cutoff + k)) residue))
    · change
        (residue + certificate.period * (certificate.cutoff + k)) %
            certificate.period ∈ certificate.residues
      rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hresidue_lt]
      exact hresidue
  exact Set.Infinite.mono hrange (Set.infinite_range_of_injective hinjective)

/-- Every certificate accepted by the executable checker denotes an infinite tail. -/
theorem checkInfinite_sound {certificate : EventuallyPeriodicSet}
    (hcheck : certificate.checkInfinite = true) :
    certificate.carrier.Infinite := by
  have hparts :
      decide (certificate.period ≠ 0) = true ∧
        decide (certificate.admissibleResidues ≠ ∅) = true :=
    by simpa only [checkInfinite, Bool.and_eq_true] using hcheck
  have hperiod_ne : certificate.period ≠ 0 :=
    of_decide_eq_true hparts.1
  have hadmissible_ne : certificate.admissibleResidues ≠ ∅ :=
    of_decide_eq_true hparts.2
  rcases Finset.nonempty_iff_ne_empty.mpr hadmissible_ne with
    ⟨residue, hresidue⟩
  have hfiltered := Finset.mem_filter.mp hresidue
  exact carrier_infinite_of_residue (Nat.pos_of_ne_zero hperiod_ne)
    hfiltered.1 hfiltered.2

#print axioms Hyperreals.EventuallyPeriodicSet.checkInfinite_sound

end EventuallyPeriodicSet

end Hyperreals
