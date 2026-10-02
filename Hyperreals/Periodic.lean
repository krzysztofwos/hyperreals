import Hyperreals.PeriodicCore
import Hyperreals.Expressions
import Hyperreals.Completion
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Algebra.Ring.Commute

/-!
# An executable exact periodic comparison kernel

Rational constants and alternating signs generate two-periodic real sequences.
The compiler below computes both residues using exact rational arithmetic. Its
correctness theorem relates these computations to `Expr.denote`, rather than
assuming a certificate of that relationship. A nonempty support is sufficient
for a free completion of every observation accepted by the transition system.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Hyperreals.Periodic

noncomputable def Expr.toExact : Expr → Hyperreals.Expr
  | .constant value => .constant (value : ℝ)
  | .alternating => .alternatingSign
  | .add left right => .add left.toExact right.toExact
  | .sub left right => .sub left.toExact right.toExact
  | .mul left right => .mul left.toExact right.toExact

def parity (n : ℕ) : Bool := decide (n % 2 = 1)

theorem Expr.eval_correct (expression : Expr) (n : ℕ) :
    (expression.eval (parity n) : ℝ) = expression.toExact.denote n := by
  induction expression with
  | constant value => rfl
  | alternating =>
      rw [toExact, Hyperreals.Expr.denote_alternatingSign,
        neg_one_pow_eq_pow_mod_two]
      have hmod : n % 2 < 2 := Nat.mod_lt n (by decide)
      by_cases h : n % 2 = 1
      · simp [eval, parity, h]
      · have hz : n % 2 = 0 := by omega
        simp [eval, parity, hz]
  | add left right ihl ihr =>
      simpa [eval, toExact, Hyperreals.Expr.denote] using congrArg₂ (· + ·) ihl ihr
  | sub left right ihl ihr =>
      simpa [eval, toExact, Hyperreals.Expr.denote] using congrArg₂ (· - ·) ihl ihr
  | mul left right ihl ihr =>
      simpa [eval, toExact, Hyperreals.Expr.denote] using congrArg₂ (· * ·) ihl ihr

def Support.carrier (support : Support) : Set ℕ :=
  {n | support.at (parity n) = true}

@[simp] theorem Support.at_inter (left right : Support) (odd : Bool) :
    (left.inter right).at odd = (left.at odd && right.at odd) := by
  cases odd <;> rfl

@[simp] theorem Support.at_compl (support : Support) (odd : Bool) :
    support.compl.at odd = !support.at odd := by
  cases odd <;> rfl

theorem Support.carrier_inter (left right : Support) :
    (left.inter right).carrier = left.carrier ∩ right.carrier := by
  ext n
  simp [carrier]

theorem Support.carrier_compl (support : Support) :
    support.compl.carrier = support.carrierᶜ := by
  ext n
  simp [carrier]

theorem Support.carrier_infinite {support : Support}
    (h : support.nonempty = true) : support.carrier.Infinite := by
  have hparts : support.even = true ∨ support.odd = true := by
    simpa [nonempty] using h
  rcases hparts with heven | hodd
  · apply Set.Infinite.mono (s := Set.range (fun k : ℕ => 2 * k))
    · rintro n ⟨k, rfl⟩
      simp [carrier, Support.at, parity, heven]
    · exact Set.infinite_range_of_injective (by intro a b hab; dsimp at hab; omega)
  · apply Set.Infinite.mono (s := Set.range (fun k : ℕ => 2 * k + 1))
    · rintro n ⟨k, rfl⟩
      simp [carrier, Support.at, parity, hodd]
    · exact Set.infinite_range_of_injective (by intro a b hab; dsimp at hab; omega)

noncomputable def Comparison.denote (comparison : Comparison) (left right : Expr) : Set ℕ :=
  match comparison with
  | .lt => comparisonLt left.toExact.denote right.toExact.denote
  | .eq => comparisonEq left.toExact.denote right.toExact.denote

theorem Comparison.compile_correct (comparison : Comparison) (left right : Expr) :
    (comparison.compile left right).carrier = comparison.denote left right := by
  ext n
  have hl := left.eval_correct n
  have hr := right.eval_correct n
  cases comparison <;>
    simp only [Support.carrier, Set.mem_ofPred_eq, Comparison.compile, Support.at,
      Comparison.denote, comparisonLt, comparisonEq] <;>
    cases hp : parity n <;>
    simp_all only [Comparison.test, Bool.false_eq_true, ↓reduceIte, decide_eq_true_eq] <;>
    rw [← hl, ← hr] <;> norm_cast

noncomputable def Observation.denote (observation : Observation) : Set ℕ :=
  let comparison := observation.comparison.denote observation.left observation.right
  if observation.choice then comparison else comparisonᶜ

theorem Observation.mask_correct (observation : Observation) :
    observation.mask.carrier = observation.denote := by
  simp only [mask, denote]
  split <;> simp [Support.carrier_compl, Comparison.compile_correct]

theorem commit_sound {support next : Support} {observation : Observation}
    (h : commit support observation = some next) :
    next.nonempty = true ∧
      next.carrier = support.carrier ∩ observation.denote := by
  simp only [commit] at h
  split at h
  · cases h
    exact ⟨by assumption, by simp [restrict, Support.carrier_inter, Observation.mask_correct]⟩
  · contradiction

/-- The support contains only indices satisfying all previous commitments. -/
def Supports (support : Support) (commitments : Commitments) : Prop :=
  ∀ A ∈ commitments, support.carrier ⊆ A

theorem supports_empty : Supports Support.universe ∅ := by
  simp [Supports]

theorem commit_preserves_support {support next : Support} {observation : Observation}
    {commitments : Commitments} (hsupport : Supports support commitments)
    (h : commit support observation = some next) :
    Supports next (insert observation.denote commitments) := by
  intro A hA n hn
  rw [(commit_sound h).2] at hn
  rcases hA with rfl | hA
  · exact hn.2
  · exact hsupport A hA hn.1

theorem extendible_of_support {support : Support} {commitments : Commitments}
    (hne : support.nonempty = true) (hsupport : Supports support commitments) :
    Extendible commitments := by
  have hNeBot : Filter.NeBot (Filter.cofinite ⊓ Filter.principal support.carrier) :=
    (Support.carrier_infinite hne).cofinite_inf_principal_neBot
  rcases Filter.exists_ultrafilter_iff.mpr hNeBot with ⟨U, hU⟩
  refine ⟨⟨U, hU.trans inf_le_left, ?_⟩⟩
  intro A hA
  have hmem : support.carrier ∈ U :=
    (hU.trans inf_le_right) (Filter.mem_principal_self support.carrier)
  exact Filter.mem_of_superset hmem (hsupport A hA)

theorem commit_extendible {support next : Support} {observation : Observation}
    {commitments : Commitments} (hsupport : Supports support commitments)
    (h : commit support observation = some next) :
    Extendible (insert observation.denote commitments) :=
  extendible_of_support (commit_sound h).1 (commit_preserves_support hsupport h)

theorem run_sound {support next : Support} {observations : List Observation}
    {commitments : Commitments} (hne : support.nonempty = true)
    (hsupport : Supports support commitments)
    (h : run support observations = some next) :
    next.nonempty = true ∧ Supports next commitments ∧
      ∀ observation ∈ observations, next.carrier ⊆ observation.denote := by
  induction observations generalizing support commitments with
  | nil =>
      simp only [run, Option.some.injEq] at h
      subst next
      exact ⟨hne, hsupport, by simp⟩
  | cons observation rest ih =>
      simp only [run] at h
      cases hc : commit support observation with
      | none => simp [hc] at h
      | some intermediate =>
          simp only [hc, Option.bind_some] at h
          have htail := ih (commit_sound hc).1 (commit_preserves_support hsupport hc) h
          refine ⟨htail.1, ?_, ?_⟩
          · intro A hA
            exact htail.2.1 A (Set.mem_insert_of_mem _ hA)
          · intro selected hselected
            rcases List.mem_cons.mp hselected with rfl | hselected
            · exact htail.2.1 selected.denote (Set.mem_insert _ _)
            · exact htail.2.2 selected hselected

/-- Every successfully replayed trace has a genuine free-ultrafilter completion. -/
theorem run_trace_extendible {observations : List Observation} {next : Support}
    (h : run Support.universe observations = some next) :
    Extendible {A | ∃ observation ∈ observations, A = observation.denote} := by
  have hs := run_sound (by rfl : Support.universe.nonempty = true) supports_empty h
  apply extendible_of_support hs.1
  rintro A ⟨observation, hmem, rfl⟩
  exact hs.2.2 observation hmem

#print axioms Hyperreals.Periodic.Expr.eval_correct
#print axioms Hyperreals.Periodic.Comparison.compile_correct
#print axioms Hyperreals.Periodic.commit_sound
#print axioms Hyperreals.Periodic.commit_extendible
#print axioms Hyperreals.Periodic.run_trace_extendible

end Hyperreals.Periodic
