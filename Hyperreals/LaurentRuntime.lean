import Hyperreals.LaurentRuntimeCore
import Hyperreals.LaurentSign
import Hyperreals.Periodic
import Hyperreals.LaurentStandardPart

/-!
# Eventual comparisons and finite traces for periodic Laurent expressions

The compiler derives its mask and cutoff from exact normalization and the
proved sign algorithm. Finite prefixes may disagree with the mask. Every
committed observation nevertheless belongs to one joint free completion.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter

namespace Hyperreals.Laurent

def compareReal (comparison : Comparison) (left right : ℝ) : Prop :=
  match comparison with
  | .lt => left < right
  | .eq => left = right

noncomputable def comparisonSet (comparison : Comparison) (left right : Expr) : Set ℕ :=
  {n | compareReal comparison (left.denote n) (right.denote n)}

theorem compile_one_le_cutoff (comparison : Comparison) (left right : Expr) :
    1 ≤ (compile comparison left right).cutoff :=
  (Poly.one_le_cutoff _).trans (Nat.le_max_left _ _)

private theorem compareForm_correct (comparison : Comparison) (form : Form) (n : ℕ)
    (hn : form.cutoff ≤ n) :
    compareForm comparison form = true ↔ compareReal comparison (form.eval n) 0 := by
  have h := Form.sign_correct form hn
  cases comparison with
  | lt => simpa [compareForm, compareReal] using h.1
  | eq => simpa [compareForm, compareReal] using h.2.1

/-- The reported natural cutoff validates the mask at every later index. -/
theorem compile_correct (comparison : Comparison) (left right : Expr)
    (hl : left.valid = true) (hr : right.valid = true) (n : ℕ)
    (hn : (compile comparison left right).cutoff ≤ n) :
    (compile comparison left right).mask.at (Periodic.parity n) = true ↔
      n ∈ comparisonSet comparison left right := by
  have hnpos : 0 < n := lt_of_lt_of_le Nat.zero_lt_one
    ((compile_one_le_cutoff comparison left right).trans hn)
  have hx : (n : ℝ) ≠ 0 := by exact_mod_cast hnpos.ne'
  have heval : ((left.normalize (Periodic.parity n)).sub
      (right.normalize (Periodic.parity n))).eval n = left.denote n - right.denote n := by
    rw [Form.eval_sub _ _ _ hx]
    rw [Periodic.parity, left.normalize_sequence_correct n hnpos hl,
      right.normalize_sequence_correct n hnpos hr]
  have hcut : ((left.normalize (Periodic.parity n)).sub
      (right.normalize (Periodic.parity n))).cutoff ≤ n := by
    cases hp : Periodic.parity n
    · exact (Nat.le_max_left _ _).trans hn
    · exact (Nat.le_max_right _ _).trans hn
  have hform := compareForm_correct comparison
    ((left.normalize (Periodic.parity n)).sub (right.normalize (Periodic.parity n))) n hcut
  have hmask : (compile comparison left right).mask.at (Periodic.parity n) =
      compareForm comparison
        ((left.normalize (Periodic.parity n)).sub (right.normalize (Periodic.parity n))) := by
    cases hp : Periodic.parity n <;> rfl
  rw [hmask, hform, heval]
  cases comparison <;> simp [compareReal, comparisonSet, sub_neg, sub_eq_zero]

noncomputable def Observation.denote (observation : Observation) : Set ℕ :=
  let selected := comparisonSet observation.comparison observation.left observation.right
  if observation.choice then selected else selectedᶜ

theorem Observation.mask_correct (observation : Observation)
    (hvalid : observation.valid = true) :
    ∀ᶠ n in Filter.cofinite, n ∈ observation.mask.carrier ↔ n ∈ observation.denote := by
  have hv : observation.left.valid = true ∧ observation.right.valid = true := by
    simpa [Observation.valid] using hvalid
  rw [Nat.cofinite_eq_atTop]
  filter_upwards [eventually_ge_atTop
    (compile observation.comparison observation.left observation.right).cutoff] with n hn
  have hc := compile_correct observation.comparison observation.left observation.right
    hv.1 hv.2 n hn
  cases hchoice : observation.choice
  · simp only [Observation.mask, Observation.denote, hchoice, Bool.false_eq_true,
      ↓reduceIte, Periodic.Support.carrier_compl, Set.mem_compl_iff]
    exact not_congr hc
  · simpa only [Observation.mask, Observation.denote, hchoice, ↓reduceIte,
      Periodic.Support.carrier, Set.mem_ofPred_eq] using hc

/-- Each recorded observation is respected on a cofinite part of the support. -/
def Supports (support : Support) (commitments : Commitments) : Prop :=
  ∀ A ∈ commitments, ∀ᶠ n in Filter.cofinite, n ∈ support.carrier → n ∈ A

theorem supports_empty : Supports Periodic.Support.universe ∅ := by
  simp [Supports]

theorem commit_sound {support next : Support} {observation : Observation}
    (h : commit support observation = some next) :
    observation.valid = true ∧ next.nonempty = true ∧
      next = support.inter observation.mask := by
  simp only [commit] at h
  split at h
  · cases h
    rename_i hcheck
    have hparts : observation.valid = true ∧
        (support.inter observation.mask).nonempty = true := by simpa using hcheck
    exact ⟨hparts.1, hparts.2, rfl⟩
  · contradiction

theorem commit_preserves_support {support next : Support} {observation : Observation}
    {commitments : Commitments} (hsupport : Supports support commitments)
    (h : commit support observation = some next) :
    Supports next (insert observation.denote commitments) := by
  have hs := commit_sound h
  intro A hA
  rcases hA with rfl | hA
  · filter_upwards [observation.mask_correct hs.1] with n hn
    intro hnext
    rw [hs.2.2, Periodic.Support.carrier_inter] at hnext
    exact hn.mp hnext.2
  · filter_upwards [hsupport A hA] with n hn
    intro hnext
    rw [hs.2.2, Periodic.Support.carrier_inter] at hnext
    exact hn hnext.1

theorem extendible_of_support {support : Support} {commitments : Commitments}
    (hne : support.nonempty = true) (hsupport : Supports support commitments) :
    Extendible commitments := by
  have hNeBot : Filter.NeBot (Filter.cofinite ⊓ Filter.principal support.carrier) :=
    (Periodic.Support.carrier_infinite hne).cofinite_inf_principal_neBot
  rcases Filter.exists_ultrafilter_iff.mpr hNeBot with ⟨U, hU⟩
  have hcof : (U : Filter ℕ) ≤ Filter.cofinite := hU.trans inf_le_left
  have hmem : support.carrier ∈ U :=
    (hU.trans inf_le_right) (Filter.mem_principal_self support.carrier)
  refine ⟨⟨U, hcof, ?_⟩⟩
  intro A hA
  exact Filter.mem_of_superset (Filter.inter_mem hmem (hcof (hsupport A hA)))
    (fun _ hn => hn.2 hn.1)

theorem run_sound {support next : Support} {observations : List Observation}
    {commitments : Commitments} (hne : support.nonempty = true)
    (hsupport : Supports support commitments)
    (h : run support observations = some next) :
    next.nonempty = true ∧ Supports next commitments ∧
      ∀ observation ∈ observations,
        ∀ᶠ n in Filter.cofinite, n ∈ next.carrier → n ∈ observation.denote := by
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
          have htail := ih (commit_sound hc).2.1 (commit_preserves_support hsupport hc) h
          refine ⟨htail.1, ?_, ?_⟩
          · intro A hA
            exact htail.2.1 A (Set.mem_insert_of_mem _ hA)
          · intro selected hselected
            rcases List.mem_cons.mp hselected with rfl | hselected
            · exact htail.2.1 selected.denote (Set.mem_insert _ _)
            · exact htail.2.2 selected hselected

/-- One free completion satisfies every actual observation in an accepted run. -/
theorem run_trace_extendible {observations : List Observation} {next : Support}
    (h : run Periodic.Support.universe observations = some next) :
    Extendible {A | ∃ observation ∈ observations, A = observation.denote} := by
  have hs := run_sound (by rfl : Periodic.Support.universe.nonempty = true) supports_empty h
  apply extendible_of_support hs.1
  rintro A ⟨observation, hmem, rfl⟩
  exact hs.2.2 observation hmem

#print axioms Hyperreals.Laurent.compile_correct
#print axioms Hyperreals.Laurent.commit_preserves_support
#print axioms Hyperreals.Laurent.run_trace_extendible

end Hyperreals.Laurent
