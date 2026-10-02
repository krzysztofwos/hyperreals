import Hyperreals.Semantics
import Mathlib.Order.Filter.Finite
import Mathlib.Order.Filter.Ultrafilter.Basic

/-!
# Free-ultrafilter completions of finite observations

The main invariant is semantic: the committed index sets, together with every
cofinite set, must have the finite-intersection property. Boolean satisfiability
of opaque names is not a substitute for this condition.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Set

namespace Hyperreals

/-- A family of committed-true index sets. -/
abbrev Commitments := Set (Set ℕ)

/-- Commitments augmented with the Frechet (cofinite) filter basis. -/
def completionBasis (Γ : Commitments) : Set (Set ℕ) :=
  Γ ∪ {A | Aᶜ.Finite}

/-- Every finite subfamily of the commitments and cofinite sets intersects. -/
def HasFreeFIP (Γ : Commitments) : Prop :=
  ∀ T : Finset (Set ℕ),
    (↑T : Set (Set ℕ)) ⊆ completionBasis Γ →
      (⋂₀ (↑T : Set (Set ℕ))).Nonempty

/-- A classical free-ultrafilter completion containing all commitments. -/
structure Completion (Γ : Commitments) where
  ultrafilter : Ultrafilter ℕ
  extendsCofinite : (ultrafilter : Filter ℕ) ≤ Filter.cofinite
  contains : ∀ ⦃A : Set ℕ⦄, A ∈ Γ → A ∈ ultrafilter

/-- The commitments are extendible when at least one completion exists. -/
def Extendible (Γ : Commitments) : Prop := Nonempty (Completion Γ)

/-- The semantic finite-intersection condition implies existence of a free completion.

The result deliberately lives in `Prop`: extracting a particular ultrafilter
would require classical choice and would not produce executable data.
-/
theorem extendible_of_hasFreeFIP {Γ : Commitments} (hΓ : HasFreeFIP Γ) :
    Extendible Γ := by
  rcases Ultrafilter.exists_ultrafilter_of_finite_inter_nonempty
      (completionBasis Γ) hΓ with ⟨U, hU⟩
  refine ⟨⟨U, ?_, ?_⟩⟩
  · intro A hA
    exact hU (Or.inr hA)
  · intro A hA
    exact hU (Or.inl hA)

/-- Any free completion witnesses the semantic finite-intersection condition. -/
theorem hasFreeFIP_of_completion {Γ : Commitments} (C : Completion Γ) :
    HasFreeFIP Γ := by
  intro T hT
  have hmem : ⋂₀ (↑T : Set (Set ℕ)) ∈ (C.ultrafilter : Filter ℕ) := by
    rw [Filter.sInter_mem T.finite_toSet]
    intro A hAT
    rcases hT hAT with hAΓ | hAcof
    · exact C.contains hAΓ
    · exact C.extendsCofinite hAcof
  exact Filter.nonempty_of_mem hmem

/-- Completion exists exactly when the semantic free-FIP obligation holds. -/
theorem hasFreeFIP_iff_extendible {Γ : Commitments} :
    HasFreeFIP Γ ↔ Extendible Γ := by
  constructor
  · exact extendible_of_hasFreeFIP
  · rintro ⟨C⟩
    exact hasFreeFIP_of_completion C

/-- A finite runtime state has one joint semantic intersection. -/
def JointlyInfinite (Γ : Finset (Set ℕ)) : Prop :=
  (⋂₀ (↑Γ : Set (Set ℕ))).Infinite

/-- No finite set can belong to a completion extending the cofinite filter. -/
theorem Completion.notMem_of_finite {Γ : Commitments} (C : Completion Γ)
    {A : Set ℕ} (hA : A.Finite) : A ∉ C.ultrafilter := by
  intro hmem
  have hcompl : Aᶜ ∈ C.ultrafilter :=
    C.extendsCofinite hA.compl_mem_cofinite
  have hinter : A ∩ Aᶜ ∈ C.ultrafilter := Filter.inter_mem hmem hcompl
  rw [Set.inter_compl_self] at hinter
  exact C.ultrafilter.empty_notMem hinter

/-- An infinite joint intersection constructs a completion of a finite state. -/
theorem extendible_of_jointlyInfinite {Γ : Finset (Set ℕ)}
    (hΓ : JointlyInfinite Γ) : Extendible (↑Γ : Commitments) := by
  let I : Set ℕ := ⋂₀ (↑Γ : Set (Set ℕ))
  have hNeBot : NeBot (Filter.cofinite ⊓ Filter.principal I) :=
    hΓ.cofinite_inf_principal_neBot
  rcases Filter.exists_ultrafilter_iff.mpr hNeBot with ⟨U, hU⟩
  refine ⟨⟨U, hU.trans inf_le_left, ?_⟩⟩
  intro A hA
  have hI_mem : I ∈ U :=
    (hU.trans inf_le_right) (Filter.mem_principal_self I)
  exact Filter.mem_of_superset hI_mem (Set.sInter_subset_of_mem hA)

/-- Every completion of a finite state forces its joint intersection to be infinite. -/
theorem jointlyInfinite_of_extendible {Γ : Finset (Set ℕ)}
    (hΓ : Extendible (↑Γ : Commitments)) : JointlyInfinite Γ := by
  rcases hΓ with ⟨C⟩
  have hmem : ⋂₀ (↑Γ : Set (Set ℕ)) ∈ (C.ultrafilter : Filter ℕ) := by
    rw [Filter.sInter_mem Γ.finite_toSet]
    intro A hA
    exact C.contains hA
  intro hfinite
  exact C.notMem_of_finite hfinite hmem

/-- For finite runtime states, extendibility is exactly infinitude of the joint intersection. -/
theorem jointlyInfinite_iff_extendible {Γ : Finset (Set ℕ)} :
    JointlyInfinite Γ ↔ Extendible (↑Γ : Commitments) :=
  ⟨extendible_of_jointlyInfinite, jointlyInfinite_of_extendible⟩

/-- Adding a set already selected by a completion preserves that completion. -/
def Completion.insertOfMem {Γ : Commitments} (C : Completion Γ) (A : Set ℕ)
    (hA : A ∈ C.ultrafilter) : Completion (insert A Γ) where
  ultrafilter := C.ultrafilter
  extendsCofinite := C.extendsCofinite
  contains := by
    intro B hB
    rcases hB with rfl | hBΓ
    · exact hA
    · exact C.contains hBΓ

/-- Classically, at least one polarity of every new query preserves extendibility. -/
theorem extendible_insert_or_compl {Γ : Commitments} (hΓ : Extendible Γ)
    (A : Set ℕ) :
    Extendible (insert A Γ) ∨ Extendible (insert Aᶜ Γ) := by
  rcases hΓ with ⟨C⟩
  rcases C.ultrafilter.mem_or_compl_mem A with hA | hAc
  · exact Or.inl ⟨C.insertOfMem A hA⟩
  · exact Or.inr ⟨C.insertOfMem Aᶜ hAc⟩

/-- A backend may commit `A` only by discharging this semantic obligation. -/
def SafePositiveChoice (Γ : Commitments) (A : Set ℕ) : Prop :=
  HasFreeFIP (insert A Γ)

/-- A certified positive choice has a classical free-ultrafilter completion. -/
theorem safePositiveChoice_extendible {Γ : Commitments} {A : Set ℕ}
    (h : SafePositiveChoice Γ A) : Extendible (insert A Γ) :=
  hasFreeFIP_iff_extendible.mp h

#print axioms Hyperreals.extendible_of_hasFreeFIP
#print axioms Hyperreals.hasFreeFIP_iff_extendible
#print axioms Hyperreals.jointlyInfinite_iff_extendible
#print axioms Hyperreals.extendible_insert_or_compl
#print axioms Hyperreals.safePositiveChoice_extendible

end Hyperreals
