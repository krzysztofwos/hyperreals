import Hyperreals.ResidueLimitDiagnosticCore
import Hyperreals.ResidueLimitCompleteness

/-!
# Sound and exhaustive explanations of extraction

A divergence report names an active residue whose absolute value tends to
infinity. A disagreement report names two active residues with distinct exact
finite limits. Invalid-input reports occur exactly for invalid expressions or
empty support. The diagnostic is checked against the same exact normalization
used by extraction.
-/

set_option autoImplicit false
set_option relaxedAutoImplicit false

open Filter Topology

namespace Hyperreals.Residue

private def ListDiagnosticValid (values : List (Nat × Option Rat)) : LimitDiagnostic → Prop
  | .finite q => commonValue (values.map Prod.snd) = some q
  | .divergent residue => (residue, none) ∈ values
  | .disagreement r q s v => (r, some q) ∈ values ∧ (s, some v) ∈ values ∧ q ≠ v
  | .invalidInput => values = []

private theorem diagnoseLimits_valid (values : List (Nat × Option Rat)) :
    ListDiagnosticValid values (diagnoseLimits values) := by
  cases hfind : values.find? (fun entry => entry.2.isNone) with
  | some entry =>
      obtain ⟨residue, value⟩ := entry
      have hnone : value = none := by simpa using List.find?_some hfind
      subst value
      simpa [diagnoseLimits, hfind, ListDiagnosticValid] using List.mem_of_find?_eq_some hfind
  | none =>
      have hsome := List.find?_eq_none.mp hfind
      cases values with
      | nil => simp [diagnoseLimits, ListDiagnosticValid]
      | cons entry rest =>
          obtain ⟨residue, value⟩ := entry
          cases value with
          | none =>
              have := hsome (residue, none) (by simp)
              simp at this
          | some value =>
              cases hother : rest.find? (fun entry => decide (entry.2 ≠ some value)) with
              | none =>
                  simp only [diagnoseLimits, hfind, hother, ListDiagnosticValid]
                  apply commonValue_complete
                  · simp
                  · intro other hmem
                    obtain ⟨entry, hentry, rfl⟩ := List.mem_map.mp hmem
                    rcases List.mem_cons.mp hentry with rfl | hrest
                    · rfl
                    · have := List.find?_eq_none.mp hother entry hrest
                      simpa using this
              | some entry =>
                  obtain ⟨other, otherValue⟩ := entry
                  have hmem := List.mem_of_find?_eq_some hother
                  cases otherValue with
                  | none =>
                      have := hsome (other, none) (by simp [hmem])
                      simp at this
                  | some otherValue =>
                      have hne : value ≠ otherValue := by
                        have := List.find?_some hother
                        simpa [ne_comm] using this
                      simp only [diagnoseLimits, hfind, hother, ListDiagnosticValid]
                      exact ⟨by simp, by simp [hmem], hne⟩

/-- Membership in the diagnostic's input records exactly an active normalized
branch of the common period. -/
theorem mem_activeLimits_iff {support : Support} {expression : Expr}
    {residue : Nat} {value : Option Rat} :
    (residue, value) ∈ activeLimits support expression ↔
      residue < Nat.lcm support.length expression.period ∧ support.at residue = true ∧
        (expression.normalizeAt residue).standardPart? = value := by
  simp only [activeLimits, List.mem_map, List.mem_filter, List.mem_range, Prod.mk.injEq]
  constructor
  · rintro ⟨r, ⟨hr, ha⟩, heq, hvalue⟩
    subst r
    exact ⟨hr, ha, hvalue⟩
  · rintro ⟨hr, ha, hvalue⟩
    exact ⟨residue, ⟨hr, ha⟩, rfl, hvalue⟩

private theorem activeLimits_nonempty {support : Support} {expression : Expr}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true) :
    activeLimits support expression ≠ [] := by
  obtain ⟨residue, hr, ha⟩ := exists_active_residue hvalid hnonempty
  have hmem := mem_activeLimits_iff.mpr ⟨hr, ha, rfl⟩
  intro hnil
  rw [hnil] at hmem
  contradiction

/-- Invalid-input reports characterize precisely the guard rejected by the
extractor. No mathematically meaningful failure is hidden in this case. -/
theorem diagnoseStandardPart_invalid_iff {support : Support} {expression : Expr} :
    diagnoseStandardPart support expression = .invalidInput ↔
      expression.valid = false ∨ support.nonempty = false := by
  by_cases hguard : (expression.valid && support.nonempty) = true
  · have hparts : expression.valid = true ∧ support.nonempty = true := by
      simpa only [Bool.and_eq_true] using hguard
    have hnonempty := activeLimits_nonempty hparts.1 hparts.2
    have hdiag := diagnoseLimits_valid (activeLimits support expression)
    constructor
    · intro h
      simp only [diagnoseStandardPart, hguard, ↓reduceIte] at h
      rw [h] at hdiag
      exact False.elim (hnonempty hdiag)
    · intro h
      rcases h with h | h <;> simp_all
  · simp only [diagnoseStandardPart, hguard]
    cases hv : expression.valid <;> cases hs : support.nonempty <;> simp_all

/-- A finite diagnostic agrees exactly with the original extractor. -/
theorem diagnoseStandardPart_finite_iff {support : Support} {expression : Expr} {q : Rat} :
    diagnoseStandardPart support expression = .finite q ↔
      standardPart support expression = some q := by
  by_cases hguard : (expression.valid && support.nonempty) = true
  · have hparts : expression.valid = true ∧ support.nonempty = true := by
      simpa only [Bool.and_eq_true] using hguard
    have hdiag := diagnoseLimits_valid (activeLimits support expression)
    have hmap : (activeLimits support expression).map Prod.snd =
        ((List.range (Nat.lcm support.length expression.period)).filter
          (fun residue => support.at residue)).map
          (fun residue => (expression.normalizeAt residue).standardPart?) := by
      simp [activeLimits, List.map_map]
    constructor
    · intro h
      simp only [diagnoseStandardPart, hguard, ↓reduceIte] at h
      rw [h] at hdiag
      simpa only [ListDiagnosticValid, hmap, standardPart, hguard, ↓reduceIte] using hdiag
    · intro h
      have hsuccess := standardPart_success h
      cases hd : diagnoseLimits (activeLimits support expression) with
      | finite value =>
          rw [hd] at hdiag
          have hv : standardPart support expression = some value := by
            simpa only [ListDiagnosticValid, hmap, standardPart, hguard, ↓reduceIte] using hdiag
          have : value = q := Option.some.inj (hv.symm.trans h)
          simp [diagnoseStandardPart, hguard, hd, this]
      | divergent residue =>
          rw [hd] at hdiag
          obtain ⟨hr, ha, hn⟩ := mem_activeLimits_iff.mp hdiag
          have hv := hsuccess.2.2 residue hr ha
          rw [hn] at hv
          contradiction
      | disagreement r rv s sv =>
          rw [hd] at hdiag
          obtain ⟨hr, hs, hne⟩ := hdiag
          obtain ⟨hrlt, hra, hrv⟩ := mem_activeLimits_iff.mp hr
          obtain ⟨hslt, hsa, hsv⟩ := mem_activeLimits_iff.mp hs
          have hrq := Option.some.inj (hrv.symm.trans (hsuccess.2.2 r hrlt hra))
          have hsq := Option.some.inj (hsv.symm.trans (hsuccess.2.2 s hslt hsa))
          exact False.elim (hne (hrq.trans hsq.symm))
      | invalidInput =>
          rw [hd] at hdiag
          exact False.elim (activeLimits_nonempty hparts.1 hparts.2 hdiag)
  · simp [diagnoseStandardPart, standardPart, hguard]

/-- Divergence diagnostics select a valid, active, rejected Laurent branch. -/
theorem diagnoseStandardPart_divergent {support : Support} {expression : Expr} {residue : Nat}
    (hresult : diagnoseStandardPart support expression = .divergent residue) :
    expression.valid = true ∧ support.nonempty = true ∧
      residue < Nat.lcm support.length expression.period ∧ support.at residue = true ∧
        (expression.normalizeAt residue).standardPart? = none := by
  simp only [diagnoseStandardPart] at hresult
  split at hresult
  next hguard =>
    have hdiag := diagnoseLimits_valid (activeLimits support expression)
    rw [hresult] at hdiag
    have hparts : expression.valid = true ∧ support.nonempty = true := by
      simpa only [Bool.and_eq_true] using hguard
    exact ⟨hparts.1, hparts.2,
      mem_activeLimits_iff.mp hdiag⟩
  next => contradiction

/-- Disagreement diagnostics return two active branches with distinct exact
finite limits. -/
theorem diagnoseStandardPart_disagreement {support : Support} {expression : Expr}
    {r s : Nat} {q v : Rat}
    (hresult : diagnoseStandardPart support expression = .disagreement r q s v) :
    expression.valid = true ∧ support.nonempty = true ∧
      (r < Nat.lcm support.length expression.period ∧ support.at r = true ∧
        (expression.normalizeAt r).standardPart? = some q) ∧
      (s < Nat.lcm support.length expression.period ∧ support.at s = true ∧
        (expression.normalizeAt s).standardPart? = some v) ∧ q ≠ v := by
  simp only [diagnoseStandardPart] at hresult
  split at hresult
  next hguard =>
    have hdiag := diagnoseLimits_valid (activeLimits support expression)
    rw [hresult] at hdiag
    have hparts : expression.valid = true ∧ support.nonempty = true := by
      simpa only [Bool.and_eq_true] using hguard
    exact ⟨hparts.1, hparts.2,
      mem_activeLimits_iff.mp hdiag.1, mem_activeLimits_iff.mp hdiag.2.1, hdiag.2.2⟩
  next => contradiction

/-- For valid expressions on nonempty support, rejection always has one of
the two mathematically meaningful, executable explanations. -/
theorem standardPart_none_iff_diagnostic_failure {support : Support} {expression : Expr}
    (hvalid : expression.valid = true) (hnonempty : support.nonempty = true) :
    standardPart support expression = none ↔
      (∃ residue, diagnoseStandardPart support expression = .divergent residue) ∨
      ∃ r q s v, diagnoseStandardPart support expression = .disagreement r q s v := by
  cases hd : diagnoseStandardPart support expression with
  | finite q =>
      have hsome := diagnoseStandardPart_finite_iff.mp hd
      simp [hsome]
  | divergent residue =>
      have hnone : standardPart support expression = none := by
        cases hs : standardPart support expression with
        | none => rfl
        | some q =>
            have hfinite := diagnoseStandardPart_finite_iff.mpr hs
            rw [hd] at hfinite
            contradiction
      simp [hnone]
  | disagreement r q s v =>
      have hnone : standardPart support expression = none := by
        cases hs : standardPart support expression with
        | none => rfl
        | some value =>
            have hfinite := diagnoseStandardPart_finite_iff.mpr hs
            rw [hd] at hfinite
            contradiction
      simp [hnone]
  | invalidInput =>
      have hinvalid := diagnoseStandardPart_invalid_iff.mp hd
      simp [hvalid, hnonempty] at hinvalid

noncomputable section

/-- The branch named by a divergence report actually escapes every bounded
interval. It is not merely a failure to recognize a limit. -/
theorem diagnoseStandardPart_divergent_sound {support : Support} {expression : Expr} {residue : Nat}
    (hresult : diagnoseStandardPart support expression = .divergent residue) :
    support.at residue = true ∧
      Tendsto (fun n : ℕ => |(expression.normalizeAt residue).eval n|) Filter.cofinite atTop := by
  have h := diagnoseStandardPart_divergent hresult
  exact ⟨h.2.2.2.1,
    (expression.normalizeAt residue).abs_tendsto_atTop_of_standardPart?_none h.2.2.2.2⟩

/-- The two branches named by a disagreement report have the displayed finite
limits, and those limits differ. -/
theorem diagnoseStandardPart_disagreement_sound {support : Support} {expression : Expr}
    {r s : Nat} {q v : Rat}
    (hresult : diagnoseStandardPart support expression = .disagreement r q s v) :
    Tendsto (fun n : ℕ => (expression.normalizeAt r).eval n) Filter.cofinite (𝓝 (q : ℝ)) ∧
      Tendsto (fun n : ℕ => (expression.normalizeAt s).eval n) Filter.cofinite (𝓝 (v : ℝ)) ∧
        (q : ℝ) ≠ (v : ℝ) := by
  have h := diagnoseStandardPart_disagreement hresult
  refine ⟨(expression.normalizeAt r).standardPart?_cofinite q h.2.2.1.2.2,
    (expression.normalizeAt s).standardPart?_cofinite v h.2.2.2.1.2.2, ?_⟩
  exact_mod_cast h.2.2.2.2

/-- A divergence report has a compatible completion in which the source
expression itself grows without bound in absolute value. -/
theorem diagnoseStandardPart_divergent_completion {support : Support} {expression : Expr}
    {residue : Nat}
    (hresult : diagnoseStandardPart support expression = .divergent residue) :
    ∃ U : Ultrafilter ℕ, (U : Filter ℕ) ≤ Filter.cofinite ∧ support.carrier ∈ U ∧
      Tendsto (fun n => |expression.denote n|) (U : Filter ℕ) atTop := by
  have h := diagnoseStandardPart_divergent hresult
  obtain ⟨U, hfree, hs, heq⟩ :=
    exists_compatible_completion_on_residue h.1 h.2.1 h.2.2.1 h.2.2.2.1
  refine ⟨U, hfree, hs, ?_⟩
  apply ((diagnoseStandardPart_divergent_sound hresult).2.mono_left hfree).congr'
  filter_upwards [heq] with n hn
  exact congrArg abs hn.symm

/-- A disagreement report supplies two compatible completions with the two
reported, distinct standard parts of the source expression. -/
theorem diagnoseStandardPart_disagreement_completions {support : Support} {expression : Expr}
    {r s : Nat} {q v : Rat}
    (hresult : diagnoseStandardPart support expression = .disagreement r q s v) :
    ∃ U V : Ultrafilter ℕ,
      ((U : Filter ℕ) ≤ Filter.cofinite ∧ support.carrier ∈ U ∧
        NearStandardAt U expression.denote (q : ℝ)) ∧
      ((V : Filter ℕ) ≤ Filter.cofinite ∧ support.carrier ∈ V ∧
        NearStandardAt V expression.denote (v : ℝ)) ∧ (q : ℝ) ≠ (v : ℝ) := by
  have h := diagnoseStandardPart_disagreement hresult
  have hlimits := diagnoseStandardPart_disagreement_sound hresult
  obtain ⟨U, hUfree, hUs, hUeq⟩ :=
    exists_compatible_completion_on_residue h.1 h.2.1 h.2.2.1.1 h.2.2.1.2.1
  obtain ⟨V, hVfree, hVs, hVeq⟩ :=
    exists_compatible_completion_on_residue h.1 h.2.1 h.2.2.2.1.1 h.2.2.2.1.2.1
  exact ⟨U, V, ⟨hUfree, hUs, (hlimits.1.mono_left hUfree).congr' hUeq.symm⟩,
    ⟨hVfree, hVs, (hlimits.2.1.mono_left hVfree).congr' hVeq.symm⟩, hlimits.2.2⟩

#print axioms Hyperreals.Residue.diagnoseStandardPart_divergent_completion
#print axioms Hyperreals.Residue.diagnoseStandardPart_disagreement_completions
#print axioms Hyperreals.Residue.standardPart_none_iff_diagnostic_failure
#print axioms Hyperreals.Residue.diagnoseStandardPart_invalid_iff
#print axioms Hyperreals.Residue.diagnoseStandardPart_finite_iff
#print axioms Hyperreals.Residue.diagnoseStandardPart_divergent
#print axioms Hyperreals.Residue.diagnoseStandardPart_disagreement
#print axioms Hyperreals.Residue.diagnoseStandardPart_divergent_sound
#print axioms Hyperreals.Residue.diagnoseStandardPart_disagreement_sound

end

end Hyperreals.Residue
