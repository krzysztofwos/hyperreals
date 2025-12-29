"""Tests for completion-sensitive standard part extraction."""

from hyperreal.hyperreal import HyperrealSystem


class TestChooseStandardPartInvariant:
    """Test that choose_standard_part returns exact results for invariant cases."""

    def test_constant_returns_exact(self):
        """A constant hyperreal returns its exact value."""
        sys = HyperrealSystem()
        x = sys.constant(3.14159)
        result = x.choose_standard_part()
        assert result is not None
        assert result.low == result.high == result.approx == 3.14159
        assert result.chosen == 0  # No choices needed

    def test_one_plus_epsilon_returns_one(self):
        """1 + ε has standard part 1."""
        sys = HyperrealSystem()
        x = sys.constant(1.0) + sys.infinitesimal()
        result = x.choose_standard_part()
        assert result is not None
        assert abs(result.approx - 1.0) < 1e-10
        assert result.chosen == 0  # No choices needed

    def test_tanh_omega_returns_one(self):
        """tanh(ω) has standard part 1."""
        sys = HyperrealSystem()
        x = sys.tanh(sys.infinite())
        result = x.choose_standard_part()
        assert result is not None
        assert abs(result.approx - 1.0) < 1e-10

    def test_sin_over_log1p_returns_zero(self):
        """sin(ω) / log(1+ω) has standard part 0."""
        sys = HyperrealSystem()
        omega = sys.infinite()
        x = sys.sin(omega) / sys.log1p(omega)
        result = x.choose_standard_part()
        assert result is not None
        assert abs(result.approx) < 1e-10


class TestChooseStandardPartCompletion:
    """Test completion-sensitive extraction for bounded oscillatory sequences."""

    def test_alt_returns_some_value(self):
        """(-1)^n should return some value in [-1, 1]."""
        sys = HyperrealSystem()
        alt = sys.alt()
        result = alt.choose_standard_part(bits=10)
        assert result is not None
        assert -1.0 <= result.approx <= 1.0
        # Should have made choices since this is completion-dependent
        assert result.chosen > 0

    def test_sin_omega_returns_some_value(self):
        """sin(ω) should return some value in [-1, 1]."""
        sys = HyperrealSystem()
        x = sys.sin(sys.infinite())
        result = x.choose_standard_part(bits=10)
        assert result is not None
        assert -1.0 <= result.approx <= 1.0

    def test_tie_break_lower_vs_upper(self):
        """tie_break='lower' and 'upper' may give different results."""
        sys = HyperrealSystem()
        alt = sys.alt()

        # Note: We can't test exact values since they depend on ultrafilter state,
        # but we can verify the results are valid and within bounds.
        result_lower = alt.choose_standard_part(bits=8, tie_break="lower")
        assert result_lower is not None
        assert -1.0 <= result_lower.approx <= 1.0


class TestChooseStandardPartInfinite:
    """Test that infinite hyperreals fail to bracket."""

    def test_omega_returns_none(self):
        """ω = n is infinite, should fail to bracket."""
        sys = HyperrealSystem()
        omega = sys.infinite()
        result = omega.choose_standard_part(max_bracket_steps=10)
        assert result is None

    def test_exp_omega_returns_none(self):
        """exp(ω) is infinite, should fail to bracket."""
        sys = HyperrealSystem()
        x = sys.exp(sys.infinite())
        result = x.choose_standard_part(max_bracket_steps=10)
        assert result is None


class TestProbeAndCommit:
    """Test the probe and commit methods on PartialUltrafilter."""

    def test_probe_forced_finite(self):
        """probe should detect forced-false for finite sets."""
        from hyperreal.algebra import Atom
        from hyperreal.sequence import Const, Exp, NVar
        from hyperreal.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        # {n: exp(n) < 10} is finite
        atom = Atom("LT", Exp(NVar()), Const(10.0))
        can_be_false, can_be_true = puf.probe(atom)
        assert can_be_false is True
        assert can_be_true is False

    def test_probe_forced_cofinite(self):
        """probe should detect forced-true for cofinite sets."""
        from hyperreal.algebra import Atom
        from hyperreal.sequence import Const, Exp, NVar
        from hyperreal.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        # {n: 10 < exp(n)} is cofinite
        atom = Atom("LT", Const(10.0), Exp(NVar()))
        can_be_false, can_be_true = puf.probe(atom)
        assert can_be_false is False
        assert can_be_true is True

    def test_commit_respects_finite(self):
        """commit should fail for finite sets when trying to commit True."""
        from hyperreal.algebra import Atom
        from hyperreal.sequence import Const, Exp, NVar
        from hyperreal.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        atom = Atom("LT", Exp(NVar()), Const(10.0))
        # Should fail to commit True (finite sets must be False)
        assert puf.commit(atom, True) is False
        # Should succeed to commit False
        puf2 = PartialUltrafilter()
        assert puf2.commit(atom, False) is True

    def test_commit_respects_cofinite(self):
        """commit should fail for cofinite sets when trying to commit False."""
        from hyperreal.algebra import Atom
        from hyperreal.sequence import Const, Exp, NVar
        from hyperreal.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        atom = Atom("LT", Const(10.0), Exp(NVar()))
        # Should fail to commit False (cofinite sets must be True)
        assert puf.commit(atom, False) is False
        # Should succeed to commit True
        puf2 = PartialUltrafilter()
        assert puf2.commit(atom, True) is True
