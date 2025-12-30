"""Tests for the Tier-2 completion-invariant asymptotic analyzer."""

import math

import pytest

from hyperreals.asymptotic import standard_part_extended
from hyperreals.asymptotic_facts import (
    analyze,
    clear_cache,
)
from hyperreals.sequence import (
    Add,
    AltSign,
    Const,
    Cos,
    Cosh,
    Div,
    Exp,
    InvN,
    Log1p,
    Mul,
    NVar,
    Sin,
    Sinh,
    Sqrt1p,
    Sub,
    Tanh,
)


@pytest.fixture(autouse=True)
def clear_analysis_cache():
    """Clear the analysis cache before each test."""
    clear_cache()
    yield
    clear_cache()


class TestBasePrimitives:
    """Test analysis of base primitives."""

    def test_const(self):
        fact = analyze(Const(3.5))
        assert fact.kind == "finite"
        assert fact.limit == 3.5

    def test_const_zero(self):
        fact = analyze(Const(0.0))
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_invn(self):
        fact = analyze(InvN())
        assert fact.kind == "finite"
        assert fact.limit == 0.0
        # abs_bound may be 0.0 (from limit) or 1.0 (from direct analysis)
        # Both are valid upper bounds for eventually

    def test_nvar(self):
        fact = analyze(NVar())
        assert fact.kind == "plus_inf"
        assert fact.sign == 1

    def test_altsign(self):
        fact = analyze(AltSign())
        assert fact.kind == "unknown"
        assert fact.abs_bound == 1.0


class TestAlgebraicOperations:
    """Test analysis of algebraic operations."""

    def test_add_finite_finite(self):
        seq = Add(Const(2.0), Const(3.0))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 5.0

    def test_sub_finite_finite(self):
        seq = Sub(Const(5.0), Const(3.0))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 2.0

    def test_mul_finite_finite(self):
        seq = Mul(Const(2.0), Const(3.0))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 6.0

    def test_div_finite_nonzero(self):
        seq = Div(Const(6.0), Const(2.0))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 3.0

    def test_add_plus_inf_finite(self):
        seq = Add(NVar(), Const(5.0))
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_sub_finite_plus_inf(self):
        seq = Sub(Const(5.0), NVar())
        fact = analyze(seq)
        assert fact.kind == "minus_inf"


class TestBoundedVanishingClosure:
    """Test BV closure: bounded × infinitesimal → 0."""

    def test_zero_times_bounded(self):
        # 0 * bounded = 0
        seq = Mul(Const(0.0), AltSign())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_infinitesimal_times_bounded(self):
        # (1/n) * bounded = 0
        seq = Mul(InvN(), AltSign())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_bounded_times_infinitesimal(self):
        # bounded * (1/n) = 0
        seq = Mul(Sin(NVar()), InvN())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0


class TestBoundedDividingDiverging:
    """Test bounded / diverging → 0 (key Tier-2 rule)."""

    def test_bounded_over_nvar(self):
        # 1 / n -> 0
        seq = Div(Const(1.0), NVar())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_sin_over_log1p(self):
        # sin(n) / log(1+n) -> 0
        seq = Div(Sin(NVar()), Log1p(NVar()))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_one_over_exp(self):
        # 1 / exp(n) -> 0
        seq = Div(Const(1.0), Exp(NVar()))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_bounded_over_exp(self):
        # sin(n) / exp(n) -> 0
        seq = Div(Sin(NVar()), Exp(NVar()))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0


class TestMonotoneDivergence:
    """Test monotone divergence propagation through functions."""

    def test_exp_plus_inf(self):
        # exp(n) -> +inf
        seq = Exp(NVar())
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_exp_minus_inf(self):
        # exp(-n) -> 0
        seq = Exp(Sub(Const(0.0), NVar()))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_log1p_plus_inf(self):
        # log(1+n) -> +inf
        seq = Log1p(NVar())
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_sqrt1p_plus_inf(self):
        # sqrt(1+n) -> +inf
        seq = Sqrt1p(NVar())
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_cosh_plus_inf(self):
        # cosh(n) -> +inf
        seq = Cosh(NVar())
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_cosh_minus_inf(self):
        # cosh(-n) -> +inf (even function)
        seq = Cosh(Sub(Const(0.0), NVar()))
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_sinh_plus_inf(self):
        # sinh(n) -> +inf
        seq = Sinh(NVar())
        fact = analyze(seq)
        assert fact.kind == "plus_inf"

    def test_sinh_minus_inf(self):
        # sinh(-n) -> -inf (odd function)
        seq = Sinh(Sub(Const(0.0), NVar()))
        fact = analyze(seq)
        assert fact.kind == "minus_inf"


class TestMonotoneLimits:
    """Test monotone limits at infinity for bounded functions."""

    def test_tanh_plus_inf(self):
        # tanh(n) -> 1
        seq = Tanh(NVar())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 1.0

    def test_tanh_minus_inf(self):
        # tanh(-n) -> -1
        seq = Tanh(Sub(Const(0.0), NVar()))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == -1.0

    def test_tanh_log1p(self):
        # tanh(log(1+n)) -> 1
        seq = Tanh(Log1p(NVar()))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 1.0

    def test_exp_neg_log1p(self):
        # exp(-log(1+n)) = 1/(1+n) -> 0
        seq = Exp(Sub(Const(0.0), Log1p(NVar())))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0


class TestStandardPartExtended:
    """Test the standard_part_extended function with Tier-2 capabilities."""

    def test_sin_over_log1p(self):
        # sin(n) / log(1+n) -> 0
        seq = Div(Sin(NVar()), Log1p(NVar()))
        st = standard_part_extended(seq)
        assert st == 0.0

    def test_tanh_log1p(self):
        # tanh(log(1+n)) -> 1
        seq = Tanh(Log1p(NVar()))
        st = standard_part_extended(seq)
        assert st == 1.0

    def test_exp_neg_log1p(self):
        # exp(-log(1+n)) -> 0
        seq = Exp(Sub(Const(0.0), Log1p(NVar())))
        st = standard_part_extended(seq)
        assert st == 0.0

    def test_one_over_exp(self):
        # 1 / exp(n) -> 0
        seq = Div(Const(1.0), Exp(NVar()))
        st = standard_part_extended(seq)
        assert st == 0.0


class TestOscillationsRemainUnknown:
    """Regression: oscillatory sequences should have no standard part."""

    def test_sin_n_no_standard_part(self):
        seq = Sin(NVar())
        st = standard_part_extended(seq)
        assert st is None

    def test_alt_sign_no_standard_part(self):
        seq = AltSign()
        st = standard_part_extended(seq)
        assert st is None

    def test_cos_n_no_standard_part(self):
        seq = Cos(NVar())
        st = standard_part_extended(seq)
        assert st is None


class TestUltrafilterIntegration:
    """Test that Tier-2 analysis improves ultrafilter finite/cofinite detection."""

    def test_exp_n_less_than_const_is_finite(self):
        """exp(n) < 10 should be detected as finite (eventually false)."""
        from hyperreals.algebra import Atom
        from hyperreals.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        # {n: exp(n) < 10} is finite since exp(n) -> +inf
        atom = Atom("LT", Exp(NVar()), Const(10.0))
        assert puf._is_finite(atom) is True
        assert puf._is_cofinite(atom) is False

    def test_log1p_n_less_than_const_is_finite(self):
        """log(1+n) < 5 should be detected as finite (eventually false)."""
        from hyperreals.algebra import Atom
        from hyperreals.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        # {n: log(1+n) < 5} is finite since log(1+n) -> +inf
        atom = Atom("LT", Log1p(NVar()), Const(5.0))
        assert puf._is_finite(atom) is True

    def test_const_less_than_exp_n_is_cofinite(self):
        """10 < exp(n) should be detected as cofinite (eventually true)."""
        from hyperreals.algebra import Atom
        from hyperreals.ultrafilter import PartialUltrafilter

        puf = PartialUltrafilter()
        # {n: 10 < exp(n)} is cofinite since exp(n) -> +inf
        atom = Atom("LT", Const(10.0), Exp(NVar()))
        assert puf._is_cofinite(atom) is True
        assert puf._is_finite(atom) is False


class TestCompositions:
    """Test compositions of functions."""

    def test_tanh_tanh(self):
        # tanh(tanh(x)) at x=0.5 -> tanh(tanh(0.5))
        seq = Tanh(Tanh(Const(0.5)))
        fact = analyze(seq)
        assert fact.kind == "finite"
        expected = math.tanh(math.tanh(0.5))
        assert abs(fact.limit - expected) < 1e-10

    def test_exp_of_negative_constant(self):
        seq = Exp(Const(-5.0))
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert abs(fact.limit - math.exp(-5.0)) < 1e-10

    def test_log1p_of_invn(self):
        # log(1 + 1/n) -> 0
        seq = Log1p(InvN())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 0.0

    def test_sqrt1p_of_invn(self):
        # sqrt(1 + 1/n) -> 1
        seq = Sqrt1p(InvN())
        fact = analyze(seq)
        assert fact.kind == "finite"
        assert fact.limit == 1.0
