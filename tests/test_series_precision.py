"""Regressions for precision shifts and conservative standard-part analysis."""

import math
from functools import reduce
from operator import mul

import pytest

from hyperreals import HyperrealSystem
from hyperreals import asymptotic_facts
from hyperreals.sequence import InvN
from hyperreals.series import series_from_seq


@pytest.fixture(autouse=True)
def clear_analysis_cache():
    asymptotic_facts.clear_cache()
    yield
    asymptotic_facts.clear_cache()


def test_zero_limit_quotient_has_no_positive_lower_bound():
    system = HyperrealSystem()
    epsilon = system.infinitesimal()
    denominator = epsilon / system.tanh(system.infinite())

    fact = asymptotic_facts.analyze(denominator.seq)
    assert fact.limit == 0.0
    assert fact.abs_lower is None
    # The quotient equals tanh(n). Until cancellation is recognized, it must
    # remain unknown rather than be assigned the false standard part zero.
    assert (epsilon / denominator).standard_part() is None


def test_nonzero_quotient_lower_bound_uses_its_own_limit():
    system = HyperrealSystem()
    quotient = system.constant(0.25) / system.tanh(system.infinite())
    fact = asymptotic_facts.analyze(quotient.seq)
    assert fact.limit == 0.25
    assert fact.abs_lower == 0.125


@pytest.mark.parametrize("operation", ["multiply", "divide"])
def test_laurent_shifts_preserve_high_order_terms(operation):
    system = HyperrealSystem()
    epsilon_power = reduce(mul, [system.infinitesimal()] * 11)
    if operation == "multiply":
        inverse_power = reduce(mul, [system.infinite()] * 11)
        result = epsilon_power * inverse_power
    else:
        result = epsilon_power / epsilon_power

    assert result.standard_part() == 1.0
    assert result.coeff(0, order=0) == 1.0
    assert result == system.constant(1.0)


def test_shifted_analytic_series_requests_sufficient_child_precision():
    system = HyperrealSystem()
    epsilon_power = reduce(mul, [system.infinitesimal()] * 11)
    inverse_power = reduce(mul, [system.infinite()] * 11)
    result = system.sin(epsilon_power) * inverse_power
    assert result.standard_part() == 1.0
    assert result.coeff(0, order=0) == 1.0


def test_division_shift_preserves_first_uncancelled_taylor_term():
    system = HyperrealSystem()
    epsilon = system.infinitesimal()
    result = (system.sin(epsilon) - epsilon) / (epsilon * epsilon * epsilon)
    assert result.coeff(0, order=0) == pytest.approx(-1.0 / 6.0)
    assert result.standard_part() == pytest.approx(-1.0 / 6.0)


def test_truncated_denominator_is_not_treated_as_exact_monomial():
    system = HyperrealSystem()
    epsilon = system.infinitesimal()
    n = system.infinite()
    denominator = epsilon + reduce(mul, [epsilon] * 11)
    result = reduce(mul, [n] * 9) / denominator - reduce(mul, [n] * 10)
    # Its true limit is -1. A non-monomial inverse is currently unsupported,
    # so dropping the denominator's delta^11 term must not fabricate zero.
    assert result.series() is None
    assert result.standard_part() is None


def test_rounded_cancellation_cannot_certify_a_monomial_denominator():
    system = HyperrealSystem()
    epsilon = system.infinitesimal()
    n = system.infinite()
    large = system.constant(float(2**53))
    coefficient = (large + system.constant(1.0)) - large
    denominator = epsilon + coefficient * epsilon * epsilon
    result = n / denominator - n * n
    # The exact coefficient is one, although float coefficient arithmetic
    # rounds it to zero. The result diverges and has no finite standard part.
    assert result.series() is None
    assert result.standard_part() is None


@pytest.mark.parametrize("function,constant", [("tanh", 1000.0), ("log1p", 1e100)])
def test_overflowing_series_falls_back_to_finite_limit(function, constant):
    system = HyperrealSystem()
    argument = system.constant(constant) + system.infinitesimal()
    result = getattr(system, function)(argument)

    assert result.series() is None
    assert result.standard_part() == getattr(math, function)(constant)


def test_unrepresentable_expansion_returns_unknown_without_raising():
    system = HyperrealSystem()
    result = system.exp(system.constant(1000.0) + system.infinitesimal())
    assert result.series() is None
    assert result.standard_part() is None


def test_analysis_cache_honors_requested_order(monkeypatch):
    original = asymptotic_facts._analyze_impl
    requested_orders = []

    def record_analysis(sequence, *, order):
        requested_orders.append(order)
        return original(sequence, order=order)

    monkeypatch.setattr(asymptotic_facts, "_analyze_impl", record_analysis)
    sequence = InvN()
    asymptotic_facts.analyze(sequence, order=2)
    asymptotic_facts.analyze(sequence, order=6)
    asymptotic_facts.analyze(sequence, order=2)
    assert requested_orders == [2, 6]


def test_negative_precision_cannot_discard_the_constant_term():
    system = HyperrealSystem()
    sequence = system.constant(1.0).seq
    with pytest.raises(ValueError, match="nonnegative"):
        asymptotic_facts.analyze(sequence, order=-1)
    with pytest.raises(ValueError, match="nonnegative"):
        series_from_seq(sequence, order=-1)
