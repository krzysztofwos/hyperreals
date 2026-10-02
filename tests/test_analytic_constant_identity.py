"""Rounded analytic evaluations must not become exact symbolic constants."""

import math

import pytest

from hyperreals import HyperrealSystem
from hyperreals.sequence import Const, Cos, Cosh, Exp, Log1p, Sin, Sinh, Sqrt1p, Tan, Tanh


def test_positive_exponential_is_not_erased_by_float_underflow():
    system = HyperrealSystem()
    exponential = system.exp(system.constant(-1000))
    zero = system.constant(0)

    assert isinstance(exponential.seq, Exp)
    assert zero < exponential
    assert not (exponential < zero)
    assert exponential.compare_eq(zero) is None


def test_rounded_sine_value_is_not_an_exact_comparison_certificate():
    system = HyperrealSystem()
    sine = system.sin(system.constant(1))
    rounded = system.constant(math.sin(1))

    assert isinstance(sine.seq, Sin)
    assert sine.compare_eq(rounded) is None
    assert sine.compare_lt(rounded) is None


@pytest.mark.parametrize(
    ("constructor", "expected"),
    [(Sin, 0), (Cos, 1), (Tan, 0), (Tanh, 0), (Exp, 1), (Log1p, 0), (Sqrt1p, 1),
     (Cosh, 1), (Sinh, 0)],
)
def test_analytic_zero_identities_remain_exact(constructor, expected):
    assert constructor(Const(0)).simplify().key() == Const(expected).key()


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        ("sin", math.sin(0.5)),
        ("cos", math.cos(0.5)),
        ("tan", math.tan(0.5)),
        ("tanh", math.tanh(0.5)),
        ("exp", math.exp(0.5)),
        ("log1p", math.log1p(0.5)),
        ("sqrt1p", math.sqrt(1.5)),
        ("cosh", math.cosh(0.5)),
        ("sinh", math.sinh(0.5)),
    ],
)
def test_ordinary_analytic_constant_limits_remain_numerically_available(function, expected):
    system = HyperrealSystem()
    value = getattr(system, function)(system.constant(0.5))

    assert not isinstance(value.seq, Const)
    assert value.standard_part() == pytest.approx(expected)
    assert value.seq.at(1) == pytest.approx(expected)


@pytest.mark.parametrize("expression", [Log1p(Const(-2)), Sqrt1p(Const(-2)), Exp(Const(1000))])
def test_simplification_does_not_numerically_evaluate_domains_or_overflow(expression):
    assert expression.simplify().key() == expression.key()
