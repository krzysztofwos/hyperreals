"""Regression tests for exact arithmetic at the semantic comparison boundary."""

from fractions import Fraction

import pytest

from hyperreals import HyperrealSystem
from hyperreals.algebra import Atom
from hyperreals.exact_arithmetic import exact_laurent_coefficients
from hyperreals.semantic_sets import EventuallyPeriodicSet, eventually_periodic_set
from hyperreals.sequence import Add, AltSign, Const, Div, Exp, InvN, Mul, NVar, Sin, Sub


def test_large_laurent_coefficients_do_not_erase_a_nonzero_difference():
    system = HyperrealSystem()
    index = system.infinite()
    original = index * system.constant(2**53)
    larger = original + index

    assert not (original == larger)
    assert original < larger
    assert larger - original == index
    assert system.puf.is_semantically_extendible()


@pytest.mark.parametrize("coefficient", [1e-200, 1e308])
def test_laurent_coefficient_products_do_not_underflow_or_overflow(coefficient):
    system = HyperrealSystem()
    term = system.infinite() * system.constant(coefficient)
    product = term * term
    zero = system.constant(0)

    assert zero < product
    assert not (product == zero)


def test_periodic_coefficients_preserve_large_integer_differences():
    system = HyperrealSystem()
    alternating = system.alt()
    original = alternating * system.constant(2**53)
    larger_on_even_indices = original + alternating

    assert not (original == larger_on_even_indices)
    assert original < larger_on_even_indices
    assert system.puf.semantic_support == EventuallyPeriodicSet(0, 2, frozenset({0}))


def test_periodic_coefficient_products_do_not_underflow():
    system = HyperrealSystem()
    tiny = system.constant(1e-200)
    alternating = (system.alt() * tiny) * tiny
    zero = system.constant(0)

    assert not (alternating == zero)
    assert zero < alternating
    assert system.puf.semantic_support == EventuallyPeriodicSet(0, 2, frozenset({0}))


@pytest.mark.parametrize("constant", [1e-309, 5e-324])
def test_tiny_finite_threshold_comparisons_do_not_overflow(constant):
    system = HyperrealSystem()
    epsilon = system.infinitesimal()
    threshold = system.constant(constant)

    assert epsilon < threshold
    assert not (threshold < epsilon)
    assert not (epsilon == threshold)
    support = system.puf.semantic_support
    assert support is not None
    assert Fraction(1, support.cutoff) < Fraction.from_float(constant)


def test_raw_constant_arithmetic_is_not_simplified_with_floats():
    rounded = Const(2**53)
    exact_sum = Add(rounded, Const(1))

    equality = eventually_periodic_set(Atom("EQ", rounded, exact_sum))
    ordering = eventually_periodic_set(Atom("LT", rounded, exact_sum))

    assert equality == EventuallyPeriodicSet.empty()
    assert ordering == EventuallyPeriodicSet.universe()


def test_laurent_extraction_uses_exact_division_and_cancellation():
    third = Div(NVar(), Const(3))
    reconstructed = Mul(third, Const(3))

    assert exact_laurent_coefficients(third) == {-1: Fraction(1, 3)}
    assert exact_laurent_coefficients(Sub(reconstructed, NVar())) == {}
    assert exact_laurent_coefficients(Div(Const(1), Mul(Const(3), InvN()))) == {
        -1: Fraction(1, 3)
    }


@pytest.mark.parametrize(
    "unsupported",
    [Div(Const(0), Const(0)), Div(Const(1), Sub(AltSign(), Const(1))), Sin(NVar())],
)
def test_reflexivity_and_exp_positivity_require_supported_total_operands(unsupported):
    assert eventually_periodic_set(Atom("EQ", unsupported, unsupported)) is None
    assert eventually_periodic_set(Atom("LT", Const(0), Exp(unsupported))) is None
    assert exact_laurent_coefficients(Sub(unsupported, unsupported)) is None


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_constants_have_no_exact_comparison_certificate(value):
    constant = Const(value)

    assert exact_laurent_coefficients(constant) is None
    assert eventually_periodic_set(Atom("EQ", constant, constant)) is None
