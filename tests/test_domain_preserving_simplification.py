"""Operand-discarding rewrites must preserve unsupported expression domains."""

import pytest

from hyperreals import HyperrealSystem, UnderdeterminedComparisonError
from hyperreals.sequence import Const, Div, Exp, InvN, Log1p, Mul, NVar, Sin, Sqrt1p, Sub
from hyperreals.sequence.domains import is_eventually_real


@pytest.mark.parametrize("operation", ["subtract_self", "multiply_left", "multiply_right"])
def test_zero_division_cannot_be_erased_into_a_certified_zero(operation):
    system = HyperrealSystem()
    zero = system.constant(0)
    undefined = zero / zero
    if operation == "subtract_self":
        expression = undefined - undefined
    elif operation == "multiply_left":
        expression = zero * undefined
    else:
        expression = undefined * zero

    assert expression.compare_eq(zero) is None
    assert expression.compare_lt(zero) is None
    with pytest.raises(UnderdeterminedComparisonError):
        _ = expression == zero
    assert system.puf.is_semantically_extendible()


def test_an_everywhere_real_function_does_not_hide_its_undefined_argument():
    system = HyperrealSystem()
    zero = system.constant(0)
    undefined = system.exp(zero / zero)

    assert (undefined - undefined).compare_eq(zero) is None
    assert (zero * undefined).compare_eq(zero) is None


@pytest.mark.parametrize("constant", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_constants_do_not_cancel_to_finite_values(constant):
    value = Const(constant)

    assert isinstance(Sub(value, value).simplify(), Sub)
    assert isinstance(Mul(Const(0), value).simplify(), Mul)
    assert isinstance(Div(value, value).simplify(), Div)


@pytest.mark.parametrize(
    "defined",
    [
        NVar(),
        InvN(),
        Sin(NVar()),
        Exp(NVar()),
        Div(Sin(NVar()), Exp(NVar())),
        Div(InvN(), Sub(NVar(), Const(1))),
        Log1p(NVar()),
        Sqrt1p(NVar()),
    ],
)
def test_recognized_eventual_domains_keep_valid_zero_simplifications(defined):
    assert is_eventually_real(defined)
    assert Sub(defined, defined).simplify().key() == Const(0).key()
    assert Mul(Const(0), defined).simplify().key() == Const(0).key()
    assert Mul(defined, Const(0)).simplify().key() == Const(0).key()


@pytest.mark.parametrize(
    "unsupported",
    [Div(Const(0), Const(0)), Log1p(Mul(Const(-1), NVar())), Sqrt1p(Mul(Const(-1), NVar()))],
)
def test_unsupported_domains_remain_visible_to_the_compiler(unsupported):
    assert not is_eventually_real(unsupported)
    assert isinstance(Sub(unsupported, unsupported).simplify(), Sub)
    assert isinstance(Mul(Const(0), unsupported).simplify(), Mul)
