"""Regressions for completion ownership, brackets, and exact simplification."""

from operator import add, sub, mul, truediv, lt, le, eq, gt, ge

import pytest

from hyperreals import HyperrealSystem


@pytest.mark.parametrize("operation", [add, sub, mul, truediv, lt, le, eq, gt, ge])
def test_incompatible_completion_contexts_are_rejected(operation):
    first, second = HyperrealSystem(), HyperrealSystem()
    x, y = first.alt(), second.alt()
    assert x < first.constant(0)
    assert y > second.constant(0)
    before = (first.puf.semantic_support, second.puf.semantic_support)
    with pytest.raises(ValueError, match="same system"):
        operation(x, y)
    assert before == (first.puf.semantic_support, second.puf.semantic_support)


@pytest.mark.parametrize("function", ["sin", "cos", "tan", "tanh", "exp", "log1p", "sqrt1p", "cosh", "sinh"])
def test_function_factory_does_not_reinterpret_foreign_values(function):
    first, second = HyperrealSystem(), HyperrealSystem()
    with pytest.raises(ValueError, match="this system"):
        getattr(first, function)(second.alt())


def test_supplied_bracket_selects_only_a_completion_inside_both_bounds():
    system = HyperrealSystem()
    x = system.alt()
    result = x.choose_standard_part(bracket=(0, 2), bits=8)
    assert result is not None
    assert result.low <= 1 <= result.high
    assert result.high - result.low <= 2 / 2**8
    assert x == system.constant(1)


def test_infeasible_bracket_makes_no_partial_commitment():
    system = HyperrealSystem()
    x = system.alt()
    support = system.puf.semantic_support
    committed = set(system.puf._committed_true)
    assert x.choose_standard_part(bracket=(-0.5, 0.5)) is None
    assert system.puf.semantic_support == support
    assert system.puf._committed_true == committed
    assert x == system.constant(1)


def test_infinite_expression_cannot_use_a_supplied_finite_bracket():
    system = HyperrealSystem()
    assert system.infinite().choose_standard_part(bracket=(-2, 2)) is None


def test_invariant_result_honors_requested_bracket():
    system = HyperrealSystem()
    assert system.constant(2).choose_standard_part(bracket=(0, 1)) is None


@pytest.mark.parametrize("bits", [0, 1])
def test_singleton_subnormal_bracket_preserves_its_value(bits):
    system = HyperrealSystem()
    tiny = 5e-324
    x = system.constant(tiny) * system.alt()
    result = x.choose_standard_part(bracket=(tiny, tiny), bits=bits)
    assert result is not None
    assert result.low == result.high == result.approx == tiny


@pytest.mark.parametrize("bracket", [(2, 1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_bracket_is_rejected_before_any_choice(bracket):
    system = HyperrealSystem()
    with pytest.raises(ValueError, match="bracket"):
        system.alt().choose_standard_part(bracket=bracket)


@pytest.mark.parametrize("policy, expected", [("lower", -1), ("upper", 1)])
def test_automatic_bracketing_preserves_both_parities_until_bisection(policy, expected):
    system = HyperrealSystem()
    x = system.alt()
    result = x.choose_standard_part(tie_break=policy, bits=8)
    assert result is not None
    assert result.low <= expected <= result.high
    assert x == system.constant(expected)


def test_simplification_does_not_round_away_a_constant_addend():
    system = HyperrealSystem()
    large = system.constant(2**53)
    larger = large + system.constant(1)
    assert large < larger
    assert not (large == larger)
    assert larger - large == system.constant(1)


def test_constant_division_preserves_nonbinary_rationals():
    system = HyperrealSystem()
    one, three = system.constant(1), system.constant(3)
    assert (one / three) * three == one


def test_small_nonzero_product_is_not_simplified_to_zero():
    system = HyperrealSystem()
    small = system.constant(1e-200)
    assert small * small > system.constant(0)


def test_large_constant_product_can_be_compared_without_float_overflow():
    system = HyperrealSystem()
    large = system.constant(1e200)
    assert large * large > large
