"""Regression tests for semantic finite-intersection checking."""

import pytest

from hyperreals import HyperrealSystem, UnderdeterminedComparisonError
from hyperreals.algebra import Atom
from hyperreals.semantic_sets import EventuallyPeriodicSet, eventually_periodic_set
from hyperreals.sequence import AltSign, Const


def test_eventually_periodic_intersection_detects_disjoint_parities():
    negative = eventually_periodic_set(Atom("LT", AltSign(), Const(0.0)))
    equal_one = eventually_periodic_set(Atom("EQ", AltSign(), Const(1.0)))

    assert negative == EventuallyPeriodicSet(0, 2, frozenset({1}))
    assert equal_one == EventuallyPeriodicSet(0, 2, frozenset({0}))
    assert negative.intersect(equal_one).is_infinite() is False


def test_alternating_choices_preserve_an_infinite_joint_support():
    system = HyperrealSystem()
    alternating = system.alt()

    assert alternating < system.constant(0.0)
    assert not (alternating == system.constant(1.0))
    assert system.puf.is_semantically_extendible()
    assert system.puf.semantic_support == EventuallyPeriodicSet(
        0, 2, frozenset({1})
    )


def test_reversing_query_order_selects_the_other_completion():
    system = HyperrealSystem()
    alternating = system.alt()

    assert alternating == system.constant(1.0)
    assert not (alternating < system.constant(0.0))
    assert system.puf.semantic_support == EventuallyPeriodicSet(
        0, 2, frozenset({0})
    )


def test_second_disjoint_strict_comparison_is_forced_false():
    system = HyperrealSystem()
    alternating = system.alt()

    assert alternating < system.constant(0.0)
    assert not (system.constant(-0.5) < alternating)
    assert system.puf.is_semantically_extendible()


def test_unknown_comparison_does_not_mutate_semantic_state():
    system = HyperrealSystem()
    left = system.sin(system.infinite())
    right = system.cos(system.infinite())
    before_support = system.puf.semantic_support
    before_committed = set(system.puf._committed_true)

    assert left.compare_lt(right) is None
    assert system.puf.semantic_support == before_support
    assert system.puf._committed_true == before_committed

    with pytest.raises(UnderdeterminedComparisonError, match="supported semantic"):
        _ = left < right


def test_legacy_unknown_choice_invalidates_the_certificate_state():
    system = HyperrealSystem(allow_uncertified_choices=True)

    assert system.sin(system.infinite()) < system.cos(system.infinite())
    assert system.puf.semantic_support is None
    assert not system.puf.is_semantically_extendible()
