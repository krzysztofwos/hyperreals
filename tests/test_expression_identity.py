"""Regression tests for exact symbolic expression identity."""

from hyperreals.algebra import Atom
from hyperreals.asymptotic_facts import analyze, clear_cache
from hyperreals.sequence import Add, Const, NVar, Sub
from hyperreals.ultrafilter import PartialUltrafilter


def test_close_constants_have_distinct_expression_and_atom_keys():
    one = Const(1.0)
    near_one = Const(1.0 + 5e-13)

    assert repr(one) == repr(near_one)
    assert one.key() != near_one.key()

    forward = Atom("LT", one, near_one)
    reverse = Atom("LT", near_one, one)
    assert forward.key() != reverse.key()

    puf = PartialUltrafilter()
    assert puf._ensure_var(forward) != puf._ensure_var(reverse)


def test_simplification_does_not_cancel_rendering_collisions():
    left = Add(NVar(), Const(1.0)).simplify()
    right = Add(NVar(), Const(1.0 + 5e-13)).simplify()

    difference = Sub(left, right).simplify()

    assert isinstance(difference, Sub)
    assert difference.key() == ("sub", left.key(), right.key())


def test_asymptotic_cache_uses_structural_identity():
    clear_cache()
    one = analyze(Const(1.0))
    near_one = analyze(Const(1.0 + 5e-13))

    assert one.limit == 1.0
    assert near_one.limit == 1.0 + 5e-13
