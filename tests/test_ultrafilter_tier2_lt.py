"""Tests for Tier-2 asymptotic facts integration with LT classification."""

from hyperreals import HyperrealSystem


def test_lt_exp_n_const_forced_false_and_true():
    sys = HyperrealSystem()
    n = sys.infinite()

    # exp(n) eventually exceeds every constant -> {n: exp(n) < C} is finite.
    assert (sys.exp(n) < sys.constant(1000.0)) is False

    # exp(n) is strictly positive -> {n: 0 < exp(n)} is cofinite.
    assert (sys.constant(0.0) < sys.exp(n)) is True


def test_lt_exp_minus_n_sign_sensitive():
    sys = HyperrealSystem()
    n = sys.infinite()
    e = sys.exp(sys.constant(-1.0) * n)

    # exp(-n) is strictly positive for all n, even though it tends to 0.
    assert (sys.constant(0.0) < e) is True
    assert (e < sys.constant(0.0)) is False
