from hyperreal import HyperrealSystem


def test_standard_part_bounded_times_infinitesimal_alt():
    sys = HyperrealSystem()
    eps = sys.infinitesimal()
    a = sys.alt() * eps
    assert a.standard_part() == 0.0


def test_standard_part_constant_plus_infinitesimal_tail():
    sys = HyperrealSystem()
    eps = sys.infinitesimal()
    x = sys.constant(1.0) + (sys.alt() * eps)
    assert x.standard_part() == 1.0


def test_standard_part_sin_over_n():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = sys.sin(n) / n
    assert x.standard_part() == 0.0


def test_standard_part_tanh_pm_infty():
    sys = HyperrealSystem()
    n = sys.infinite()
    assert sys.tanh(n).standard_part() == 1.0
    assert sys.tanh(sys.constant(-1.0) * n).standard_part() == -1.0


def test_standard_part_exp_minus_infty():
    sys = HyperrealSystem()
    n = sys.infinite()
    assert sys.exp(sys.constant(-1.0) * n).standard_part() == 0.0


def test_standard_part_alt_sign_is_completion_dependent():
    sys = HyperrealSystem()
    assert sys.alt().standard_part() is None


def test_standard_part_sin_over_log1p_n():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = sys.sin(n) / sys.log1p(n)
    assert x.standard_part() == 0.0


def test_standard_part_tanh_log1p_n():
    sys = HyperrealSystem()
    n = sys.infinite()
    assert sys.tanh(sys.log1p(n)).standard_part() == 1.0


def test_standard_part_exp_neg_log1p_n():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = sys.exp(sys.constant(-1.0) * sys.log1p(n))
    assert x.standard_part() == 0.0
