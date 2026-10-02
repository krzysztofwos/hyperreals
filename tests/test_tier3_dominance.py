from hyperreals import HyperrealSystem


def test_log_over_n_goes_to_zero():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = sys.log1p(n) / n
    assert x.standard_part() == 0.0


def test_n_over_exp_n_goes_to_zero():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = n / sys.exp(n)
    assert x.standard_part() == 0.0


def test_polynomial_ratio_leading_coeff():
    sys = HyperrealSystem()
    n = sys.infinite()
    one = sys.constant(1.0)
    n2 = n * n
    x = (n2 + one) / (n2 + n)
    assert x.standard_part() == 1.0


def test_lower_degree_over_higher_degree_poly():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = n / (n * n + n)
    assert x.standard_part() == 0.0


def test_poly_times_exp_minus_n():
    sys = HyperrealSystem()
    n = sys.infinite()
    x = sys.exp(sys.constant(-1.0) * n) * (n * n)
    assert x.standard_part() == 0.0
