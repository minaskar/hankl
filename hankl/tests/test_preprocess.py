import numpy as np
import pytest

import hankl


def f(r, mu=0.0):
    return r**(mu+1.0) * np.exp(-r**2.0 / 2.0)


def test_preprocess(seed=42):
    np.random.seed(seed)
    x = np.logspace(-2, 2, 100)
    x_prime, f_prime, N_left, N_right = hankl.preprocess(x, f(x), ext=(3, 1), range=(1e-9, 1e+9))
    assert N_left == N_right
    assert x_prime.size == x.size + N_left + N_right
    assert np.max(np.abs(f_prime-f(x_prime))) < 1e-6
