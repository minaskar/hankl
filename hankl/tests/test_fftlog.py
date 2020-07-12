import numpy as np
import pytest

import hankl


def f(r, mu=0.0):
    return r**(mu+1.0) * np.exp(-r**2.0 / 2.0)


def g(k, mu=0.0):
    return k**(mu+1.0) * np.exp(-k**2.0 / 2.0)


def test_fftlog(seed=42):
    np.random.seed(seed)
    r = np.logspace(-4, 4, 2**10)
    mu = 0.0
    k_hankl, G_hankl = hankl.FFTLog(r, f(r, mu), q=0.0, mu=mu, lowring=True)
    assert np.abs(G_hankl-g(k_hankl, mu)).max() < 0.001
