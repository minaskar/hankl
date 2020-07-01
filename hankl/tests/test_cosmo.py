import numpy as np
import pytest

import hankl

def test_cosmo(seed=42):
    np.random.seed(seed)
    r = np.logspace(-3, 3, 100)
    xi = 1.0 / (1.0 + r * r) ** 1.5
    k, P = hankl.xi2P(r, xi, l=0)
    r_new, xi_new = hankl.P2xi(k, P, l=0)
    assert np.allclose(r, r_new)
    assert np.allclose(xi, xi_new)
