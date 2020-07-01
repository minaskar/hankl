import numpy as np
from .fftlog import FFTLog


def P2xi(k, P, l, n=0, lowring=False):
    r'''
    Hankel Transform Power Spectrum Multipole to Correlation Function Multipole.

    \xi_{l}^{(n)}(r) = i^{l} \int_{0}^{\infty} k^{2} dk / (2 \pi^{2}) (kr)^{-n} P_{l}^{(n)}(k) j_{l}(ks)

    Args:
        k (array): Array of log-separated wavenumbers.
        P (array): Array of respective Power Spectrum values.
        l (int): Degree of Power Spectrum multipole.
        n (int): Order of expansion (Default is 0, plane-parallel).
    Returns:
        separation array and correlation function array.

    '''
    r, f = FFTLog(k, P*k**1.5, q=-n, mu=l+0.5, lowring=lowring)
    return r, f * (2.0*np.pi)**(-1.5) * r**(-1.5) * (1j)**l


def xi2P(r, xi, l, n=0, lowring=False):
    r'''
    Hankel Transform Correlation Function Multipole to Power Spectrum Multipole.

    P_{l}^{(n)}(k) = 4 \pi (-i)^{l} \int_{0}^{\infty} r^{2} dr (kr)^{n} \xi_{l}^{(n)}(r) j_{l}(kr)

    Args:
        r (array): Array of log-separated separations.
        xi (array): Array of respective two point correlation function values.
        l (int): Degree of Power Spectrum multipole.
        n (int): Order of expansion (Default is 0, plane-parallel).
    Returns:
        wavenumber array and power spectrum array.
    '''
    k, F = FFTLog(r, xi*r**1.5, q=n, mu=l+0.5, lowring=lowring)
    return k, F * (2.0*np.pi)**1.5 * k**(-1.5) * (-1j)**l
 
