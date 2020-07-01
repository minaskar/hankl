import numpy as np
from numpy.fft import fft, ifft , fftshift, ifftshift , rfft, irfft
from numpy import exp, log, log10, cos, sin, pi
from scipy.special import loggamma, gamma
from time import time
from numpy import gradient as grad
import sys

log2 = log(2)
cut = 200


def g_m_vals(mu, q):
	imag_q = np.imag(q)

	g_m = np.zeros(q.size, dtype=complex)

	asym_q = q[np.absolute(imag_q) > cut]
	asym_plus = (mu+1+asym_q)/2.
	asym_minus = (mu+1-asym_q)/2.

	q_good = q[(np.absolute(imag_q) <= cut) & (q != mu + 1 + 0.0j)]

	alpha_plus = (mu+1+q_good)/2.
	alpha_minus = (mu+1-q_good)/2.

	g_m[(np.absolute(imag_q) <= cut) & (q != mu + 1 + 0.0j)] = gamma(alpha_plus)/gamma(alpha_minus)

	# high-order expansion
	g_m[np.absolute(imag_q) > cut] = exp((asym_plus-0.5)*log(asym_plus) - (asym_minus-0.5)*log(asym_minus) - asym_q
                                      + 1./12 * (1./asym_plus - 1./asym_minus) + 1./360.*(1./asym_minus**3 - 1./asym_plus**3) + 1./1260*(1./asym_plus**5 - 1./asym_minus**5))

	g_m[np.where(q == mu+1+0.0j)[0]] = 0.+0.0j

	return g_m


def _lowring_kr(mu, q, L, N, kr=1.0):
	r'''
	Compute kr so that 

		(k r)^{- i pi/dlnr} U_{\mu}(q + i pi/dlnr)

	is real thus reducing lowringing of the Hankel Transform.

	Args:
		mu (float): Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
		q (float): Exponent of power law bias; q may be any real number, positive or negative.
		L (float): Range of uniformly logarithmically spaced points.
		N (int): Number of uniformly logarithmically spaced points.
		kr (float): Input value of kr (Default is 1).
	Returns:
		kr (float): Low-ringing value of kr nearest to input kr.
	'''

	delta_L = L/float(N)

	x = q + 1j*pi/delta_L

	x_plus = (mu+1+x)/2.
	x_minus = (mu+1-x)/2.

	phip = np.imag(loggamma(x_plus))
	phim = np.imag(loggamma(x_minus))

	arg = log(2.0/kr)/delta_L + (phip - phim)/pi
	iarg = np.rint(arg)
	if (arg != iarg):
		kr = kr*exp((arg-iarg)*delta_L)

	return kr


def u_m_vals(m, mu, q, kr, L):

    omega = 1j*2*pi*m/L

    x = q + omega

    U_mu = 2**x*g_m_vals(mu, x)

    u_m = (kr)**(-omega)*U_mu

    u_m[m.size-1] = np.real(u_m[m.size-1])

    return u_m


def FFTLog(k, f_k, q, mu, kr=1.0, lowring=False):
	r'''
	Hankel Transform based on the FFTLog algorithm of [1] and [2].

	Defined as:

	F(k)= \int_0^\infty f(r) (kr)^q J_\mu(kr) k dr 
	f(r)= \int_0^\infty F(k) (kr)^{-q} J_\mu(kr) r dk

	References
		[1] J. D. Talman. Numerical Fourier and Bessel Transforms in Logarithmic Variables.
            Journal of Computational Physics, 29:35-48, October 1978.
		[2] A. J. S. Hamilton. Uncorrelated modes of the non-linear power spectrum.
            MNRAS, 312:257-284, February 2000.

	Args:
		k (array):
		f_k (array):
		q (float): Exponent of power law bias; q may be any real number, positive or negative.
		mu (float): Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
		kr (float): Input value of kr (Default is 1).
		lowring (bool): If True, then use low-ringing value of kr closest to input value of kr.
	Returns:
		r (array):
		A (array):
	'''

	N = f_k.size
	delta_L = (log(np.max(k))-log(np.min(k)))/float(N-1)
	L = (log(np.max(k))-log(np.min(k)))


	log_k0 = log(k[N//2])
	k0 = exp(log_k0)

	# Fourier transform input data and get m values, shifted so the zero point is at the center

	c_m = rfft(f_k)
	m = np.fft.rfftfreq(N, d=1.)*float(N)
	# make r vector
	if lowring:
		kr = _lowring_kr(mu, q, L, N, kr)
	r0 = kr/k0
	log_r0 = log(r0)

	m = np.fft.rfftfreq(N, d=1.)*float(N)
	m_r = np.arange(-N//2, N//2)
	m_shift = np.fft.fftshift(m_r)

	#s-array
	s = delta_L*(-m_r)+log_r0
	id = m_shift
	r = 10**(s[id]/log(10))

	u_m = u_m_vals(m, mu, q, kr, L)

	b = c_m*u_m

	A_m = irfft(b)

	A = A_m[id]

	# reverse the order
	A = A[::-1]
	r = r[::-1]

	if (q != 0):
		A = A*(r)**(-float(q))

	return r, A


