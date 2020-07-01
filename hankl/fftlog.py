import numpy as np
from scipy.special import gamma


def _gamma_term(mu, x, cut=200.0):
	r'''
	Compute the following term:
	
		\Gamma[(\mu + 1 + x) / 2] / \Gamma[(\mu + 1 - x) / 2]

	(see eq.16 in https://jila.colorado.edu/~ajsh/FFTLog/)

	Args:
		mu (float): Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
		x ():
		cut (float): Cuttoff value to switch to Gamma function limiting case.
	Returns:
		Gamma fraction term of eq. 16
	'''

	imag_x = np.imag(x)

	g_m = np.zeros(x.size, dtype=complex)

	asym_x = x[np.absolute(imag_x) > cut]
	asym_plus = (mu+1+asym_x)/2.
	asym_minus = (mu+1-asym_x)/2.

	x_good = x[(np.absolute(imag_x) <= cut) & (x != mu + 1 + 0.0j)]

	alpha_plus = (mu+1+x_good)/2.
	alpha_minus = (mu+1-x_good)/2.

	g_m[(np.absolute(imag_x) <= cut) & (x != mu + 1 + 0.0j)] = gamma(alpha_plus)/gamma(alpha_minus)

	# high-order expansion
	g_m[np.absolute(imag_x) > cut] = np.exp((asym_plus-0.5)*np.log(asym_plus) - (asym_minus-0.5)*np.log(asym_minus) - asym_x
                                      + 1./12 * (1./asym_plus - 1./asym_minus) + 1./360.*(1./asym_minus**3 - 1./asym_plus**3) + 1./1260*(1./asym_plus**5 - 1./asym_minus**5))

	g_m[np.where(x == mu+1+0.0j)[0]] = 0.+0.0j

	return g_m


def _lowring_kr(mu, q, L, N, kr=1.0):
	r'''
	Compute kr so that 

		(k r)^{- i \pi/dlnr} U_{\mu}(q + i \pi/dlnr)

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

	x = q + 1j*np.pi/delta_L

	x_plus = (mu+1+x)/2.
	x_minus = (mu+1-x)/2.

	phip = np.imag(np.log(gamma(x_plus)))
	phim = np.imag(np.log(gamma(x_minus)))

	arg = np.log(2.0/kr)/delta_L + (phip - phim)/np.pi
	iarg = np.rint(arg)
	if (arg != iarg):
		kr = kr*np.exp((arg-iarg)*delta_L)

	return kr


def _u_m_term(m, mu, q, kr, L):
	r'''
	Compute u_{m}(\mu, q) term defined as

		u_{m}(\mu, q) = (k_{0}r_{0})^{-2\pi i m / L} U_{\mu}(q + 2\pi i m/ L)

	(see eq.18 in https://jila.colorado.edu/~ajsh/FFTLog/)

	Args:
		m (float): Index of u_{m} term.
		mu (float): Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
		q (float):  Exponent of power law bias; q may be any real number, positive or negative.
		kr (float): Value of kr.
		L (float): Range of uniformly logarithmically spaced points.
	Returns:
		u_{m}(\mu, q) (float) : u_{m} term of eq. 18
	'''

	omega = 1j*2*np.pi*m/float(L)

	x = q + omega

	U_mu = 2**x*_gamma_term(mu, x)

	u_m = (kr)**(-omega)*U_mu

	u_m[m.size-1] = np.real(u_m[m.size-1])

	return u_m


def FFTLog(k, f_k, q, mu, kr=1.0, lowring=False):
	r'''
	Hankel Transform based on the FFTLog algorithm of [1] and [2].

	Defined as:

	.. math:: F(k)= \int_0^\infty f(r) (kr)^q J_\mu(kr) k dr 

	.. math:: f(r)= \int_0^\infty F(k) (kr)^{-q} J_\mu(kr) r dk

	Args:
		k (array):
		f_k (array):
		q (float): Exponent of power law bias; q may be any real number, positive or negative.
		mu (float): Index of J_mu in Hankel transform; mu may be any real number, positive or negative.
		kr (float): Input value of kr (Default is 1).
		lowring (bool): If True, then use low-ringing value of kr closest to input value of kr.
	Returns:
		r (array): x
		A (array): x

	References:
		[1] J. D. Talman. Numerical Fourier and Bessel Transforms in Logarithmic Variables. Journal of Computational Physics, 29:35-48, October 1978.
		
		[2] A. J. S. Hamilton. Uncorrelated modes of the non-linear power spectrum. MNRAS, 312:257-284, February 2000.
	'''

	N = f_k.size
	delta_L = (np.log(np.max(k))-np.log(np.min(k)))/float(N-1)
	L = (np.log(np.max(k))-np.log(np.min(k)))
	log_k0 = np.log(k[N//2])
	k0 = np.exp(log_k0)

	c_m = np.fft.rfft(f_k)
	m = np.fft.rfftfreq(N, d=1.)*float(N)
	if lowring:
		kr = _lowring_kr(mu, q, L, N, kr)
	r0 = kr/k0
	log_r0 = np.log(r0)

	m = np.fft.rfftfreq(N, d=1.)*float(N)
	m_r = np.arange(-N//2, N//2)
	m_shift = np.fft.fftshift(m_r)

	s = delta_L*(-m_r)+log_r0
	id = m_shift
	r = 10**(s[id]/np.log(10))

	u_m = _u_m_term(m, mu, q, kr, L)
	b = c_m*u_m

	A_m = np.fft.irfft(b)
	A = A_m[id]

	A = A[::-1]
	r = r[::-1]

	if (q != 0):
		A = A*(r)**(-float(q))

	return r, A


