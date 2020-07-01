r''' This version of the fast Hankle transform is due to 
    Andrew Hamilton (see http://casa.colorado.edu/~ajsh/FFTLog/). 
    
	Joseph E. McEwen 
	McEwen Laboratories (c) 2016 
	email: jmcewen314@gmail.com
		
	This code is available for anyone to use, but please give approriate reference to 
	Joseph E. McEwen and the authors of the algorithm. 
	
	The Hankel transform in this code is defined as : 
	F(k)= \int_0^\infty f(r) (kr)^q J_\mu(kr) k dr 
	f(r)= \int_0^\infty F(k) (kr)^{-q} J_\mu(kr) r dk .  
		
'''

from __future__ import division

import numpy as np
from numpy.fft import fft, ifft , fftshift, ifftshift , rfft, irfft
from numpy import exp, log, log10, cos, sin, pi
from scipy.special import gamma
from time import time
from numpy import gradient as grad
import sys

log2 = log(2)
cut = 200


def g_m_vals(mu, q):
	"""
	This function is copied from gamma_funcs.py in fastpt.
	We repeat it so that HT.py is self-contained, requiring only
	numpy and scipy.
	"""
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


def get_k0(N, mu, q, r0, L, k0):

	kr = float(k0*r0)
	delta_L = L/float(N)

	x = q + 1j*pi/delta_L

	x_plus = (mu+1+x)/2.
	x_minus = (mu+1-x)/2.

	phip = np.imag(log(gamma(x_plus)))
	phim = np.imag(log(gamma(x_minus)))

	arg = log(2/kr)/delta_L + (phip - phim)/pi
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


def FFTLog(k, f_k, q, mu, lowring=False):

	if ((q+mu) < -1):
		print('Error in reality condition for Bessel function integration.')
		print(' q+mu is less than -1.')
		print('See Abramowitz and Stegun. Handbook of Mathematical Functions pg. 486')

	if (q > 1/2.):
		print('Error in reality condition for Bessel function integration.')
		print(' q is greater than 1/2')
		print('See Abramowitz and Stegun. Handbook of Mathematical Functions pg. 486')

	N = f_k.size
	delta_L = (log(np.max(k))-log(np.min(k)))/float(N-1)
	L = (log(np.max(k))-log(np.min(k)))

	# find a better way to check if it is evenly spaced in log
	diff = np.diff(np.log(k))
	diff = np.diff(diff)
	if (np.sum(diff) >= 1e-10):
		print('You need to send in data that is sampled evenly in logspace')
		print('Terminating code in fft_log')
		sys.exit()

	log_k0 = log(k[N//2])
	k0 = exp(log_k0)

	# Fourier transform input data
	# get m values, shifted so the zero point is at the center

	c_m = rfft(f_k)
	m = np.fft.rfftfreq(N, d=1.)*float(N)
	# make r vector
	if lowring:
		kr=get_k0(float(N),mu,q,1/k0,L,k0)
	else:
		kr = 1
	r0 = kr/k0
	log_r0 = log(r0)

	m = np.fft.rfftfreq(N, d=1.)*float(N)
	m_r = np.arange(-N//2, N//2)
	m_shift = np.fft.fftshift(m_r)

	#s-array
	s = delta_L*(-m_r)+log_r0
	id = m_shift
	r = 10**(s[id]/log(10))

	#m_shift=np.fft.fftshift(m)

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


