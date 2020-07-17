========
Examples
========

This section includes two examples. The first one uses the General FFTLog module and the second uses the Cosmology module.

General FFTLog Example
----------------------

This is a simple example, used by Hamilton to illustrate how the FFTLog algorithm works.
We will use **hankl**'s General FFTLog module to compute the following transform:

    .. math:: \int_{0}^{\infty} r^{\mu + 1} exp\Big(-\frac{r^{2}}{2}\Big) J_{\mu}(k r) k dr = k^{\mu+1} exp\Big(-\frac{k^{2}}{2}\Big)

Now in this example we know the analytical form of the result so we will use this to
demonstrate the accuracy of the transformation.

The general form of the Hankel transform that **hankl** computes is the following:

    .. math:: g(k) = \int_0^\infty f(r) (kr)^{q} J_\mu(kr) k dr

This means that the function that we want to transform is:

    .. math:: f(r) = r^{\mu+1} exp\Big(-\frac{r^{2}}{2}\Big)

and the result should be

    .. math:: g(k) = k^{\mu+1} exp\Big(-\frac{k^{2}}{2}\Big)

which is of course the right hand side (RHS) of the first equation.

Now let's start by importing everything we need::

    import hankl
    import numpy as np
    import matplotlib.pyplot as pyplot

The next thing we want is to define the functions f(r) and g(r) as well as the integration range for *r*::

    def f(r, mu=0.0):
        return r**(mu+1.0) * np.exp(-r**2.0 / 2.0)

    def g(k, mu=0.0):
        return k**(mu+1.0) * np.exp(- k**2.0 / 2.0)

    r = np.logspace(-5, 5, 2**10)

As you can see, we used an integration range for *r* which is quite wide and we also chose its size to be
a power of two, this will make the algorithm faster and more accurate.

Now let's perform the Hankel transform::

    k, G = hankl.FFTLog(r, f(r, mu=0.0), q=0.0, mu=0.0)

Finally, we can plot the results::

    plt.figure(figsize=(10,6))

    ax1 = plt.subplot(121)
    plt.loglog(r, f(r))
    plt.title('$f(r) = r \; exp(-r^{2}/2)$')
    plt.xlabel('$r$')
    plt.ylim(10**(-6), 1)
    plt.xlim(10**(-5), 10)

    ax1.yaxis.tick_left()
    ax2.yaxis.set_label_position("left")

    ax2 = plt.subplot(122, sharey=ax1)
    plt.loglog(k, g(k), label='Analytical')
    plt.loglog(k, G, ls='--', label='hankl - FFTLog')
    plt.title('$g(k) = k \; exp(-k^{2}/2)$')
    plt.xlabel('$k$')
    plt.ylim(10**(-6), 1)
    plt.xlim(10**(-5), 10)
    plt.legend()

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.tight_layout()

    plt.show()

.. figure:: hankl_test.png

We can further impove the performace of **hankl** by enabling the 'lowring' option, extrapolating or zero/constant padding the function (See the API for more information).


Cosmology Example
-----------------

For instance, if you wanted to Hankel-transform a 2-point Correlation Function to get the Power Spectrum, you would do something like::

    import numpy as np
    import hankl

    # Create mock 2-point Correlation Function
    r = np.logspace(-3, 3, 100)
    xi = 1.0 / (1.0 + r * r) ** 1.5

    # Hankel-transform it to get the Power Spectrum
    k, P = hankl.xi2P(r, xi, l=0)

    # Hankel-transform the Power Spectrum back to Configuration Space
    r_new, xi_new = hankl.P2xi(k, P, l=0)

    # Check results
    np.allclose(r, r_new)
    np.allclose(xi, xi_new)


.. toctree::
   :maxdepth: 4
   :caption: Contents:
   :hidden: