==================
User Guide and API
==================

Hankel Transformations and FFTLog
---------------------------------

The FFTLog algorithm can be thought of as the Fast Fourier Transform (FFT) of a logarithmically spaced
periodic sequence  (= Hankel Transform). 

**hankl** consists of two modules, the General FFTLog module and the Cosmology one. The latter is suited
for modern cosmological application and relies heavily on the former to perform the Hankel transforms.

The user can find more information about the FFTLog algorithm `here
<https://jila.colorado.edu/~ajsh/FFTLog/>`_ .

Here we will provide some usefull suggestions:

- Accuracy of the method usually improves as the range of integration is enlarged. FFTlog prefers an interval that spans many orders of magnitude.
- Resulution is important. Low resolution will introduce sharp features which in turn will cause ringing.
- When possible, use the provided preprocessing tools to zero/constant pad or extrapolate the function along a larger interval.
- FFTLog works better when the input arrays have size that is a power of 2. You can easily manage this by enabling extrapolation (ext).
- Use the lowringing value of kr (or xy).

For more information about how to use **hankl** see the API below and visit the :doc:`examples` page.

Cosmology module
----------------

.. autofunction:: hankl.P2xi

.. autofunction:: hankl.xi2P

General FFTLog module
---------------------

.. autofunction:: hankl.FFTLog


.. toctree::
   :maxdepth: 4
   :caption: Contents:
   :hidden: