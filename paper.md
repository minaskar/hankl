---
title: 'hankl: A lightweight Python implementation of the FFTLog algorithm for Cosmology'
tags:
  - Python
  - astronomy
  - cosmology
  - fftlog
  - numerical intergration
  - Hankel transform
authors:
  - name: Minas Karamanis
    orcid: 0000-0001-9489-4612
    affiliation: 1
  - name: Florian Beutler
    orcid: 0000-0003-0467-5438
    affiliation: 1
affiliations:
 - name: Institute for Astronomy, University of Edinburgh, Royal Observatory, Blackford Hill, Edinburgh EH9 3HJ, UK
   index: 1
date: 8 July 2020
bibliography: paper.bib
---

# Summary

The Hankel transform (also known as the Fourier-Bessel transform) is an integral
transformation whose kernel is a Bessel function. The Hankel transform appears
very often in physical problems with spherical or cylindrical symmetry as it
emerges when one writes the usual Fourier transform in spherical coordinates.
The Hankel transform finds application in a wide range of scientific fields,
namely cosmology, astrophysics, geophysics, and fluid mechanics.

As an example, in modern cosmology, the large-scale clustering of galaxies in
the observable universe is often described by means of the configuration-space
2-point Correlation function and its Fourier-space counterpart, the Power
Spectrum [@peebles:1980]. Due to the statistical isotropy of the universe these
two quantities are related by a Hankel transformation. The ability to perform
such transformations in a fast and accurate manner is of paramount importance
for studies of the large-scale structure of the universe.

However, the implementation of the Hankel transform poses some serious numerical
challenges. Most importantly, the Bessel function kernel is a highly oscillatory
function and any naive implementation of the quadrature numerical integration
methods could lead to grossly inaccurate results. To successfully overcome this
issue, @talman:1978, and later @hamilton:2000, introduced the FFTLog algorithm,
which can be thought of as the Fast Fourier Transform of a logarithmically
spaced periodic sequence.

# Statement of need 

`hankl` is a lightweight Python implementation of the FFTLog algorithm with
particular focus on cosmological applications. `hankl` relies on the `NumPy` and
`SciPy` libraries in order to provide fast and accurate Hankel transforms with
minimal computational overhead. `hankl` is well suited for scientific
applications that require a dead-simple and modular Python interface along with
C-level performance.

# References