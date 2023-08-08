.. Vpower documentation master file, created by
   sphinx-quickstart on Tue Aug  8 01:57:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Vpower's documentation!
==================================


**Vpower** is a Python library for velocity power spectrum estimation from astrophysical N-body simulations. The code is a product of a research project aiming to study interstellar turbulence through simulation power spectrum. Vpower is specialized to calculate velocity power spectrum over a large dynamical range using the folding technique. It has a nearest neighbor (like) interpolator using `ANN <http://www.cs.umd.edu/~mount/ANN/>`_ and additionally a smoothed interpolator with `Voxelize <https://github.com/leanderthiele/voxelize>`_. 

Vpower's main features include:

- Center of mass velocity field interpolation
- Velocity power spectrum computation with the folding technique [HWK1989]_, allowing a more extensive dynamical range

It can also

- Interpolate density field
- Calculate momentum, kinetic energy field

.. note:: 

   This project is under active development. If you have any question, feel free to contact us at `yujie.jay.he@foxmail.com <mailto:yujie.jay.he@foxmail.com>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started

.. [HWK1989] Performing Fourier transforms on extremely long data streams, Hocking WK. 1989. Comput. Phys. 3(1):59 `DOI:10.1063/1.168338 <https://doi.org/10.1063/1.168338>`_

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
