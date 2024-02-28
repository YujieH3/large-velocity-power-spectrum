# Large Velocity Power Spectrum

[![](https://img.shields.io/github/license/YujieH3/Large-Velocity-Power-Spectrum.svg)](LICENSE.md)
[![Documentation Status](https://readthedocs.org/projects/vpower/badge/?version=latest)](https://vpower.readthedocs.io/en/latest/?badge=latest)
[![](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![](https://img.shields.io/github/last-commit/YujieH3/Large-Velocity-Power-Spectrum.svg)]()

This code is aimed to calculate velocity, momentum, kinetic energy power spectrum over a large dynamical range. It is still in active development and testing.
<!-- For more information, please refer to Vpower's [documentation](https://vpower.readthedocs.io/en/latest/). -->


To run the code
```
mpiexec -n 8 python parallel.py
```

To enable memory profiler, run the code with
```
mpiexec -n 8 mpirun python -m memory_profiler parallel.py
```

## Benchmark

On the same snapshot of ~ 10 million particles (9,619,086). 