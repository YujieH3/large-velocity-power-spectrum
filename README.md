# Large Velocity Power Spectrum

[![](https://img.shields.io/github/license/YujieH3/Large-Velocity-Power-Spectrum.svg)](LICENSE.md)
![Work in Progress](https://img.shields.io/badge/status-in%20progress-yellow)

> **Warning:** This project is in active development. Some features may change, and full documentation will be available soon.

<!-- [![Documentation Status](https://readthedocs.org/projects/vpower/badge/?version=latest)](https://vpower.readthedocs.io/en/latest/?badge=latest) -->

This project aims to calculate the velocity, momentum, and kinetic energy power spectrum over a large dynamical range for N-body simulations. This code is designed to handle large-scale simulations using parallelization via MPI, making it scalable across multiple cores.

<!-- For more information, please refer to Vpower's [documentation](https://vpower.readthedocs.io/en/latest/). -->

## Features
- Compute power spectra for velocity, momentum, and kinetic energy from N-body simulation outputs.
- Scalable and optimized for distributed computing using MPI.

<!-- ## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YujieH3/Large-Velocity-Power-Spectrum.git
   ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ``` -->

## Running the Code

To run the script on multiple cores using MPI:
```
mpiexec -n <number of cores> python ./scripts/parallel_optimized.py
```




