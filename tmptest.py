# Transforming three components at the same time
import pyfftw
import time
import numpy as np
from memory_profiler import profile
NSIZE = 1024

@profile
def run():
  t0 = time.perf_counter()
  arr = np.random.rand(NSIZE, NSIZE, NSIZE) + 1j * np.random.rand(NSIZE, NSIZE, NSIZE)
  arr = arr.astype('complex64')
  t = time.perf_counter() - t0
  print("Random field creation: {} s".format(t))

  # Create FFTW object
  t0 = time.perf_counter()
  a = pyfftw.empty_aligned((NSIZE, NSIZE, NSIZE), dtype='complex64')
  fft_object = pyfftw.FFTW(a, a, axes=(0, 1, 2), threads=32)
  t = time.perf_counter() - t0
  print("Time for FFTW object creation: {} s".format(t))

  # Perform FFT
  for i in range(5):
    t0 = time.perf_counter()
    arr = fft_object(arr)
    t = time.perf_counter() - t0
    print("Time for FFTW: {} s".format(t))

if __name__ == '__main__':
  run()
