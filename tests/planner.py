# This script aims to provide a set of functions to easily measure the 
# computational cost (memory and time) of interpolation and FFT, in order to
# plan the final implementation.

from memory_profiler import profile
import time

import numpy as np
import h5py
import pyfftw

@profile
def time_memory(func, *args, **kwargs):
  """
  Measure the time and memory cost of a function.
  """
  t0 = time.perf_counter()
  output = func(*args, **kwargs)
  t = time.perf_counter() - t0
  print("Time cost: ", t, "s")
  return output

@profile
def numpy_fft_time_memory(Nsize):
  """
  Measure the time and memory cost of numpy fft.
  """
  grid = np.random.rand(Nsize, Nsize, Nsize)
  t0 = time.perf_counter()
  grid = np.fft.fftn(grid)
  t = time.perf_counter() - t0
  print("Time cost: ", t, "s")
  return grid

@profile
def fftw_time_memory(Nsize, threads):
  """
  Measure the time and memory cost of FFTW.
  """
  # Create FFTW object
  t0 = time.perf_counter()
  a = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex128')
  fft_object = pyfftw.FFTW(a, a, axes=(0, 1, 2), threads=threads)
  t = time.perf_counter() - t0
  print("FFTW object creation time: ", t, "s")

  # Load array
  t0 = time.perf_counter()
  # a[:] = np.load('random_field_2048.npy')
  with h5py.File('random_field_2048.hdf5', 'r') as f:
    a[:] = f['random_field'][:] # type: ignore
  t = time.perf_counter() - t0
  print("Load array time cost: ", t, "s")

  # FFTW
  t0 = time.perf_counter()
  a = fft_object(a)
  t = time.perf_counter() - t0
  print("FFT time cost: ", t, "s")
  return a


if __name__ == "__main__":
  Nsize = 2048
  threads = 16
  output = fftw_time_memory(Nsize, threads)
