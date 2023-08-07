import numpy as np
import time
from memory_profiler import profile
import os


@profile
def test_numpy(Nsize, type):
  if type == 'complex128':
    arr = np.random.rand(Nsize, Nsize, Nsize, 3) + 1j * np.random.rand(Nsize, Nsize, Nsize, 3)
    arr = arr.astype(np.complex128)
  elif type == 'complex64':
    arr = np.random.rand(Nsize, Nsize, Nsize, 3) + 1j * np.random.rand(Nsize, Nsize, Nsize, 3)
    arr = arr.astype(np.complex64)
  elif type == 'float64':
    arr = np.random.rand(Nsize, Nsize, Nsize, 3)
    arr = arr.astype(np.float64)
  elif type == 'float32':
    arr = np.random.rand(Nsize, Nsize, Nsize, 3)
    arr = arr.astype(np.float32)
  else:
    raise ValueError("type not supported")

  for i in range(3):
    t0 = time.perf_counter()
    np.save('test.npy', arr)
    t = time.perf_counter() - t0
    print("numpy save: {} s".format(t))

    t0 = time.perf_counter()
    arr = np.load('test.npy')
    t = time.perf_counter() - t0
    print("numpy load: {} s".format(t))

    os.system('rm test.npy')

if __name__ == '__main__':
  test_numpy(Nsize=1024, type='complex64')
