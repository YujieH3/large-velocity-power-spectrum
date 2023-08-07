import numpy as np
import pyfftw
import time
import h5py

test_repeat = 8 # Number of FFTs to perform for each configuration

for Nsize in [256]:
  # f = h5py.File('fft_benchmark512.hdf5','w') # Create HDF5 file to store results
  with h5py.File('fft_benchmark.hdf5','a') as f:
    f.require_group('N{}'.format(Nsize))

  # Generate test data cube
  np.random.seed(1)
  grid = np.random.rand(Nsize, Nsize, Nsize)
  for threads in [1, 4, 8, 16]:
    subgroup_name = 'N{}/threads{}'.format(Nsize, threads)
    skip_fftw = False
    skip_fftw_interface = False
    skip_fft_numpy = False
    with h5py.File('fft_benchmark.hdf5','a') as f:
      f.require_group(subgroup_name)
      if 'fftw' in f[subgroup_name].keys():
        skip_fftw = True
      if 'fftw_interface' in f[subgroup_name].keys():
        skip_fftw_interface = True
      if 'fft_numpy' in f[subgroup_name].keys():
        skip_fft_numpy = True

    if skip_fftw == False:
      # Create FFTW object
      t0 = time.time()
      a = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex128')
      b = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex128')
      fft_object = pyfftw.builders.fftn(a, threads=threads)
      t1 = time.time()

      with h5py.File('fft_benchmark.hdf5','a') as f: # Write
        f[subgroup_name].attrs['fftw_object'] = t1 - t0
      print('FFTW object creation time: {}'.format(t1 - t0))

      # FFTW
      times = []
      for i in range(test_repeat):
        t0 = time.time()
        fft_object(grid)
        t1 = time.time()
        t = t1 - t0
        times.append(t)
        print('Method: fftw, Nsize: {}, threads: {}, time: {}'.format(Nsize, threads, t))

      del a, b, fft_object # Clear memory

      with h5py.File('fft_benchmark.hdf5','a') as f: # Write
        f[subgroup_name + '/fftw'] = times # type: ignore

    if skip_fftw_interface == False:
      # FFTW interface
      pyfftw.interfaces.cache.enable()
      times = []
      for i in range(test_repeat):
        t0 = time.time()
        pyfftw.interfaces.numpy_fft.fftn(grid, threads=threads)
        t1 = time.time()
        t = t1 - t0
        times.append(t)
        print('Method: fftw_interface, Nsize: {}, threads: {}, time: {}'.format(Nsize, threads, t))
        
      with h5py.File('fft_benchmark.hdf5','a') as f: # Write
        f[subgroup_name + '/fftw_interface'] = times # type: ignore

    if skip_fft_numpy == False:
      # Numpy fft
      if threads == 1:
        times = []
        for i in range(test_repeat):
          t0 = time.time()
          np.fft.fftn(grid)
          t1 = time.time()
          t = t1 - t0
          times.append(t)
          print('Method: fft_numpy, Nsize: {}, threads: {}, time: {}'.format(Nsize, threads, t))

        with h5py.File('fft_benchmark.hdf5','a') as f: # Write
          f[subgroup_name + '/fft_numpy'] = times # type: ignore
