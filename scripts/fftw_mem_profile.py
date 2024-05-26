# python -m memory_profiler fftw_mem_profile.py


from memory_profiler import profile
import pyfftw
import numpy as np

import time

@profile
def FFTW_power(f, Lbox, Nsize):
    """
    Same with vector_power, but using the fft_object to allow full power of 
    FFTW.
    """
    # Fourier transform
    const = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3

    a = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    b = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b 
    # maybe initialize f as an empty aligned array?
    
    a[:] = f
    fft_object() # the result is stored in b
    del f
    fk = 0.5 * np.abs(b * const)**2

    return fk


#@profile
def fftw_vector_power(fx, fy, fz, lbox, nsize):
    """
    same with vector_power, but using the fft_object to allow full power of 
    fftw.
    """
    # fourier transform
    const = (lbox / (2 * np.pi)) ** 1.5 / nsize ** 3

    a = pyfftw.empty_aligned((nsize, nsize, nsize), dtype='complex64')
    b = pyfftw.empty_aligned((nsize, nsize, nsize), dtype='complex64')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b 
    # maybe initialize f as an empty aligned array?
    
    a = fx
    fft_object() # the result is stored in b
    del fx
    fk = 0.5 * np.abs(b * const)**2

    a = fy
    fft_object()
    del fy
    fk += 0.5 * np.abs(b * const)**2

    a = fz
    fft_object()
    del fz
    fk += 0.5 * np.abs(b * const)**2

    return fk


#@profile
def fftw_vector_power2(fx, fy, fz, lbox, nsize):
    """
    same with vector_power, but using the fft_object to allow full power of 
    fftw.
    """
    # fourier transform
    const = (lbox / (2 * np.pi)) ** 1.5 / nsize ** 3

    a = pyfftw.empty_aligned((nsize, nsize, nsize), dtype='complex64')
    b = pyfftw.empty_aligned((nsize, nsize, nsize), dtype='complex64')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b 
    # maybe initialize f as an empty aligned array?
    
    a[:] = fx
    fft_object() # the result is stored in b
    del fx
    fk = 0.5 * np.abs(b * const)**2

    a[:] = fy
    fft_object()
    del fy
    fk += 0.5 * np.abs(b * const)**2

    a[:] = fz
    fft_object()
    del fz
    fk += 0.5 * np.abs(b * const)**2

    return fk




#@profile
def main():
    Nsize = 512
    f = np.random.rand(Nsize, Nsize, Nsize, 3).astype('float32')
    f = f + 1j*f
    f = f.astype('complex64')

    # Use same random field, compare the output
    g = f.copy()
    t1 = time.perf_counter()
    f = fftw_vector_power(f[...,0], f[...,1], f[...,2], 1, Nsize)
    print(time.perf_counter() - t1)

    t1 = time.perf_counter()
    g = fftw_vector_power2(g[...,0], g[...,1], g[...,2], 1, Nsize)
    print(time.perf_counter() - t1)

    print(np.sum(f - g))


if __name__ == '__main__':
    main()

