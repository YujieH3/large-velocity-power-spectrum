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
    
    a = f
    fft_object() # the result is stored in b
    del f
    b = 0.5 * np.abs(b * const)**2

    return b


@profile
def fftw_vector_power(fx, fy, fz, lbox, Nsize):
    """
    same with vector_power, but using the fft_object to allow full power of 
    fftw.
    """
    # fourier transform
    const = (lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3

    a = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64') 
    b = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b # important to create using empty arrays. Use fx slows down by 2x 
    # maybe initialize f as an empty aligned array?

    a = fx    
    fft_object() # the result is stored in b
    fk = 0.5 * np.abs(b * const)**2

    a = fy
    fft_object()
    fk += 0.5 * np.abs(b * const)**2

    a = fz
    fft_object()
    fk += 0.5 * np.abs(b * const)**2

    return fk


@profile
def fftw_vector_power2(fx, fy, fz, lbox, Nsize):
    """
    same with vector_power, but using the fft_object to allow full power of 
    fftw.
    """
    # fourier transform
    const = (lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3

    a = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    b = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b 
    # maybe initialize f as an empty aligned array?
    
    a[:] = fx
    fft_object() # the result is stored in b
    fk = 0.5 * np.abs(b * const)**2

    a[:] = fy
    fft_object()
    fk += 0.5 * np.abs(b * const)**2

    a[:] = fz
    fft_object()
    fk += 0.5 * np.abs(b * const)**2

    return fk




@profile
def main():
    Nsize = 512
    f = pyfftw.empty_aligned((Nsize, Nsize, Nsize, 3), dtype='complex64')
    f += np.random.rand(Nsize, Nsize, Nsize, 3).astype('float32')
    f += 1j*f

    # Use same random field, compare the output
    for i in range(3):
        t1 = time.perf_counter()
        result3 = FFTW_power(f[...,0], 1, Nsize)
        result3 += FFTW_power(f[...,1], 1, Nsize)
        result3 += FFTW_power(f[...,2], 1, Nsize)
        print(time.perf_counter() - t1)
        break 



if __name__ == '__main__':
    main()

