# python -m memory_profiler fftw_mem_profile.py


from memory_profiler import profile
import pyfftw
import numpy as np

import time
import gc

@profile
def fftw_power_copy(f, Lbox, Nsize):
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
    
    #a[:] = f
    f = fft_object(f) # the result is stored in b
    f = 0.5 * np.abs(f * const)**2

    return f


@profile
def fftw_vector_power_copy(fx, fy, fz, Lbox, Nsize):
    """
    same with vector_power, but using the fft_object to allow full power of 
    fftw.
    """
    # fourier transform
    const = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3

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



    # The old, standard version.
    Nsize = 512
    np.random.seed(1)
    f = np.empty((Nsize, Nsize, Nsize, 3), dtype='complex64')
    f += np.random.rand(Nsize, Nsize, Nsize, 3).astype('float32')
    f += 1j*np.random.rand(Nsize, Nsize, Nsize, 3).astype('float32')

    # Use same random field, compare the output
    for i in range(3):
        t1 = time.perf_counter()
        P =  fftw_power_copy(f[...,0], 1, Nsize)
        P += fftw_power_copy(f[...,1], 1, Nsize)
        P += fftw_power_copy(f[...,2], 1, Nsize)
        print(time.perf_counter() - t1)
        break


    print('\n')


    # Do it in the main function
    Nsize = 512
    np.random.seed(1)
    f = np.empty((Nsize, Nsize, Nsize, 3), dtype='complex64')
    f += np.random.rand(Nsize, Nsize, Nsize, 3).astype('float32')
    f += 1j*np.random.rand(Nsize, Nsize, Nsize, 3).astype('float32')

    # Use same random field, compare the output
    t1 = time.perf_counter()
    
    # Fourier transform
    const = (1. / (2 * np.pi)) ** 1.5 / Nsize ** 3

    #a = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    #b = pyfftw.empty_aligned((Nsize, Nsize, Nsize), dtype='complex64')
    #fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b 
    # maybe initialize f as an empty aligned array?

    fx = f[..., 0]
    fy = f[..., 1]
    fz = f[..., 2]
    del f

    #fx = pyfftw.byte_align(fx)
    fx = pyfftw.interfaces.numpy_fft.fftn(fx, overwrite_input=True)
    P1 = 0.5 * np.abs(fx * const)**2

    #fy = pyfftw.byte_align(fy)
    fy = pyfftw.interfaces.numpy_fft.fftn(fy, overwrite_input=True)
    P1 += 0.5 * np.abs(fy * const)**2

    #fz = pyfftw.byte_align(fz)
    fz = pyfftw.interfaces.numpy_fft.fftn(fz, overwrite_input=True)
    P1 += 0.5 * np.abs(fz * const)**2

    
    print(time.perf_counter() - t1)


    #print(P)
    #print(P1)
    # Check if they produce the same results
    print('The two methods gives the same results:', np.allclose(P, P1))



    






if __name__ == '__main__':
    main()

