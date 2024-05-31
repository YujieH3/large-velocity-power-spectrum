from mpi4py import MPI
import numpy as np
import pyfftw

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    b = pyfftw.empty_aligned((size, 5), dtype='complex64')
else:
    b = None

a = pyfftw.empty_aligned((5), dtype='complex64')
a[:] = np.arange(5) + rank

comm.Gather(a, b, root=0)
print(b)
print(np.shape(b))
print(type(a))
