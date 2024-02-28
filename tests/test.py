from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Each process has an array of size 10 with values equal to the rank of the process
local_array = np.full(10, rank, dtype='i')
print(f'Rank {rank} has local_array: {local_array}')

# Create an empty array on the root process to hold the result
a = np.ones((10, 3)) * rank

# Use Reduce to sum the arrays on all processes element-wise
result = comm.allgather(a)

# Process 0 prints the result
if rank == 0:
    print('The element-wise sum of arrays on all processes is:', result, np.shape(result))

