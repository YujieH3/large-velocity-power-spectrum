from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Each process has an array of size 10 with values equal to the rank of the process
local_array = np.full(10, rank, dtype='i')
print(f'Rank {rank} has local_array: {local_array}')

# Create an empty array on the root process to hold the result
result_array = np.empty(10, dtype='i') if rank == 0 else None

# Use Reduce to sum the arrays on all processes element-wise
comm.Reduce(local_array, result_array, op=MPI.SUM, root=0)

# Process 0 prints the result
if rank == 0:
    print('The element-wise sum of arrays on all processes is:', result_array)