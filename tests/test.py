from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

from tqdm import tqdm
import time

# Create a progress bar with a total of 100\
pbar = tqdm(total=100) if rank == 0 else None

for i in range(10):
    # Do some work
    time.sleep(0.1)
    # Update the progress bar
    pbar.update(0.5) if rank == 0 else None

# Close the progress bar
pbar.close() if rank == 0 else None

