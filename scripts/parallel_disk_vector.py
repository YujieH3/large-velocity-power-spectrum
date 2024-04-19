"""
Through out this code the negative phase norm is used. FFT is defined as 
F(k) = ∫f(x)exp(-ikx)dx. The phase shift exp(-ixß).
"""

from mpi4py import MPI
import h5py
import numpy as np
import math
from numba import jit, njit
import pyfftw
pyfftw.interfaces.cache.enable()

from annoy import AnnoyIndex

import argparse
import tqdm # progress bar
import os

# debug
import sys # debug
import datetime # benchmarking
import gc # garbage collection
from memory_profiler import profile
import warnings
#warnings.filterwarnings("error") # so that you can catch warining by try except

# TODO numba acceleration of the loop? No the loop is not the bottleneck.

# ------------------------------------CONFIG------------------------------------
SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
# '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
NTOT = 1000 # total resolution. The dynamical range would be NTOT/2, from 2pi/NTOT to pi/LCELL
MAXNBOX = 500 # maximum box size allowed by memory

LTOT = 1 # kpc

NBUFFER = 10000 # number of points to query before synchronize.
SAVEDIR = '../output/'

remove_bulk_velocity = True
# -----------------------------------PARSE ARGS---------------------------------

parser = argparse.ArgumentParser(description="""Compute power spectrum in parallel. 
                                 The program will make a plan and ask for permission 
                                 before starting the computation.""",
                                 usage='mpiexec -n <thread numbers> python %(prog)s [options]')
parser.add_argument('-i', '--input', nargs='?',type=str, default=SNAPSHOT, help='Path to the snapshot file.')
parser.add_argument('-o', '--output', nargs='?',type=str, default=SAVEDIR, help='Directory to save the power spectrum.')
parser.add_argument('-N', '--ntot', nargs='?',type=int, default=NTOT, help='Total resolution.')
parser.add_argument('-M', '--maxnbox', nargs='?',type=int, default=MAXNBOX, help='Maximum box size allowed by memory. A planner will decide the optimal box size.')
parser.add_argument('-l', '--ltot', nargs='?',type=int, default=LTOT, help='Total length of the box.')
parser.add_argument('-b', '--nbuffer', nargs='?',type=int, default=NBUFFER, help='Number of points to query before synchronize.')
parser.add_argument('-f', action='store_true', help='Skip confirmation and start the computation.')
args = parser.parse_args()

SNAPSHOT = args.input
SAVEDIR  = args.output
NTOT     = args.ntot
MAXNBOX  = args.maxnbox
LTOT     = args.ltot
NBUFFER  = args.nbuffer
FORCE    = args.f

# -----------------------------------FUNCTIONS----------------------------------

def planner(n_total_res, l_total_length, n_box_affordable, n_total_threads):
    n_threads_per_axis = math.cbrt(n_total_threads)  # number of threads in each dimension = the folding factor
    assert n_threads_per_axis.is_integer()==True, 'Number of threads must be a cube of an integer. Support for any number is not yet implemented.'
    n_threads_per_axis = int(n_threads_per_axis)

    n_loops_per_axis = 1
    n_full_box = n_total_res / n_threads_per_axis
    assert n_full_box.is_integer()==True, 'Divided Nbox must be an integer.'

    n_box = n_full_box
    while n_box > n_box_affordable or n_box.is_integer()==False:
        n_loops_per_axis += 1
        n_box = n_full_box / n_loops_per_axis
    n_loops = n_loops_per_axis**3
    n_box = int(n_box) # number of pixels in each dimension of the box

    l_box = n_box / n_total_res * l_total_length # length of the box

    return n_loops, n_threads_per_axis, n_box, l_box


@njit(fastmath=True)
def make_xyz(x_start, y_start, z_start, Nbox, Lcell, f_idx, n_queue):
    """
    Calculate phase factor array for the coordinate range we need. Output is
    the x, y, z array of the phase factor, each a 1-D array of length n_queue.
    We use loops to fill an array instead of a meshgrid-reshape-slice approach
    to reduce memory consumption.

    Note
    ----
    In the last loop, f_idx1 + n_queue will exceed Nbox**3, which leads to a
    small waste of memory. This will not affect the result, later we will make
    slice to use only the part up to Nbox**3.
    """
    f_idx1 = 0
    queue_idx = 0
    x_queue = np.zeros(n_queue, dtype=np.float32)
    y_queue = np.zeros(n_queue, dtype=np.float32)
    z_queue = np.zeros(n_queue, dtype=np.float32)
    for i,j,k in np.ndindex(Nbox, Nbox, Nbox):
        if f_idx1 < f_idx:
            f_idx1 += 1
            continue
        elif f_idx1 >= f_idx and queue_idx < n_queue:
            x_queue[queue_idx] = x_start + i * Lcell
            y_queue[queue_idx] = y_start + j * Lcell
            z_queue[queue_idx] = z_start + k * Lcell
            queue_idx += 1
        elif queue_idx >= n_queue:
            break
        else:
            raise Exception('Unexpected behaviour in make_xyz()')

    return x_queue, y_queue, z_queue


# @profile
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


def pair_power(Pk, Lbox, Nbox, shift=np.array([0, 0, 0])):
    """
  Create pairs of power and k. Sampling in concentric spherical
  shells with PkSample to get power spectrum. Called by PowerSpec3D.
  This function is independent of the physics and definition of the
  power spectrum
  """
    # Initialize k space
    Lcell = Lbox / float(Nbox)
    kSpace = 2 * np.pi * np.fft.fftfreq(Nbox, Lcell)
    # Create k and power pairs
    kx, ky, kz = np.meshgrid(kSpace, kSpace, kSpace, indexing="ij")
    # Apply shift
    if shift[0] != 0:
        kx = kx - shift[0] # shift is relative to f, k shifted by -i is f shifted by +i.
    if shift[1] != 0:
        ky = ky - shift[1]
    if shift[2] != 0:
        kz = kz - shift[2]
    #
    k = np.sqrt(kx*kx + ky*ky + kz*kz)
    del kx, ky, kz
    # Construct a (n,2) shape array
    k = np.ravel(k)
    Pk = np.ravel(Pk)
    Pk_pair = np.column_stack((k, Pk))  # Shape (n, 2)

    return Pk_pair



def hist_sample(Pk_pair, kmin, kmax, spacing):
    """ Calculate mean power from Pk pair with specified spacing. """
    n_bins = int((kmax - kmin) / spacing) + 1  # include kmin and kmax
    bin_centers = np.linspace(kmin, kmax, n_bins)
    bin_edges = np.linspace(kmin - spacing / 2, kmax + spacing / 2, n_bins + 1)
    Psum, _ = np.histogram(Pk_pair[:, 0], bins=bin_edges, weights=Pk_pair[:, 1])
    Nsample, _ = np.histogram(Pk_pair[:, 0], bins=bin_edges)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        P = Psum / Nsample # might raise warning of division by zero
    # But P is always set by Psum/Nsample and won't be used anywhere so we don't care

    Pvk = np.column_stack((bin_centers, P, Psum, Nsample))

    return Pvk


# -----------------------------------MAIN---------------------------------------

# @profile # this slow down the program tremendously, remember to comment it out
# except when profiling.
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # thread number
    NTHREADS = comm.Get_size() # number of threads

    LCELL = LTOT / NTOT  # kpc 

    # ------------------------------ALLOCATE WORK-------------------------------
    # Interpolation box assignment, with auto planner
    # NTOT = n_threads_per_axis * Nbox * n_loops
    outputfile = os.path.join(SAVEDIR, 'Pk.txt')
    assert os.path.isdir(SAVEDIR), 'Output directory does not exist.'
    assert os.path.isfile(SNAPSHOT), 'Snapshot file does not exist.'
    TEMPDIR = os.path.join(SAVEDIR, 'temp')
    if rank == 0:
        os.makedirs(TEMPDIR, exist_ok=True) # exit_ok: do nothing if the directory exists
    
    n_loops, n_threads_per_axis, Nbox, Lbox = planner(NTOT, LTOT, MAXNBOX, NTHREADS)
    
    # n_loops is the number of boxes one core needs to interpolate and calculate phase
    n = math.cbrt(n_loops) # spectrum smoothing in each dimension
    n = int(n)

    # Seperate NBUFFER into n_queue of n_loops
    assert NBUFFER % n_loops == 0, 'NBUFFER must be divisible by n loops.'
    n_queue = NBUFFER // n_loops

    m = n_threads_per_axis * n # folding factor

    # Box index assignment
    r = n * (rank % n_threads_per_axis)
    s = n * ((rank % n_threads_per_axis**2) // n_threads_per_axis)
    t = n * (rank // n_threads_per_axis**2)

    # Phase factor assignment. Because every queried point is needed for every FFT,
    bx = r # bx, by, bz are not necessarily equal to r, s, t. and is assigned to each CPU
    by = s # can modify this later to allow for more flexible
    bz = t # task assignment.

    if rank == 0:
        print(f'[{datetime.datetime.now()}] Planner: {n_loops} loops, {m} fold, {Nbox} size each thread.', flush=True)
        print('Accept plan? (y/n)', flush=True)
        if not FORCE:
            if input() != 'y':
                print('Plan rejected. Press any key to exit.', flush=True) if rank == 0 else None
                sys.exit(0)
        print('Plan confirmed. Starting computation.', flush=True)
    
        # Print the set of key parameters used.
        print(f'Snapshot: {SNAPSHOT}', flush=True)
        print(f'Output file: {outputfile}', flush=True)
        print(f'NTOT: {NTOT}', flush=True)
        print(f'MAXNBOX: {MAXNBOX}', flush=True)
        print(f'LTOT: {LTOT}', flush=True)
        print(f'NBUFFER: {NBUFFER}', flush=True)    
    comm.Barrier() # synchronize all threads before starting the computation

    # print(f'Rank: {rank}, r: {r}, s: {s}, t: {t}, Nbox per box: {Nbox}')

    # ------------------------------PROGRESS BAR--------------------------------

    pbar = tqdm.tqdm(total=100*n_loops*3, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]') if rank == 0 else None

    # --------------------------------LOAD DATA---------------------------------
    #print(f'[{datetime.datetime.now()}] Load snapshot: {SNAPSHOT}')

    f = h5py.File(SNAPSHOT, 'r')
    coords   = f["PartType0/Coordinates"][:] # of shape (N, 3) # type: ignore
    mass     = f["PartType0/Masses"][:] # type: ignore
    velocity = f["PartType0/Velocities"][:] # type: ignore
    f.close()

    # Shift to origin. We can update to allow any origin and any shape in the future,
    # should the need ever arise.
    coords[:, 0] -= np.min(coords[:, 0]) # type: ignore
    coords[:, 1] -= np.min(coords[:, 1]) # type: ignore
    coords[:, 2] -= np.min(coords[:, 2]) # type: ignore

    # Remove bulk velocity
    if remove_bulk_velocity:
        M = np.sum(mass) # type: ignore
        velocity[:, 0] -= np.sum(mass * velocity[:, 0]) / M # type: ignore
        velocity[:, 1] -= np.sum(mass * velocity[:, 1]) / M # type: ignore
        velocity[:, 2] -= np.sum(mass * velocity[:, 2]) / M # type: ignore
        # free memory. python garbage collector will take care of this after.
        del mass, M

    # --------------------------------BUILD INDEX-------------------------------
    #print(f'[{datetime.datetime.now()}] Build index: {coords.shape}') # type: ignore # TODO save and load index
    
    indexfile = os.path.join(SAVEDIR, 'index.ann')
    if os.path.isfile(indexfile):
        ann_idx = AnnoyIndex(3, 'euclidean')
        ann_idx.load(indexfile) # prefault: load index to memory, could this resolve the speed issue?
        #print(f'[{datetime.datetime.now()}] Index loaded')
    else:
        ann_idx = AnnoyIndex(3, 'euclidean')  # Length of item vector that will be indexed
        for i in range(len(coords)): # Can distribute to threads. # type: ignore
            ann_idx.add_item(i, coords[i]) # type: ignore
        # print(f'[{datetime.datetime.now()}] Index added')
        ann_idx.build(1, n_jobs=-1) # use all available threads? Seems to be using only 1.
        if rank == 0 and os.path.isfile(indexfile)==False:
            ann_idx.save(indexfile)
        #print(f'[{datetime.datetime.now()}] Index built')
    pbar.update(5) if rank == 0 else None # type:ignore

    # sys.exit(0)
    # ------------------------------QUERY AND SAVE------------------------------
    
    # Interpolation
    for n1, n2, n3 in np.ndindex(n, n, n):
        query_idx_arr = np.empty((Nbox**3), dtype=np.int32)
        for i, j, k in np.ndindex(Nbox, Nbox, Nbox):

            # One tradeoff: more points queried at once, less communication,
            # less time needed, but more memory usage. Make it a tunable parameter.
            x = ((r+n1) * Nbox + i) * LCELL
            y = ((s+n2) * Nbox + j) * LCELL
            z = ((t+n3) * Nbox + k) * LCELL

            # Query the nearest neighbor
            query = np.array([x, y, z], dtype=np.float32) # specific location to query, dependent on rank

            # nb is of shape (1,) because we query only the 1 nearest neighbor, and toggled include_distances to False
            nb = ann_idx.get_nns_by_vector(query, n=1, search_k=-1, include_distances=False) # type: ignore
            query_idx = nb[0] # type: ignore # extract the velocity of corresponding index

            # Store it in 1d, easier for later slicing
            query_idx_arr[i*Nbox**2 + j*Nbox + k] = query_idx 
        # Save to hdf5. Right now it's unnecessary to use paralle h5py.
        box_file = os.path.join(TEMPDIR, f'query_idx_nx{r+n1}_ny{s+n2}_nz{t+n3}.hdf5')

        # name the files according to the pixel coordinate of their first element
        with h5py.File(box_file, 'w') as hf:
            hf.create_dataset('query_index', data=query_idx_arr, dtype=np.int32)

    # -----------------------------PHASE ASSIGNMENT-----------------------------
    # loop over vx, vy, vz component
    
    # bidx keeps track of current loop
    bidx = 0

    for nb1, nb2, nb3 in np.ndindex(n, n, n):
        # f_idx is the index of the current element in f array
        # and also the index of the current element in each box
        f_idx = 0

        # Memory allocation for queue and f array
        f = np.empty((Nbox**3), dtype=np.complex64) # folded field with phase.
        v_queue = np.empty((n_loops, n_queue, 3), dtype=np.float32)
        x_queue = np.empty((n_loops, n_queue), dtype=np.float32)
        y_queue = np.empty((n_loops, n_queue), dtype=np.float32)
        z_queue = np.empty((n_loops, n_queue), dtype=np.float32)
        while f_idx < Nbox**3:
            # n_idx is the idx of the current loop, number of boxes covered each core 
            n_idx = 0

            for n1, n2, n3 in np.ndindex(n, n, n):
                # Each file will be read 3*n_loop**2 times
                box_file = os.path.join(TEMPDIR, f'query_idx_nx{r+n1}_ny{s+n2}_nz{t+n3}.hdf5')
                with h5py.File(box_file, 'r') as hf:
                    # Read only what we need using hdf5 slice.
                    if f_idx + n_queue < Nbox**3:
                        i_arr = hf['query_index'][f_idx:f_idx+n_queue] # type:ignore
                    else:
                        i_arr = hf['query_index'][f_idx:] # type:ignore
                
                # 1-D array of velocity of the queried points
                v_queue[n_idx, :len(i_arr), :] = velocity[i_arr, :] # type:ignore
                
                # Coordinates in the sequence as it's filled up. 
                x_start = (r+n1) * Nbox * LCELL
                y_start = (s+n2) * Nbox * LCELL
                z_start = (t+n3) * Nbox * LCELL
                x_queue0, y_queue0, z_queue0 = make_xyz(x_start, y_start, z_start, Nbox, LCELL, f_idx, n_queue)
                
                x_queue[n_idx, :] = x_queue0
                y_queue[n_idx, :] = y_queue0
                z_queue[n_idx, :] = z_queue0

                n_idx += 1

            # synchronize and gather the queue to memory of each core
            v_gathered = np.array(comm.allgather(v_queue), dtype=np.float32) # a_arr of shape [NTHREADS, n_loops, NBUFFER, 3] all_gather add an axis in the front
            x_gathered = np.array(comm.allgather(x_queue), dtype=np.float32)
            y_gathered = np.array(comm.allgather(y_queue), dtype=np.float32)
            z_gathered = np.array(comm.allgather(z_queue), dtype=np.float32)

            # Reshape. The 0 axis is the axis we want to sum over
            v_gathered = v_gathered.reshape(NTHREADS*n_loops, n_queue, 3) # last variable changing fastest.
            x_gathered = x_gathered.reshape(NTHREADS*n_loops, n_queue)
            y_gathered = y_gathered.reshape(NTHREADS*n_loops, n_queue)
            z_gathered = z_gathered.reshape(NTHREADS*n_loops, n_queue)

            # This phase application is in fact an FFT of some sort.
            # TODO use fft here to speed things up?

            # Phase factor array. In shape (NTHREADS*n_loops, n_queue).
            phase = np.exp(
                -1j * (2 * np.pi / LTOT) * (
                    (bx+nb1) * x_gathered + 
                    (by+nb2) * y_gathered + 
                    (bz+nb3) * z_gathered
                    )    # x, y, z / L or nx, ny, nz / N
                )

            # Apply phase factor and append to our final fft array
            if f_idx + n_queue < Nbox**3:
                f[f_idx:f_idx+n_queue] = np.sum(v_gathered * phase[...,None], axis=0) / m**1.5 # shape (NBUFFER, 3)
                f_idx += n_queue # update the index of the first empty element in f array

                # Update progress bar
                pbar.update(n_queue / (Nbox**3) * 80) if rank == 0 else None # type: ignore
            else:
                temp_idx = Nbox**3 - f_idx
                f[f_idx:] = np.sum(v_gathered[:,:temp_idx] * phase[:,:temp_idx,None], axis=0) / m**1.5 # indices after leftover are from the precious loop
                f_idx = Nbox**3 # update to the end of the array. f_idx will not be used anymore

                # Update progress bar
                pbar.update(temp_idx / (Nbox**3) * 80) if rank == 0 else None # type: ignore

            
                    
        # f is constructed last index fastest, so C order can reconstruct to the 
        # correct cube. C order is the default value of reshape btw.
        f = np.reshape(f, (Nbox, Nbox, Nbox), order='C') # reshape for FFT.

        # --------------------------------FFT-----------------------------------
        #print(f'[{datetime.datetime.now()}] FFTW: {f.shape}')

        # Try to keep max memory usage to 4 Nbox**3 arrays.
        fx = f[:,:,:,0]
        fy = f[:,:,:,1]
        fz = f[:,:,:,2]
        del f

        # TODO save plan if doesn't exist, read if it exists
        P = FFTW_power(fx, Lbox, Nbox)
        del fx
        P += FFTW_power(fy, Lbox, Nbox)
        del fy
        P += FFTW_power(fz, Lbox, Nbox)
        del fz

        #print(f'[{datetime.datetime.now()}] FFTW finished: {P.shape}')

        pbar.update(10) if rank == 0 else None # update progress bar # type: ignore
        
        # -------------------------------SAMPLE---------------------------------

        # Sampling k in concentric spheres
        Pk = pair_power(P, Lbox, Nbox, shift=-2*np.pi*np.array([bx+nb1, by+nb2, bz+nb3])/LTOT)
        # print(f'[{datetime.datetime.now()}] Pk: {Pk.shape}')
        del P

        # Sample power spectrum P(k), from the fundamental mode to the Nyquist frequency (half frequency of the smallest scale)
        Pkk = hist_sample(Pk, kmin=2*np.pi/LTOT, kmax=np.pi/LCELL, spacing=2*np.pi/LTOT)
        # Pkk = (k, P, Psum, Nsample) of shape (NTOT, 4)
        del Pk
        
        # Use energy spectral density of dimension ~velocity^2/k
        Pkk[:, 1] *= 4 * np.pi * Pkk[:, 0] ** 2 

        Pkk = np.array(Pkk, dtype=np.float32) # use float32
        p_sum = Pkk[:, 2].copy()
        n_sample = Pkk[:, 3].copy()    

        # print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')

        # ----------------------COMBINE DIFFERENT THREADS-----------------------
        if rank == 0: # allocate memory for the total sum. It couldn't work on Pkk[:, 2] directly
            n_pk = len(Pkk)
            p_sum_tot = np.empty(n_pk, dtype=np.float32)
            n_sample_tot = np.empty(n_pk, dtype=np.float32)
        else:
            p_sum_tot = None
            n_sample_tot = None

        comm.Reduce(sendbuf=p_sum, recvbuf=p_sum_tot, op=MPI.SUM, root=0) # sum up all power spectra to rank 0
        comm.Reduce(sendbuf=n_sample, recvbuf=n_sample_tot, op=MPI.SUM, root=0)
        
        if rank == 0:
            Pkk[:, 2] = p_sum_tot
            Pkk[:, 3] = n_sample_tot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                Pkk[:, 1] = Pkk[:, 2] / Pkk[:, 3] * (4 * np.pi * Pkk[:, 0] ** 2)
            # print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')
        
        # ------------------SAVE AND COMBINE DIFFERENT LOOPS--------------------
        
        if rank == 0 and bidx == 0:

            # Create the file. Overwrite if exists.
            np.savetxt(outputfile, Pkk)
            # print(f'[{datetime.datetime.now()}] Saved: {outputfile}')
        elif rank == 0:

            # Weighted combination
            Pkk_tot = np.loadtxt(outputfile)
            Pkk_tot[:, 2] += Pkk[:, 2] # Psum
            Pkk_tot[:, 3] += Pkk[:, 3] # Nsamples

            # Pk will be overwrite every save, all nans will be removed in the end. No need to flag warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in divide")
                Pkk_tot[:, 1] = Pkk_tot[:, 2] / Pkk_tot[:, 3] * (4 * np.pi * Pkk_tot[:, 0] ** 2)
            
            # Save
            np.savetxt(outputfile, Pkk_tot)
            # print(f'[{datetime.datetime.now()}] Combined and saved: {Pkk.shape}')

        pbar.update(5) if rank == 0 else None # update progress bar # type: ignore

        bidx += 1 # beta index for the big loop

        gc.collect() # garbage collection

    # ------------------------------CLEAN UP-----------------------------------
    pbar.close() if rank == 0 else None # type: ignore

    return 0

if __name__ == '__main__':
    if main()==0:
        print(f'[{datetime.datetime.now()}] Computation finished.')



"""
From pyFFTW documentation:
--------------------------------------------------------------------------------
The first call for a given transform size and shape and dtype and so on may be 
slow, this is down to FFTW needing to plan the transform for the first time. 
Once this has been done, subsequent equivalent transforms during the same 
session are much faster. It's possible to export and save the internal knowledge 
(the wisdom) about how the transform is done. This is described below.

Even after the first transform of a given specification has been performed, 
subsequent transforms are never as fast as using pyfftw.FFTW objects directly, 
and in many cases are substantially slower. This is because of the internal 
overhead of creating a new pyfftw.FFTW object on every call. For this reason, 
a cache is provided, which is recommended to be used whenever pyfftw.interfaces 
is used. Turn the cache on using pyfftw.interfaces.cache.enable(). This function 
turns the cache on globally. Note that using the cache invokes the threading     # threading module, maybe it's this?
module.

The cache temporarily stores a copy of any interim pyfftw.FFTW objects that are 
created. If they are not used for some period of time, which can be set with 
pyfftw.interfaces.cache.set_keepalive_time(), then they are removed from the 
cache (liberating any associated memory). The default keepalive time is 0.1 
seconds.
"""