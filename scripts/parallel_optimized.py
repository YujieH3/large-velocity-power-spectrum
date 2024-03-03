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
import sys # debug
import os
import datetime # benchmarking

# TODO numba acceleration of the loop? No the loop is not the bottleneck. Possibly the synchronize
# SOLVED the fftw threading bug? Turns out to be a conda package issue.
# DONE use fft_object instead of pyfftw.interfaces.numpy_fft.fftn

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

# ------------------------------------CONFIG------------------------------------
SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
NTOT = 128 # total resolution. The dynamical range would be NTOT/2, from 2pi/NTOT to pi/LCELL
MAXNBOX = 32 # maximum box size allowed by memory

LTOT = 1 # kpc

NBUFFER = 2000 # number of points to query before synchronize.
SAVEDIR = '.'

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
args = parser.parse_args()

SNAPSHOT = args.input
SAVEDIR  = args.output
NTOT     = args.ntot
MAXNBOX  = args.maxnbox
LTOT     = args.ltot
NBUFFER  = args.nbuffer

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


def vector_power(fx, fy, fz, Lbox, Nbox):
    """
    Calculate FFT and power grid before sampling. This function does the main 
    math and physics in power spectrum computation.

    Default normalization is such that
    `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
    """
    # Fourier transform
    const = (Lbox / (2 * np.pi)) ** 1.5 / Nbox ** 3
    fkx = pyfftw.interfaces.numpy_fft.fftn(fx, threads=1) * const # need to specify threads=1 or else mpi will raise an error
    fky = pyfftw.interfaces.numpy_fft.fftn(fy, threads=1) * const
    fkz = pyfftw.interfaces.numpy_fft.fftn(fz, threads=1) * const
    # Definition of velocity power spectrum
    Pk = 0.5 * (np.abs(fkx) ** 2 + np.abs(fky) ** 2 + np.abs(fkz) ** 2)
    return Pk

def FFTW_vector_power(fx, fy, fz, Lbox, Nsize):
    """
    Same with vector_power, but using the fft_object to allow full power of 
    FFTW.
    """
    # Fourier transform
    const = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3

    a = pyfftw.empty_aligned((Nbox, Nbox, Nbox), dtype='complex64')
    b = pyfftw.empty_aligned((Nbox, Nbox, Nbox), dtype='complex64')
    fft_object = pyfftw.FFTW(a, b, axes=(0,1,2)) # input is a, output is b 
    # maybe initialize f as an empty aligned array?
    
    a[:] = fx
    fft_object()
    fk = 0.5 * np.abs(b * const)**2

    a[:] = fy
    fft_object()
    fk += 0.5 * np.abs(b * const)**2

    a[:] = fz
    fft_object()
    fk += 0.5 * np.abs(b * const)**2

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
    P = Psum / Nsample # might raise warning of division by zero
    # But P is always set by Psum/Nsample and won't be used anywhere so we don't care

    Pvk = np.column_stack((bin_centers, P, Psum, Nsample))

    return Pvk


# -----------------------------------MAIN---------------------------------------
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # thread number
    NTHREADS = comm.Get_size() # number of threads

    LCELL = LTOT / NTOT  # kpc 

    # ------------------------------ALLOCATE WORK-------------------------------
    # Interpolation box assignment, with auto planner
    # NTOT = n_threads_per_axis * Nbox * n_loops
    outputfile = os.path.join(SAVEDIR, 'Pk.txt')
    
    n_loops, n_threads_per_axis, Nbox, Lbox = planner(NTOT, LTOT, MAXNBOX, NTHREADS)
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
    # if we cannot assign one chunk per core because of memory limit, we'll
    # have to make each core to do the work of, say, 8 cores, to query 8 times before
    # synchronize and assign phase and sum up. However if we also apply 8 phases
    # and sum 8 times, we will have to save 8 times the memory, which is exactly
    # what we want to avoid. It seems that there is no way around so we do only
    # one FFT each core and hence get a slightly lower spectral resolution, but
    # the same dynamical range. Or else we can do the same interpolation of 
    # 10000**3 points 8 times, splitting to 1000 cores would only take 47*8 minutes
    # actually. It's doable but not efficient. Let's see if the first option can work
    bx = r # bx, by, bz are not necessarily equal to r, s, t. and is assigned to each CPU
    by = s # can modify this later to allow for more flexible
    bz = t # task assignment.

    if rank == 0:
        print(f'[{datetime.datetime.now()}] Planner: {n_loops} loops, {m} fold, {Nbox} size each thread.', flush=True)
        print('Accept plan? (y/n)', flush=True)
        if input() != 'y':
            print('Plan rejected. Press any key to exit.', flush=True)
            sys.exit(0)
        print('Plan confirmed. Starting computation.', flush=True)
    # print(f'Rank: {rank}, r: {r}, s: {s}, t: {t}, Nbox per box: {Nbox}')

    pbar = tqdm.tqdm(total=100 * n_loops, ncols=50, bar_format='{l_bar}{bar}| [{elapsed}<{remaining}]') if rank == 0 else None

    # --------------------------------LOAD DATA---------------------------------
    print(f'[{datetime.datetime.now()}] Load snapshot: {SNAPSHOT}')

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
        mass = None
        M = None

    # --------------------------------BUILD INDEX-------------------------------
    print(f'[{datetime.datetime.now()}] Build index: {coords.shape}') # type: ignore # TODO save and load index
    ann_idx = AnnoyIndex(3, 'euclidean')  # Length of item vector that will be indexed
    for i in range(len(coords)): # Can distribute to threads. # type: ignore
        ann_idx.add_item(i, coords[i]) # type: ignore
    print(f'[{datetime.datetime.now()}] Index added')
    ann_idx.build(1, n_jobs=-1) # use all available threads? Seems to be using only 1.
    print(f'[{datetime.datetime.now()}] Tree built')

    pbar.update(5) if rank == 0 else None # type:ignore

    # sys.exit(0)
    # --------------------------------QUERY-------------------------------------
    # TODO a big loop to interpolate-FFT-combine-save and again for a new box
    # to save memory, spread task over time. Something like Nmaxbox, the biggest
    # box size that can fit into memory. Spread NTOT=m*Nbox*n_loops
    # where Nbox is the largest integer < Nmaxbox that keeps n_loops an integer.
    # So it's a factorization problem of NTOT/m.
    
    bidx = 0
    for nb1 in range(n):
        for nb2 in range(n):
            for nb3 in range(n):
                qidx = 0
                f_idx = 0 # index track the index of the first empty element in f array
                a_queue = np.empty((n_loops, n_queue, 3), dtype=np.float32)
                x_queue = np.empty((n_loops, n_queue), dtype=np.float32)
                y_queue = np.empty((n_loops, n_queue), dtype=np.float32)
                z_queue = np.empty((n_loops, n_queue), dtype=np.float32)
                f = np.empty((Nbox**3, 3), dtype=np.complex64) # folded field with phase.
                for i in range(Nbox):
                    for j in range(Nbox):
                        for k in range(Nbox):
                            iidx = 0
                            for n1 in range(n): 
                                for n2 in range(n):
                                    for n3 in range(n): # the art of for loops
                                        # TODO make a loop to query some points before synchronize
                                        # One tradeoff: more points queried at once, less communication,
                                        # less time needed, but more memory usage. Make it a tunable parameter.
                                        x = ((r+n1) * Nbox + i) * LCELL
                                        y = ((s+n2) * Nbox + j) * LCELL
                                        z = ((t+n3) * Nbox + k) * LCELL
                                        query = np.array([x, y, z], dtype=np.float32) # specific location to query, dependent on rank

                                        nb = ann_idx.get_nns_by_vector(query, n=1, search_k=-1, # type: ignore
                                                                    include_distances=False) 
                                        # print(nb) # of shape (1,) because we query only the 1 nearest neighbor, and doesn't include distance
                                        a = velocity[nb[0]] # type: ignore # extract the velocity of corresponding index
                                        
                                        a_queue[iidx, qidx] = a
                                        x_queue[iidx, qidx] = x
                                        y_queue[iidx, qidx] = y
                                        z_queue[iidx, qidx] = z

                                        iidx += 1
                            qidx += 1

                            if qidx >= n_queue or qidx + f_idx >= Nbox**3:
                                qidx = 0 # reset the queue index
                                # synchronize and gather the queue to memory of each core
                                a_arr = np.array(comm.allgather(a_queue), dtype=np.float32) # a_arr of shape [NTHREADS, n_loops, NBUFFER, 3] all_gather add an axis in the front
                                x_arr = np.array(comm.allgather(x_queue), dtype=np.float32)# or generate x_arr each cpu.
                                y_arr = np.array(comm.allgather(y_queue), dtype=np.float32)# depends on which is faster.
                                z_arr = np.array(comm.allgather(z_queue), dtype=np.float32)
                                
                                a_arr = a_arr.reshape(NTHREADS*n_loops, n_queue, 3) # last variable changing fastest.
                                x_arr = x_arr.reshape(NTHREADS*n_loops, n_queue) 
                                y_arr = y_arr.reshape(NTHREADS*n_loops, n_queue) # Tested to work
                                z_arr = z_arr.reshape(NTHREADS*n_loops, n_queue)

                                # This phase application is infact an FFT of some sort.
                                # TODO use fft here to speed things up
                                phase = np.exp(
                                    -1j * (2 * np.pi / LTOT) * ((bx+nb1) * x_arr + (by+nb2) * y_arr + (bz+nb3) * z_arr)    # x, y, z / L or nx, ny, nz / N
                                ) # phase of shape [NTHREADS, NBUFFER]

                                if f_idx + n_queue < Nbox**3:
                                    f[f_idx:f_idx+n_queue, :] = np.sum(a_arr * phase[...,None], axis=0) / m**1.5 # shape (NBUFFER, 3)
                                    f_idx += n_queue # update the index of the first empty element in f array
                                    pbar.update(NBUFFER / (Nbox**3) * 80) if rank == 0 else None # type: ignore
                                else:
                                    temp_idx = Nbox**3 - f_idx
                                    f[f_idx:, :] = np.sum(a_arr[:,:temp_idx,:] * phase[:,:temp_idx,None], axis=0) / m**1.5 # indices after leftover are from the precious loop
                                    f_idx = Nbox**3 # update to the end of the array. f_idx will not be used anymore
                                    pbar.update(temp_idx / (Nbox**3) * 80) if rank == 0 else None # type: ignore
                            else:
                                continue
                # f is constructed last index fastest, so C order can reconstruct to the 
                # correct cube. C order is the default value of reshape btw.
                f = np.reshape(f, (Nbox, Nbox, Nbox, 3), order='C') # reshape for FFT.

                # ----------------------------FFT-------------------------------
                print(f'[{datetime.datetime.now()}] FFTW: {f.shape}')

                # TODO save plan if doesn't exist, read if it exists
                P = FFTW_vector_power(f[:,:,:,0], f[:,:,:,1], f[:,:,:,2], Lbox, Nbox)
                print(f'[{datetime.datetime.now()}] FFTW finished: {P.shape}')

                pbar.update(10) if rank == 0 else None # update progress bar # type: ignore
                
                # ---------------------------SAMPLE-----------------------------

                # Sampling k in concentric spheres
                Pk = pair_power(P, Lbox, Nbox, shift=-2*np.pi*np.array([bx+nb1, by+nb2, bz+nb3])/LTOT)
                print(f'[{datetime.datetime.now()}] Pk: {Pk.shape}')

                # Sample power spectrum P(k), from the fundamental mode to the Nyquist frequency (half frequency of the smallest scale)
                Pkk = hist_sample(Pk, kmin=2*np.pi/LTOT, kmax=np.pi/LCELL, spacing=2*np.pi/LTOT) # Pkk = (k, P, Psum, Nsample) of shape (NTOT, 4)
                # Use energy spectral density of dimension ~velocity^2/k
                Pkk[:, 1] *= 4 * np.pi * Pkk[:, 0] ** 2 
                
                Pkk = np.array(Pkk, dtype=np.float32) # use float32
                p_sum = Pkk[:, 2].copy()
                n_sample = Pkk[:, 3].copy()    

                # print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')

                # ---------------------------COMBINE----------------------------
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
                    Pkk[:, 1] = Pkk[:, 2] / Pkk[:, 3] * (4 * np.pi * Pkk[:, 0] ** 2)
                    # print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')
                
                # ----------------------------SAVE------------------------------
                
                if rank == 0 and bidx == 0:
                    np.savetxt(outputfile, Pkk) # save however you want. Let's make a simple text file for now.
                    print(f'[{datetime.datetime.now()}] Saved: {outputfile}')
                elif rank == 0 and bidx > 0:
                    Pkk_tot = np.loadtxt(outputfile)
                    Pkk_tot[:, 2] += Pkk[:, 2]
                    Pkk_tot[:, 3] += Pkk[:, 3]
                    Pkk_tot[:, 1] = Pkk_tot[:, 2] / Pkk_tot[:, 3] * (4 * np.pi * Pkk_tot[:, 0] ** 2)
                    np.savetxt(outputfile, Pkk_tot)
                    print(f'[{datetime.datetime.now()}] Combined and saved: {Pkk.shape}')

                pbar.update(5) if rank == 0 else None # update progress bar # type: ignore

                bidx += 1

    pbar.close() if rank == 0 else None # type: ignore




