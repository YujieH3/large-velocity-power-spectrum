from mpi4py import MPI
import h5py
import numpy as np
import math
from numba import jit, njit
import pyfftw
pyfftw.interfaces.cache.enable()

from annoy import AnnoyIndex

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

# ------------------------CONFIG------------------------
SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
NTOT = 512 # total resolution. The dynamical range would be NTOT/2, from 2pi/NTOT to pi/LCELL
LTOT = 1 # kpc
NPTS = 1000 # number of points to query before synchronize.
MAXNBOX = 256 # maximum box size allowed by memory

SAVEDIR = '.'
REMOVECACHE = True


remove_bulk_velocity = True

# -----------------------FUNCTIONS----------------------


def planner(n_total_res, l_total_length, n_box_affordable, n_total_threads):
    m = math.cbrt(n_total_threads)  # number of threads in each dimension = the folding factor
    assert m.is_integer()==True, 'Number of threads must be a cube of an integer. Support for any number is not yet implemented.'
    m = int(m)

    n_loops = 1
    Nfullbox = n_total_res / m
    assert Nfullbox.is_integer()==True, 'Divided Nbox must be an integer.'

    Nbox = Nfullbox
    while Nbox > n_box_affordable or Nbox.is_integer()==False:
        n_loops += 1
        Nbox = Nfullbox / n_loops
    Nbox = int(Nbox) # number of pixels in each dimension of the box

    Lbox = Nbox / n_total_res * l_total_length # length of the box

    return n_loops, m, Nbox, Lbox


def FFTW_vector_power(fx, fy, fz, Lbox, Nsize, fft_object):
    """
    Same with vector_power, but using the fft_object to allow full power of 
    FFTW.
    """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    # Overwrite to save memory
    fk = np.abs(fft_object(fx) * a) # fkx = fft_object(fx) * a
    fk += np.abs(fft_object(fy) * a) # fky = fft_object(fy) * a
    fk += np.abs(fft_object(fz) * a) # fkz = fft_object(fz) * a
    # Definition of velocity power spectrum
    return 0.5 * (fk) ** 2


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
    if shift[0] > 0:
        kx = kx + shift[0]
    if shift[1] > 0:
        ky = ky + shift[1]
    if shift[2] > 0:
        kz = kz + shift[2]
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
    Psum, bin_edges_ = np.histogram(
        Pk_pair[:, 0], bins=bin_edges, weights=Pk_pair[:, 1]
    )
    Nsample, bin_edges_ = np.histogram(Pk_pair[:, 0], bins=bin_edges)
    P = Psum / Nsample
    P[Nsample == 0] = 0  # Set P=0 if no sample

    Pvk = np.column_stack((bin_centers, P, Psum, Nsample))

    return Pvk


####################################   MAIN   ##################################
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # thread number
    NTHREADS = comm.Get_size() # number of threads

    LCELL = LTOT / NTOT  # kpc 

    # ------------------------------ALLOCATE WORK-------------------------------
    # Interpolation box assignment, with auto planner
    # NTOT = n_threads * Nbox * n_loops
    
    n_loops, m, Nbox, Lbox = planner(NTOT, LTOT, MAXNBOX, NTHREADS)
    
    # Box index assignment
    r = rank // m**2
    s = (rank % m**2) // m
    t = (rank % m) // 1

    # Phase factor assignment
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

    pbar = tqdm.tqdm(total=100 * n_loops) if rank == 0 else None

    # --------------------------------LOAD DATA---------------------------------
    print(f'[{datetime.datetime.now()}] Load snapshot: {SNAPSHOT}')

    f = h5py.File(SNAPSHOT, 'r')
    coords = f["PartType0/Coordinates"][:] # of shape (N, 3) # type: ignore
    mass = f["PartType0/Masses"][:] # type: ignore
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
        # free memory. python garbage collector will take care of this, but not immediately.
        mass = None
        M = None

    # --------------------------------BUILD INDEX-------------------------------
    print(f'[{datetime.datetime.now()}] Build index: {coords.shape}') # type: ignore
    ann_idx = AnnoyIndex(3, 'euclidean')  # Length of item vector that will be indexed
    for i in range(len(coords)): # Can distribute to threads. # type: ignore
        ann_idx.add_item(i, coords[i]) # type: ignore
    print(f'[{datetime.datetime.now()}] Index added')
    ann_idx.build(1, n_jobs=-1) # use all available threads? Seems to be using only 1.
    print(f'[{datetime.datetime.now()}] Tree built')

    pbar.update(5) if rank == 0 else None

    # sys.exit(0)
    # --------------------------------QUERY-------------------------------------
    # TODO a big loop to interpolate-FFT-combine-save and again for a new box
    # to save memory, spread task over time. Something like Nmaxbox, the biggest
    # box size that can fit into memory. Spread NTOT=m*Nbox*n_loops
    # where Nbox is the largest integer < Nmaxbox that keeps n_loops an integer.
    # So it's a factorization problem of NTOT/m.
    
    for lp in range(n_loops):

        queue_idx = 0
        f_idx = 0 # index track the index of the first empty element in f array
        x_queue = np.empty(NPTS, dtype=np.float32)
        y_queue = np.empty(NPTS, dtype=np.float32)
        z_queue = np.empty(NPTS, dtype=np.float32)
        a_queue = np.empty((NPTS, 3), dtype=np.float32)
        f = np.empty((Nbox**3, 3), dtype=np.complex64) # folded field with phase.
        for i in range(Nbox):
            for j in range(Nbox):
                for k in range(Nbox):
                    # TODO make a loop to query some points before synchronize
                    # One tradeoff: more points queried at once, less communication,
                    # less time needed, but more memory usage. Make it a tunable parameter.
                    x = (r * Nbox + i) * LCELL
                    y = (s * Nbox + j) * LCELL
                    z = (t * Nbox + k) * LCELL

                    query = np.array([x, y, z], dtype=np.float32) # specific location to query, dependent on rank
                    # print(f'Rank: {rank}, query: {query} of shape {query.shape}')
                    nb = ann_idx.get_nns_by_vector(query, n=1, search_k=-1, 
                                                include_distances=False) # type: ignore
                    # print(nb) # of shape (1,) because we query only the 1 nearest neighbor, and doesn't include distance
                    a = velocity[nb[0]] # type: ignore # extract the velocity of corresponding index

                    
                    a_queue[queue_idx] = a
                    x_queue[queue_idx] = x
                    y_queue[queue_idx] = y
                    z_queue[queue_idx] = z

                    queue_idx += 1

                    if queue_idx < NPTS:
                        continue
                    else:
                        queue_idx = 0 # reset the queue index

                        # synchronize and gather the queue to memory of each core
                        a_arr = np.array(comm.allgather(a_queue), dtype=np.float32) # a_arr of shape [NTHREADS, NPTS, 3] all_gather add an axis in the front
                        nx_arr = np.array(comm.allgather(x_queue), dtype=np.float32) / LCELL# or generate x_arr each cpu.
                        ny_arr = np.array(comm.allgather(y_queue), dtype=np.float32) / LCELL# depends on which is faster.
                        nz_arr = np.array(comm.allgather(z_queue), dtype=np.float32) / LCELL

                        phase = np.exp(
                            -1j * (2 * np.pi / NTOT) * (bx * nx_arr + by * ny_arr + bz * nz_arr)    
                        ) # phase of shape [NTHREADS, NPTS]

                        if f_idx + NPTS < Nbox**3:
                            f[f_idx:f_idx+NPTS, :] = np.sum(a_arr * phase[...,None], axis=0) / m**1.5 # shape (NPTS, 3)
                            f_idx += NPTS # update the index of the first empty element in f array
                            pbar.update(round(NPTS / (Nbox**3) * 80, ndigits=2)) if rank == 0 else None
                        else:
                            f[f_idx:, :] = np.sum(a_arr * phase[...,None], axis=0) / m**1.5
                            f_idx = Nbox**3 # update to the end of the array. f_idx will not be used anymore
                            pbar.update(round(len(a_arr) / (Nbox**3) * 80, ndigits=2)) if rank == 0 else None

        # f is constructed last index fastest, so C order can reconstruct to the 
        # correct cube. C order is the default value of reshape btw.
        f = np.reshape(f, (Nbox, Nbox, Nbox, 3), order='C') # reshape for FFT.

        # --------------------------------FFT-----------------------------------
        print(f'[{datetime.datetime.now()}] FFTW: {f.shape}')

        a = pyfftw.empty_aligned((Nbox, Nbox, Nbox), dtype='complex64')
        b = pyfftw.empty_aligned((Nbox, Nbox, Nbox), dtype='complex64')
        fft_object = pyfftw.builders.fftn(a, threads=1)

        P = FFTW_vector_power(f[:,:,:,0], f[:,:,:,1], f[:,:,:,2], Lbox, Nbox, fft_object)

        print(f'[{datetime.datetime.now()}] FFTW finished: {P.shape}')

        pbar.update(10) if rank == 0 else None # update progress bar
        # -------------------------------COMBINE--------------------------------

        # Sampling k in concentric spheres
        Pk = pair_power(P, Lbox, Nbox, shift=2*np.pi*np.array([bx, by, bz])/LTOT)
        print(f'[{datetime.datetime.now()}] Pk: {Pk.shape}')

        # Sample power spectrum P(k), from the fundamental mode to the Nyquist frequency (half frequency of the smallest scale)
        Pkk = hist_sample(Pk, kmin=2*np.pi/LTOT, kmax=np.pi/LCELL, spacing=2*np.pi/LTOT)
        # Use energy spectral density of dimension ~velocity^2/k
        Pkk[:, 1] *= 4 * np.pi * Pkk[:, 0] ** 2 # Pkk = (k, P, Psum, Nsample) of shape (n, 4)
        
        Pkk = np.array(Pkk, dtype=np.float32) # convert to float64 for MPI
        p_sum = Pkk[:, 2].copy()
        n_sample = Pkk[:, 3].copy()    

        print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')

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
            print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')
        
        # --------------------------------SAVE----------------------------------
        if rank == 0 and n_loops == 1:
            print(f'[{datetime.datetime.now()}] Save: {Pkk.shape}')
            np.savetxt(os.path.join(SAVEDIR, 'Pk.txt'), Pkk) # save however you want. Let's make a simple text file for now.
        elif rank == 0 and n_loops > 1:
            print(f'[{datetime.datetime.now()}] Save: {Pkk.shape}')
            np.savetxt(os.path.join(SAVEDIR, f'Pk{lp}.txt'), Pkk)

        pbar.update(4) if rank == 0 else None # update progress bar

    if rank == 0 and n_loops > 1: # Combine different loops if applicable
        for lp in range(n_loops):
            Pkk = np.loadtxt(os.path.join(SAVEDIR, f'Pk{lp}.txt'))
            if lp == 0:
                Pkk_tot = Pkk
            else:
                Pkk_tot[:, 2] += Pkk[:, 2]
                Pkk_tot[:, 3] += Pkk[:, 3]
            if REMOVECACHE:
                os.remove(os.path.join(SAVEDIR, f'Pk{lp}.txt'))
            pbar.update(1) # update progress bar
        Pkk_tot[:, 1] = Pkk_tot[:, 2] / Pkk_tot[:, 3] * (4 * np.pi * Pkk_tot[:, 0] ** 2)
        np.savetxt(os.path.join(SAVEDIR, 'Pk.txt'), Pkk_tot)

    # Close the progress bar. Sum of all updates should be 100(n_loops).
    pbar.close() if rank == 0 else None




