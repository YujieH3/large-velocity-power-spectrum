from mpi4py import MPI
import h5py
import numpy as np
from numba import njit
import pyfftw
pyfftw.interfaces.cache.enable()

from annoy import AnnoyIndex

import sys # debug
import datetime # benchmarking

# TODO
# numba acceleration of the loop
# use fft_object instead of pyfftw.interfaces.numpy_fft.fftn

# ------------------------CONFIG------------------------
SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
NTOT = 512 # total resolution. The dynamical range would be 256, from 2pi/NTOT to pi/LCELL
LTOT = 1 # kpc
remove_bulk_velocity = True

# -----------------------FUNCTIONS----------------------

def vector_power(fx, fy, fz, Lbox, Nbox):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nbox ** 3
    fkx = pyfftw.interfaces.numpy_fft.fftn(fx, threads=1) * a # need to specify threads=1 or else mpi will raise an error
    fky = pyfftw.interfaces.numpy_fft.fftn(fy, threads=1) * a
    fkz = pyfftw.interfaces.numpy_fft.fftn(fz, threads=1) * a
    # Definition of velocity power spectrum
    Pk = 0.5 * (np.abs(fkx) ** 2 + np.abs(fky) ** 2 + np.abs(fkz) ** 2)
    return Pk


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


################################################################################
if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # thread number
    THREADS = comm.Get_size() # number of threads

    LCELL = LTOT / NTOT  # kpc 

    # -------------------ALLOCATE WORK---------------------
    m = THREADS**(1/3)  # number of threads in each dimension, and the folding factor
    assert m.is_integer(), 'Number of threads must be a cube of an integer, for now.'
    r = rank // m**2
    s = (rank % m**2) // m
    t = (rank % m) // 1
    Nbox = NTOT / m
    Lbox  = LTOT / m
    assert Nbox.is_integer(), 'Divided Nbox must be an integer.'
    Nbox = int(Nbox)

    # use the core number and a total resolution only to define the spilitting scheme?
    print(f'Rank: {rank}, r: {r}, s: {s}, t: {t}, Nbox per box: {Nbox}')


    # ----------------------LOAD DATA-----------------------
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

    # ----------------------BUILD INDEX---------------------
    print(f'[{datetime.datetime.now()}] Build index: {coords.shape}') # type: ignore
    ann_idx = AnnoyIndex(3, 'euclidean')  # Length of item vector that will be indexed
    for i in range(len(coords)): # Can distribute to threads. # type: ignore
        ann_idx.add_item(i, coords[i]) # type: ignore
    print(f'[{datetime.datetime.now()}] Index added')
    ann_idx.build(1, n_jobs=-1) # use all available threads? Seems to be using only 1.
    print(f'[{datetime.datetime.now()}] Tree built')

    # sys.exit(0)
    # ----------------------QUERY---------------------------

    f = np.empty((Nbox, Nbox, Nbox, 3), dtype=np.complex64) # vector field only for now
    for i in range(Nbox):
        for j in range(Nbox):
            for k in range(Nbox):
                x = (r * Nbox + i) * LCELL
                y = (s * Nbox + j) * LCELL
                z = (t * Nbox + k) * LCELL
                query = np.array([x, y, z]) # specific location to query, dependent on rank
                # print(f'Rank: {rank}, query: {query} of shape {query.shape}')

                nb = ann_idx.get_nns_by_vector(query, n=1, search_k=-1, include_distances=False) # type: ignore
                # print(nb) # of shape (1,) because we query only the 1 nearest neighbor

                a = velocity[nb[0]] # type: ignore
                # use numba acceleration later

                comm.Barrier() # forced synchronization
                a_arr = comm.allgather(a) # array of size [THREADS, 3]
                x_arr = comm.allgather(x) # or generate x_arr each cpu.
                y_arr = comm.allgather(y) # depends on which is faster.
                z_arr = comm.allgather(z) 

                a_arr = np.array(a_arr) # allgather returns a list, convert to array.
                nx_arr = np.array(x_arr) * LCELL
                ny_arr = np.array(y_arr) * LCELL
                nz_arr = np.array(z_arr) * LCELL

                bx = r # bx, by, bz are not necessarily equal to r, s, t. 
                by = s # can modify this later to allow for more flexible
                bz = t # task assignment.
                phase = np.exp(
                    -1j * (2 * np.pi / NTOT) * (bx * nx_arr + by * ny_arr + bz * nz_arr)    
                )

                f[i,j,k,:] = np.sum(a_arr * phase[:,None], axis=0) / m**1.5


    # ----------------------FFT-----------------------------
    print(f'[{datetime.datetime.now()}] FFTW: {f.shape}')

    P = vector_power(f[..., 0], f[..., 1], f[..., 2], Lbox, Nbox)

    print(f'[{datetime.datetime.now()}] FFTW finished: {P.shape}')

    # ---------------------COMBINE--------------------------

    # Sampling k in concentric spheres
    Pk = pair_power(P, Lbox, Nbox, shift=2*np.pi*np.array([bx, by, bz])/LTOT)
    print(f'[{datetime.datetime.now()}] Pk: {Pk.shape}')

    # Sample power spectrum P(k), from the fundamental mode to the Nyquist frequency (half frequency of the smallest scale)
    Pkk = hist_sample(Pk, kmin=2*np.pi/LTOT, kmax=np.pi/LCELL, spacing=2*np.pi/LTOT)
    # Use energy spectral density of dimension ~velocity^2/k
    Pkk[:, 1] *= 4 * np.pi * Pkk[:, 0] ** 2 # Pkk = (k, P, Psum, Nsample) of shape (n, 4)
    
    Pkk = np.array(Pkk, dtype=np.float64) # convert to float64 for MPI
    p_sum = Pkk[:, 2].copy()
    n_sample = Pkk[:, 3].copy()    

    print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')

    if rank == 0:
        n_pk = len(Pkk)
        p_sum_tot = np.zeros(n_pk, dtype=np.float64)
        n_sample_tot = np.zeros(n_pk, dtype=np.float64)
    else:
        p_sum_tot = None
        n_sample_tot = None

    comm.Reduce(sendbuf=p_sum, recvbuf=p_sum_tot, op=MPI.SUM, root=0) # reduce to rank 0
    comm.Reduce(sendbuf=n_sample, recvbuf=n_sample_tot, op=MPI.SUM, root=0)
    
    if rank == 0:
        Pkk[:, 2] = p_sum_tot
        Pkk[:, 3] = n_sample_tot
        Pkk[:, 1] = Pkk[:, 2] / Pkk[:, 3] * (4 * np.pi * Pkk[:, 0] ** 2)
        print(f'[{datetime.datetime.now()}] Pkk: {Pkk.shape}')
    
    # ----------------------SAVE----------------------------
    if rank == 0:
        print(f'[{datetime.datetime.now()}] Save: {Pkk.shape}')
        np.savetxt('Pk.txt', Pkk)



