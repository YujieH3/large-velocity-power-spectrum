#----------------------------------------------------------------------
#		File:			interp.py
#		Programmer:		Yujie He
#		Last modified:	22/09/23                # dd/mm/yy
#		Description:	Functions for interpolation.
#----------------------------------------------------------------------
#
#       This file contains functions related to interpolation.
#
#       Functions and classes starting with '_' are for intrinsic use 
#       only. 
#
#----------------------------------------------------------------------

# For I/O
import pickle
import h5py
import os
import shutil

# For error/warning control
from sys import exit
import warnings

# For computation
import subprocess
import numpy as np
import pyfftw

pyfftw.interfaces.cache.enable()
from vpower.spctrm import PowerSpectrum

# For plotting
import matplotlib.pyplot as plt
# plt.style.use("niceplot2jay.mplstyle")

from matplotlib.colors import LogNorm

# For performance measurement
import time
from memory_profiler import profile


def init_dir(run_output_dir, auto_overwrite=False):
    # Prepare output folder
    if os.path.exists(run_output_dir) is False:
        os.mkdir(run_output_dir)
    else:
        print("Warning: output folder already exists.")
        if auto_overwrite is True:
            overwrite = True
        else:
            print("Overwrite? (y/n): ", end="")
            if input() == "y":
                overwrite = True
            else:
                overwrite = False

        if overwrite is True:
            print("Overwriting...")
            shutil.rmtree(run_output_dir)
            os.mkdir(run_output_dir)
        else:
            print("Exiting...")
            exit()
    print("Output folder of this run: {}".format(run_output_dir))
    return run_output_dir


class SimulationParticles:
    def __init__(self, pos, mass, density, velocity, Lbox) -> None:
        self.pos = pos                     # coordinate data of shape [Nparticles, 3]
        self.mass = mass                   # mass data of shape [Nparticles]
        self.density = density             # density data of shape [Nparticles]
        self.v = self.velocity = velocity  # velocity data of shape [Nparticles, 3]
        self.Lbox = Lbox                   # box size is given in creation
        self.r = self.h()                  # radii is smoothing length with rate = 1.0

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, index):
        return SimulationParticles(
            self.pos[index],
            self.mass[index],
            self.density[index],
            self.v[index],
            self.Lbox,
        )

    def data(self) -> tuple:
        return self.pos, self.mass, self.density, self.v

    def shift_to_origin(self) -> None:                     # Shift coordinates to begin at (0,0,0)
        xmin = np.min(self.pos[:, 0])
        ymin = np.min(self.pos[:, 1])
        zmin = np.min(self.pos[:, 2])
        self.pos[:, 0] -= xmin
        self.pos[:, 1] -= ymin
        self.pos[:, 2] -= zmin

    def remove_bulk_velocity(self) -> None:                # - center of mass velocity
        M = np.sum(self.mass)
        self.v[:, 0] -= np.sum(self.mass * self.v[:, 0]) / M
        self.v[:, 1] -= np.sum(self.mass * self.v[:, 1]) / M
        self.v[:, 2] -= np.sum(self.mass * self.v[:, 2]) / M

    def rho(self, smoothing_rate=1.0):                     # fixed mass, decrease rho, increase V -> larger particles
        rho = (self.density / smoothing_rate ** 3)         # Change smoothing length while keeping mass constant
        return rho

    def h(self, smoothing_rate=1.0):                      
        rho = (self.density / smoothing_rate ** 3)         # Change smoothing length while keeping mass constant
        V = self.mass / rho                                # volume = mass / density
        h = ((3 * V) / (4 * np.pi)) ** (
            1 / 3
        )                                                  # Smoothing length h = particle radius * smooth_rate
        return h

    def interp_to_field(
        self,
        Nsize,
        eps=0.0,
        auto_padding=False,
        data_file="data.pts",
        query_file="query.pts",
        output_file="ann_output.save",
    ):
        """ Interpolate velocity using a 3d histogram + ANN. 
    
    Note
    ----


    """
        t0 = time.perf_counter()
        print("Interpolating velocity field...")

        Lcell = self.Lbox / Nsize
        if auto_padding is True:
            Lpad = np.max(
                (np.max(self.pos - self.Lbox),      # 3 components : padding length of 3 upper boundaries
                 np.max(0 - self.pos))              # 3 components : padding length of 3 lower boundaries
                )                                   # maximize over 6 values corresponds to 6 sides in the box
            Lpadded = self.Lbox + 2 * Lpad
            Npadded = Nsize + 2 * int(Lpad / Lcell) # Use padded length (Lpadded) or resolution (Npadded) in histogram creation, but not in query points
            pos_padded = (self.pos + Lpad)          # move to the center and avoid negative coordinates for deposit_to_grid()
            
            print(
                "Padding complete. Padded box length: {},"
                " Padded box size: {}".format(Lpadded, Npadded)
            )
        else:
            Lpadded = self.Lbox
            Npadded = Nsize
            pos_padded = self.pos
            print("Box length: {}, box size: {}".format(Lpadded, Npadded))

        t1 = time.perf_counter() # TEMPORARY RUNTIME COUNTER

        vec = np.stack(                            # vec = [vx*m, vy*m, vz*m, m]
            (                                      
                self.v[:, 0] * self.mass,       # vectorize as much as possible
                self.v[:, 1] * self.mass,
                self.v[:, 2] * self.mass,
                self.mass,
            ),
            axis=1,
        )                                          # shape of vec is (number of particles, 4)

        vec_histgrid = deposit_to_grid(
            f=vec, pos=pos_padded, Nsize=Npadded, Lbox=Lpadded
        )  # ISSUE: POSSIBLE OVERBOUND
        vec_hist_pts = np.reshape(vec_histgrid, (Npadded ** 3, 4))  # directly reshape to (Npadded**3, 4) would produce wrong results
        vec_hist_pts /= (self.Lbox / Nsize)**3
                # now vec = [vx*rho, vy*rho, vz*rho, rho] 

        filter = tuple([vec_hist_pts[:, 3] > 0])   # this filter selects non-zero density
        vec_hist_pts = vec_hist_pts[filter]        # get the rho-v vectors for non-empty cells for deposition after ANN
        data_pos = make_grid_coords(Lbox=Lpadded, Nsize=Npadded)
        data_pos = data_pos[filter]                # select coordinates of non-empty cells for ANN

        if auto_padding is True:
            data_pos -= Lpad                       

        t = time.perf_counter() - t1 # TEMPORARY RUNTIME COUNTER
        print("Histogram deposition done. Time taken: {:.2f} s".format(t))

        vec_grid = ann_interpolate(                # now run ANN to find the nearest neighbor for each query cell
            data_pos=data_pos,                     # and link the rho-v vectors to the query cells
            f=vec_hist_pts,
            Lbox=self.Lbox,                        # grid is created within the function
            Nsize=Nsize,
            eps=eps,
            query_pos=None,
            data_file=data_file, # AVOID R/W BY ADDING PYTHON BINDING TO ANN
            query_file=query_file,
            output_file=output_file,
        )

        v_grid, m_grid = _vec_to_vm_grid(vec_grid=vec_grid, Lcell=Lcell)  # translate rho-v vectors to velocity and mass fields. or density fields if future required

        t = time.perf_counter() - t0 # TEMPORARY RUNTIME COUNTER
        print("Interpolation done. Time elapsed: {:.2f} s".format(t))

        simField3D = SimulationField3D(v_grid, m_grid, Lbox=self.Lbox, Nsize=Nsize) # create a SimulationField3D object to store the interpolated field

        return simField3D

    def interp_to_blocks(
        self, run_output_dir, nblocks, Nblock, eps, smoothing_rate=1.0
    ):
        """
    Interpolate velocity using Voxelize by v = (m*v)_i/m_i to `nblocks`^3 blocks
    (`BlockField3D`) of size `Nblock`^3 each and save them to a subfolder of
    output_dir.

    Parameters
    ----------
    output_dir : str
      Path to the output folder.
    nblocks : int
      Number of blocks in each direction.
    Nblock : int
      Size of each block in each direction.
    smoothing_rate : float
      `Smoothing length h = particle radius * smoothing_rate`. This parameter
      only affects the padding length and have no effect on the interpolation
      which is based on nearest neighbor.

    Returns
    -------
    blocksDecomp : BlocksDecomposition
      A BlocksDecomposition object containing the information of the blocks.
    
    Notes
    -----
    The output folder will be named as `[run_output_dir]Ng[Nblock*nblocks]Nb[Nblock]` 
    where Ng is the number of total grid points combined, Nb is the size of each 
    block.
    """

        # initialize a Blocks object
        Lblock = self.Lbox / nblocks
        blocksDecomp = BlocksDecomposition(run_output_dir, nblocks, Nblock, Lblock)

        blocksDecomp.run_output_dir = run_output_dir  # update data_folder attribute

        pos = self.pos
        h = self.h(smoothing_rate=smoothing_rate)
        for r in range(nblocks):                                # 0, 1, 2, ..., Nblock-1
            for s in range(nblocks):                            # Nblock * Lbox_blk = Lbox
                for t in range(nblocks):

                    selection = (                               # Potential problem: particles on the corner just outside the box with
                        (pos[:, 0] + h >= r * Lblock)           # no part in the block might be included.
                        & (pos[:, 0] - h < (r + 1) * Lblock)
                        & (pos[:, 1] + h >= s * Lblock)
                        & (pos[:, 1] - h < (s + 1) * Lblock)
                        & (pos[:, 2] + h >= t * Lblock)
                        & (pos[:, 2] - h < (t + 1) * Lblock)
                    )

                    blockParticles = self[selection]
                    blockParticles.Lbox = Lblock
                    blockParticles.pos[:, 0] -= r * Lblock      # Shift origin to the lower left corner of the block (not including margin)
                    blockParticles.pos[:, 1] -= s * Lblock
                    blockParticles.pos[:, 2] -= t * Lblock

                    subField = blockParticles.interp_to_field(
                        Nsize=Nblock, eps=eps, auto_padding=True
                    )                                           # return a padded and trimmed field.

                    blockField = blocksDecomp._BlockField3D(    # Create a BlockField3D object and save for later use
                        v=subField.get_v(),
                        mass=subField.mass,
                        Lblock=Lblock,
                        Nblock=Nblock,
                        r=r,
                        s=s,
                        t=t,
                    )
                    blockField.save_field(
                        run_output_dir
                    )                                           # save to run_output_dir/block_field_posXXX.pkl

        return blocksDecomp

    @staticmethod
    def load_snapshot(file, Lbox=1.0, remove_bulk_velocity=True, shift_to_origin=True):
        """ Load coordinates, density, masses and velocities from a snapshot to a particles
    object.

    Parameters
    ----------
    file : str
      The HD5F snapshot file name

    Returns
    -------
    simParticles : SimulationParticles
      The Particles object containing the snapshot data. See Particles class for
      more details. 

    Examples
    --------
    ```
    import utils_data as dt
    file = '/test/snapshot.hd5f'
    simParticles = dt.LoadSnapshot(file)
    pos = simParticles.pos
    mass = simParticles.mass
    density = simParticles.density
    v = simParticles.v
    ```
    """

        f = h5py.File(file, "r")
        coordinates = f["PartType0"]["Coordinates"][:]  # type:ignore
        masses = f["PartType0"]["Masses"][:]            # type:ignore
        density = f["PartType0"]["Density"][:]          # type:ignore
        velocities = f["PartType0"]["Velocities"][:]    # type:ignore
        f.close()

        simParticles = SimulationParticles(
            coordinates, masses, density, velocities, Lbox=Lbox
        )

        if remove_bulk_velocity is True:
            simParticles.remove_bulk_velocity()
        if shift_to_origin is True:
            simParticles.shift_to_origin()

        return simParticles

    def total_mass(self) -> float:
        """ Compute the total mass of the particles.
    """
        return np.sum(self.mass)

    def total_momentum(self) -> np.ndarray:
        """ Compute the total momentum of the particles.
    """
        px = np.sum(self.mass * self.v[:, 0])
        py = np.sum(self.mass * self.v[:, 1])
        pz = np.sum(self.mass * self.v[:, 2])
        return np.array([px, py, pz])

    def total_kinetic_energy(self) -> float:
        """ Compute the total kinetic energy of the particles.
    """
        return 0.5 * np.sum(
            self.mass * (self.v[:, 0] ** 2 + self.v[:, 1] ** 2 + self.v[:, 2] ** 2)
        )


class SimulationField3D:
    def __init__(self, v, mass, Lbox, Nsize) -> None:
        
        # Essential information for power spectrum computation
        
        self.Lbox = Lbox                          # box length in length unit
        self.Nsize = Nsize                        # box size in number of cells (resolution)
        self.Lcell = Lbox / float(Nsize)          # cell length in length unit
        
        # Main data
        
        self.vx = v[:, :, :, 0]                   # stores a velocity field
        self.vy = v[:, :, :, 1]
        self.vz = v[:, :, :, 2]
        self.mass = mass                          # and a density field

    def get_v(self) -> np.ndarray:                # use a method instead of attribute to avoid unnecessary memory usage
        v = np.stack((self.vx, self.vy, self.vz), axis=3)
        return v

    def get_density(self) -> np.ndarray:          
        return self.mass / self.Lcell ** 3        # use a method instead of attribute to avoid unnecessary memory usage

    def get_data(self) -> np.ndarray:
        data = np.stack((self.vx, self.vy, self.vz, self.mass), axis=3)
        return data

    def velocity_power(
        self,
    ) -> np.ndarray:  ### future update: write this outside the class
        # print_log to check conservation
        print(
            "Specific kinetic energy before FFT: {:.2e}".format(
                0.5 * np.mean(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
            )
        )
        # Calculate FFT and power
        P = _vector_power(self.vx, self.vy, self.vz, self.Lbox, self.Nsize)
        # print_log to check conservation
        print(
            "Specific kinetic energy after FFT: {:.2e}".format(
                np.sum(P) * (2 * np.pi / self.Lbox) ** 3
            )
        )
        return P

    def momentum_power(self) -> np.ndarray:
        # momentum grid
        px = self.vx * self.mass
        py = self.vx * self.mass
        pz = self.vx * self.mass
        # print_log to check conservation
        print(
            "Conserved quantity before FFT: {:.2e}".format(
                0.5 * np.mean(px ** 2 + py ** 2 + pz ** 2)
            )
        )
        # Calculate FFT and power
        P = _vector_power(px, py, pz, self.Lbox, self.Nsize)
        # print_log to check conservation
        print(
            "Conserved quantity after FFT: {:.2e}".format(
                np.sum(P) * (2 * np.pi / self.Lbox) ** 3
            )
        )

        return P

    def kinetic_energy_power(self) -> np.ndarray:
        # kinetic energy grid
        E = self.mass * (self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        # print_log to check conservation
        print("Conserved quantity before FFT: {:.2e}".format(0.5 * np.mean(E ** 2)))
        # Calculate FFT and power
        P = _scalar_power(E, self.Lbox, self.Nsize)
        # print_log to check conservation
        print(
            "Conserved quantity after FFT: {:.2e}".format(
                np.sum(P) * (2 * np.pi / self.Lbox) ** 3
            )
        )
        return P

    def spctrm(
        self, quantity="velocity", kmin=None, kmax=None, kres=None
    ) -> PowerSpectrum:
        """ Calculate a power spectrum from a given 3D velocity field. """
        # Initialize k range if not specified
        if kmin is None:
            kmin = 2 * np.pi / self.Lbox  # from pixel freq
        if kmax is None:
            kmax = np.pi / self.Lcell  # to Nyquist freq
        if kres is None:
            kres = kmin

        # Calculate FFT and power
        if quantity == "velocity":
            P = self.velocity_power()
        elif quantity == "momentum":
            P = self.momentum_power()
        elif quantity == "energy":
            P = self.kinetic_energy_power()
        else:
            raise Exception(
                """Unrecognized physical quantity name.
        Supported: 'velocity', 'momentum', 'energy'."""
            )

        # Sampling k in concentric spheres
        Pk_pair = _pair_power(P, self.Lbox, self.Nsize)
        # Sample power spectrum P(k)
        Pkk = _hist_sample(Pk_pair, kmin=kmin, kmax=kmax, spacing=kres)
        # Use energy spectral density of dimension ~v^2/k
        Pkk[:, 1] *= 4 * np.pi * Pkk[:, 0] ** 2
        spctrm = PowerSpectrum(Pkk)

        print("Conserved quantity after sampling: {:.2e}".format(spctrm.energy()))

        return spctrm

    def fold(self, m, beta, quantity="velocity"):  # m is the folding factor
        phase = _get_phase(
            beta, totalNsize=self.Nsize, Nphase=self.Nsize, x0=0, y0=0, z0=0
        )
        if quantity == "velocity":
            phi = _apply_phase(self.get_v(), phase)
            phi = fold_field(phi, m)
            phi /= m ** 1.5
            return FoldedField3D(phi, m, beta, self.Lbox / m, self.Nsize // m)
        else:
            raise Exception("""Unsupported physical quantity name.""")

    def trim(self, Nmargin, Nblock) -> None:
        n1 = Nmargin
        n2 = Nmargin + Nblock
        self.vx = self.vx[n1:n2, n1:n2, n1:n2]
        self.vy = self.vy[n1:n2, n1:n2, n1:n2]
        self.vz = self.vz[n1:n2, n1:n2, n1:n2]
        self.mass = self.mass[n1:n2, n1:n2, n1:n2]
        # update params
        self.Nsize = self.Nsize - 2 * Nmargin
        self.Lbox = self.Lbox * Nblock / (Nblock + 2 * Nmargin)

    def down_sample(self, n) -> None:
        """ Down sample mass and velocity fields. """
        new_px = down_sample(self.vx * self.mass, n)
        new_py = down_sample(self.vy * self.mass, n)
        new_pz = down_sample(self.vz * self.mass, n)
        self.mass = down_sample(self.mass, n)
        self.mass[np.where(self.mass == 0)] = 1e-10  # avoid zero mass
        # update velocity
        self.vx = new_px / self.mass
        self.vy = new_py / self.mass
        self.vz = new_pz / self.mass
        # update params
        self.Nsize /= n
        self.Lcell *= n

    def mean_kinetic_energy(self) -> float:
        E = 0.5 * np.mean(self.mass * (self.vx ** 2 + self.vy ** 2 + self.vz ** 2))
        return E

    def specific_kinetic_energy(self) -> float:
        E = 0.5 * np.mean(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        return E

    def total_kinetic_energy(self) -> float:
        E = 0.5 * np.sum(self.mass * (self.vx ** 2 + self.vy ** 2 + self.vz ** 2))
        return E

    def total_mass(self) -> float:
        """ Compute the total mass of the field.
    """
        return np.sum(self.mass)

    def total_momentum(self) -> np.ndarray:
        """ Compute the total momentum of the field.
    """
        px = np.sum(self.mass * self.vx)
        py = np.sum(self.mass * self.vy)
        pz = np.sum(self.mass * self.vz)
        return np.array([px, py, pz])

    def plot_density_slice(self, index, axis=2, ax=None, **kwargs):
        """ Plot a slice of density field. Convert to nHcgs units.
    """

        # Get the density slice
        if axis == 0:
            density_slice_nHcgs = self.get_density()[index, :, :] * 300
        elif axis == 1:
            density_slice_nHcgs = self.get_density()[:, index, :] * 300
        elif axis == 2:
            density_slice_nHcgs = self.get_density()[:, :, index] * 300
        else:
            raise Exception(
                """Unrecognized axis index.
        Supported: 0, 1, 2."""
            )

        # Plot
        plot_density2d(
            density_slice_nHcgs, Lbox=self.Lbox, Nsize=self.Nsize, ax=ax, **kwargs
        )

    def plot_velocity_slice(self, component, index, axis=2, ax=None, **kwargs):
        """ Plot a slice of velocity field. 
    """

        if component == 0:
            vi = self.vx
        elif component == 1:
            vi = self.vy
        elif component == 2:
            vi = self.vz
        else:
            raise Exception(
                """Unrecognized component.
        Supported: 1, 2, 3."""
            )

        if axis == 0:
            velocity_slice = vi[index, :, :]
        elif axis == 1:
            velocity_slice = vi[:, index, :]
        elif axis == 2:
            velocity_slice = vi[:, :, index]
        else:
            raise Exception(
                """Unrecognized axis index.
        Supported: 0, 1, 2."""
            )

        # Plot
        plot_velocity2d(
            velocity_slice, Lbox=self.Lbox, Nsize=self.Nsize, ax=ax, **kwargs
        )


def plot_density2d(
    density_slice_nHcgs, Lbox, Nsize, ax=None, vmin=0.1, vmax=1e3, **kwargs
):
    """ Plot a 2d density field.
  """

    # Create coordinate grid for pcolormesh
    xgrid = np.linspace(0, Lbox, Nsize)
    ygrid = np.linspace(0, Lbox, Nsize)
    X, Y = np.meshgrid(xgrid, ygrid)

    # Plot the density slice with unit label.
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    p = ax.pcolormesh(
        X, Y, density_slice_nHcgs, norm=LogNorm(vmin=vmin, vmax=vmax), **kwargs
    )
    ax.set_aspect("equal")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    plt.colorbar(p, label=r"$n_H$ $(\rm cm^{-3})$", ax=ax)


def plot_velocity2d(velocity_slice, Lbox, Nsize, ax=None, **kwargs):
    """ Plot a 2d velocity field of one component.
  """

    # Create coordinate grid for pcolormesh
    xgrid = np.linspace(0, Lbox, Nsize)
    ygrid = np.linspace(0, Lbox, Nsize)
    X, Y = np.meshgrid(xgrid, ygrid)

    # Plot the velocity slice with unit label.
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    p = ax.pcolormesh(X, Y, velocity_slice, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("X (kpc)")
    ax.set_ylabel("Y (kpc)")
    plt.colorbar(p, label=r"$v \, (\rm km\,s^{-1})$", ax=ax)


class FoldedField3D:
    def __init__(self, f, m, beta, Lbox, Nsize) -> None:
        # data
        self.f = f
        # field information
        self.Lbox = Lbox
        self.Nsize = Nsize
        self.Lcell = Lbox / Nsize
        # folding information
        self.m = m
        self.beta = beta
        self.totalLbox = Lbox * m

    def fold_spctrm(
        self, fft_object, beta=np.array([0, 0, 0]), kmin=None, kmax=None, kres=None
    ) -> PowerSpectrum:
        """ Calculate a power spectrum from a given 3D velocity field. """
        # Initialize k range if not specified
        if kmin is None:
            kmin = 2 * np.pi / self.totalLbox  # from pixel freq
        if kmax is None:
            kmax = np.pi / self.Lcell  # to Nyquist freq
        if kres is None:
            kres = kmin

        # Calculate FFT and power
        if len(self.f.shape) == 4:
            self.f = _FFTW_vector_power(
                self.f, self.Lbox, self.Nsize, fft_object
            )  # replace f to reduce memory usage
        elif len(self.f.shape) == 3:
            self.f = _FFTW_scalar_power(self.f, self.Lbox, self.Nsize, fft_object)
        else:
            raise Exception(
                """Unrecognized field shape.
        Supported: (N, N, N, 3), (N, N, N)."""
            )

        # Sampling k in concentric spheres
        Pk_pair = _pair_power(
            self.f, self.Lbox, self.Nsize, shift=2 * np.pi * beta / self.totalLbox
        )
        # Sample power spectrum P(k)
        Pkk = _hist_sample(Pk_pair, kmin=kmin, kmax=kmax, spacing=kres)
        # Use energy spectral density of dimension ~v^2/k
        Pkk[:, 1] *= 4 * np.pi * Pkk[:, 0] ** 2
        # Create the output spectrum object
        pwrSpctrm = PowerSpectrum(Pkk, m=self.m, beta=beta)
        type(pwrSpctrm) is PowerSpectrum  # Mistery: why is this necessary?
        return pwrSpctrm

    def save(self, run_output_dir) -> None:
        """Save the `FoldedField3D` object under `run_output_dir` using pickle."""
        filename = os.path.join(
            run_output_dir, "folded_field_b{}{}{}.pkl".format(*self.beta)
        )  # e.g. 'folded_field_b000.pkl'
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(run_output_dir, beta):
        """Load the saved `FoldedField3D` object from `run_output_dir` using pickle."""
        filename = os.path.join(
            run_output_dir, "folded_field_b{}{}{}.pkl".format(*beta)
        )
        with open(filename, "rb") as file:
            return pickle.load(file)


class BlocksDecomposition:
    def __init__(self, data_folder, nblocks, Nblock, Lblock) -> None:
        # data folder
        self.run_output_dir = data_folder
        # block params
        self.nblocks = nblocks
        self.Nblock = Nblock
        self.Lblock = Lblock
        self.totalNsize = Nblock * nblocks
        self.totalLbox = Lblock * nblocks

    class _BlockField3D(SimulationField3D):
        def __init__(self, v, mass, Lblock, Nblock, r, s, t) -> None:
            super().__init__(v, mass, Lbox=Lblock, Nsize=Nblock)
            self.r = r
            self.s = s
            self.t = t

        def fold(
            self, m, beta, totalNsize, quantity="velocity"
        ):  # m is the folding factor
            phase = _get_phase(
                beta,
                totalNsize=totalNsize,
                Nphase=self.Nsize,
                x0=self.r * self.Nsize,
                y0=self.s * self.Nsize,
                z0=self.t * self.Nsize,
            )
            if quantity == "velocity":
                phi = _apply_phase(self.get_v(), phase)
                phi = fold_field(phi, m)
                return FoldedField3D(phi, m, beta, self.Lbox / m, self.Nsize // m)
            else:
                raise Exception("""Unsupported physical quantity name.""")

        def save_field(self, run_output_dir) -> None:
            loc = (self.r, self.s, self.t)
            vvvm = np.stack((self.vx, self.vy, self.vz, self.mass), axis=3)
            filename = os.path.join(
                run_output_dir, "block_field_loc{}{}{}.npy".format(*loc)
            )
            np.save(filename, vvvm)

    def __getitem__(self, loc) -> _BlockField3D:
        r, s, t = loc
        filename = os.path.join(
            self.run_output_dir, "block_field_loc{}{}{}.npy".format(*loc)
        )
        vvvm = np.load(filename)
        v = vvvm[:, :, :, 0:3]
        mass = vvvm[:, :, :, 3]

        field = self._BlockField3D(
            v=v, mass=mass, Lblock=self.Lblock, Nblock=self.Nblock, r=r, s=s, t=t
        )
        return field

    def fold(self, m, beta, quantity="velocity", Nresult=None) -> FoldedField3D:
        # if Nresult is not specified, the result will have size totalNsize//m
        if Nresult == None:
            Nresult = self.totalNsize // m
            n = 1
        else:
            n = (self.totalNsize // m) // Nresult
            if n == 0:
                raise Exception(
                    "The folded box size totalNsize/m must be a multiple of Nresult."
                )
        # Initialize folded field
        f = np.zeros((Nresult, Nresult, Nresult, 3))
        f = np.array(f, dtype=np.complex128)  # prepare for phase
        # initialize a empty folded field object
        foldedField3D = FoldedField3D(
            f=f, m=m, beta=beta, Lbox=self.totalLbox / m, Nsize=self.totalNsize // m
        )
        for r in range(self.nblocks):
            for s in range(self.nblocks):
                for t in range(self.nblocks):
                    block_field = self[r, s, t]
                    if n > 1:
                        block_field.down_sample(
                            n=n
                        )  # block_field have size Nblock//n now

                    if m >= self.nblocks:  # fold stitch
                        newfoldedField3D = block_field.fold(
                            m=m // self.nblocks,
                            beta=beta,
                            totalNsize=self.totalNsize // n,
                            quantity=quantity,
                        )
                        # m//nblocks is the folding factor of each block
                        foldedField3D.f = (
                            foldedField3D.f + newfoldedField3D.f
                        )  # add to folded

                    elif m < self.nblocks:  # stitch fold
                        u = self.nblocks // m  # Each field is composed of u^3 files
                        phase = _get_phase(
                            beta=beta,
                            totalNsize=self.totalNsize // n,
                            Nphase=block_field.Nsize,
                            x0=r * block_field.Nsize,
                            y0=s * block_field.Nsize,
                            z0=t * block_field.Nsize,
                        )
                        f = _apply_phase(f=block_field.get_v(), phase=phase)
                        foldedField3D.f[
                            (r % u) * Nresult // u : (r % u + 1) * Nresult // u,
                            (s % u) * Nresult // u : (s % u + 1) * Nresult // u,
                            (t % u) * Nresult // u : (t % u + 1) * Nresult // u,
                            :,
                        ] += f

        # before folding, P(k) = (Lbox/2*pi)^3/Nsize^6 V(k)
        # after folding, P'(k) = (Lbox/m/2*pi)^3/(Nsize/m)^6 V(k)
        # P'(k) = m^3 P(k)
        # folding regions, the power spec increase by a factor of m^3
        # field needs to decrease by sqrt(m^3) to keep the same normalization
        foldedField3D.f /= m ** 1.5

        return foldedField3D

    def save(self) -> None:
        """Save the `BlocksDecomposition` object under `run_output_dir` using pickle."""
        filename = os.path.join(self.run_output_dir, "blocks_decomp.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(run_output_dir):
        """Load the saved `BlocksDecomposition` object from `run_output_dir` using pickle."""
        filename = os.path.join(run_output_dir, "blocks_decomp.pkl")
        with open(filename, "rb") as file:
            return pickle.load(file)


def _vec_to_vm_grid(vec_grid, Lcell):
    """ This function restore mass and velocity from vec_grid. Used internally in
  interp_field.
  """
    # Extract mass from vec_grid
    rho_grid = vec_grid[:, :, :, 3]
    m_grid = rho_grid * Lcell ** 3

    # Avoid divide by zero before extracting velocity from vec_grid.
    # Whereever rho_grid is 0, v_grid is 0. So this shouldn't cause trouble.
    # m_grid[m_grid == 0] = 1
    # This is faster than using v_grid = np.nan_to_num(v_grid) after dividing by
    # rho_grid. But the numpy filtering is still taking performance. Best case
    # scenario is to avoid this line.

    # v = (rho*v)_i/rho_i
    vec_grid[:, :, :, 0] /= rho_grid
    vec_grid[:, :, :, 1] /= rho_grid
    vec_grid[:, :, :, 2] /= rho_grid

    # Return velocity and mass grids
    return vec_grid[:, :, :, 0:3], m_grid


def _vector_power(fx, fy, fz, Lbox, Nsize):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fkx = pyfftw.interfaces.numpy_fft.fftn(fx) * a
    fky = pyfftw.interfaces.numpy_fft.fftn(fy) * a
    fkz = pyfftw.interfaces.numpy_fft.fftn(fz) * a
    # Definition of velocity power spectrum
    Pk = 0.5 * (np.abs(fkx) ** 2 + np.abs(fky) ** 2 + np.abs(fkz) ** 2)
    return Pk


def _FFTW_vector_power(f, Lbox, Nsize, fft_object):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation. The fft_object should works on
  one component of the vector field at a time.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
  """

    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fk = fft_object(f) * a
    # Definition of velocity power spectrum
    Pk = 0.5 * np.sum(np.abs(fk) ** 2, axis=3)
    return Pk


def _scalar_power(f, Lbox, Nsize):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `np.mean(scalar**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fk = pyfftw.interfaces.numpy_fft.fftn(f) * a
    # Power spectrum
    Pk = 0.5 * np.abs(fk) ** 2
    return Pk


def _FFTW_scalar_power(f, Lbox, Nsize, fft_object):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `np.mean(scalar**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fk = fft_object(f) * a
    # Power spectrum
    Pk = 0.5 * np.abs(fk) ** 2
    return Pk


def _pair_power(Pk, Lbox, Nsize, shift=np.array([0, 0, 0])):
    """
  Create pairs of power and k. Sampling in concentric spherical
  shells with PkSample to get power spectrum. Called by PowerSpec3D.
  This function is independent of the physics and definition of the
  power spectrum
  """
    # Initialize k space
    Lcell = Lbox / float(Nsize)
    kSpace = 2 * np.pi * np.fft.fftfreq(Nsize, Lcell)
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
    k = np.sqrt(kx * kx + ky * ky + kz * kz)
    # Construct a (n,2) shape array
    k = np.ravel(k)
    Pk = np.ravel(Pk)
    Pk_pair = np.stack((k, Pk))  # Shape (2,n)
    Pk_pair = np.transpose(Pk_pair)  # Shape (n,2)

    return Pk_pair


def _hist_sample(Pk_pair, kmin, kmax, spacing):
    """ Calculate mean power from Pk pair with specified spacing. """
    bin_centers = np.arange(kmin, kmax + spacing, spacing)
    bin_edges = np.arange(kmin - spacing / 2, kmax + 3 * spacing / 2, spacing)
    Psum, bin_edges_ = np.histogram(
        Pk_pair[:, 0], bins=bin_edges, weights=Pk_pair[:, 1]
    )
    Nsample, bin_edges_ = np.histogram(Pk_pair[:, 0], bins=bin_edges)
    P = Psum / Nsample
    P[Nsample == 0] = 0  # Set P=0 if no sample
    Pvk = np.column_stack((bin_centers, P, Psum, Nsample))

    return Pvk


def high_pass_filter_2d(field, Lbox, low_k=None):  # not very useful
    """
  Remove the k below low_k end of the given image, supposing the low-k end is located at 
  the center of the image. Note that in practice of numpy.fft package, the low-k
  end is at the corner while high-k is in the middle. Should apply 
  np.fft.fftshift() before using this funciton.

  if low_k is not specified, we will take low_k = 2 pi / Lcell.
  """
    dk = 2 * np.pi / Lbox
    Nsize = len(field)
    if low_k is None:
        Lcell = Nsize / Lbox
        low_k = 2 * np.pi / Lcell
    pixel_rad = low_k // dk
    grid = np.arange(0, Nsize)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    x_ctr = x - Nsize // 2
    y_ctr = y - Nsize // 2
    filter = x_ctr ** 2 + y_ctr ** 2 <= pixel_rad ** 2
    field[filter] = 0
    return field


def DFT(f, pos, kSpace):  # for test purpose only
    """
  Calculate the direct particle-wise discrete Fourier transform given k space.
  F_k = \sum_\alpha f_\alpha \exp{-i k \cdot x_\alpha}

  In practice, this is approximated by deposit to grid and an FFT. This function
  is used for evaluation of this approximation.
  """
    N = len(kSpace)
    count = 0
    total = N ** 3
    F = np.zeros((N, N, N), dtype=np.complex128)
    for n in range(N):
        for l in range(N):
            for m in range(N):
                kx = complex(kSpace[n])
                ky = complex(kSpace[l])
                kz = complex(kSpace[m])
                F_k = np.sum(
                    f * np.exp(-1j * (kx * pos[:, 0] + ky * pos[:, 1] + kz * pos[:, 2]))
                )
                F[n, l, m] = F_k
                count += 1
            print("{:.2%}".format(count / total))
    return F


def deposit_to_grid(
    f, pos, Nsize, Lbox
):  # future update: incorporate with Particles class
    """
  Deposit some physical quantity f to a uniform grid. For each particle, add f of
  the particle to the cell in which the particle is located. Consider each particle
  as a point. For particle located exactly at the edge of a box, use periodic 
  boundary condition.
  """
    if len(f.shape) == 1:
        f_grid = np.zeros((Nsize, Nsize, Nsize))
    else:
        f_grid = np.zeros((Nsize, Nsize, Nsize, f.shape[1]))

    Lcell = Lbox / float(Nsize)
    index = np.array((pos // Lcell) % Nsize, dtype=int)
    index = np.transpose(index)
    np.add.at(f_grid, tuple(index), f)

    return f_grid


def ann_interpolate(
    data_pos,
    f,
    Lbox,
    Nsize,
    eps,
    query_pos=None,
    data_file="data.pts",
    query_file="query.pts",
    output_file="ann_output.save",
    overwrite=True,
):

    t0 = time.perf_counter()
    # Prepare data.pts
    save_ann_pts(data_pos, file=data_file)
    print("Data file saved. Time taken: {:.2f} s".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    # Prepare query.pts if not existed
    if overwrite is True:  # set overwrite true when testing.
        if query_pos is None:
            query_pos = make_grid_coords(Lbox=Lbox, Nsize=Nsize)
        save_ann_pts(
            query_pos, file=query_file
        )  # prepare the grids to be interpolated, save as
        print("Query file saved. "
              "Time taken: {:.2f} s".format(time.perf_counter() - t0))
    else:
        print("Query file found.")

    # Run approximate nearest neighbor
    maxpts = len(data_pos)  
    # set maximum number of data points to the exact number of data points
    ann_run(eps=eps, maxpts=maxpts)  # call the ann library through command line

    # Read the ANN output and deposit the grids
    t0 = time.perf_counter()
    f_grid = read_ann_to_grid(f, Nsize, file=output_file)
    print("ANN output read. Time taken: {:.2f} s".format(time.perf_counter() - t0))

    return f_grid


def save_ann_data_pts(data_pos, file="data.pts") -> None:
    """
  Prepare data.pts for ANN from simulation data. The data points are the 
  coordinates of all particles.
  """
    np.savetxt(file, data_pos, delimiter="\t", fmt="%.16f")  # .16 is probably too much.


def make_grid_coords(Lbox, Nsize) -> np.ndarray:
    # Create coordinate grid
    Lcell = Lbox / Nsize
    xSpace = np.linspace(Lcell / 2, Lbox + Lcell / 2, Nsize)
    grid = np.meshgrid(xSpace, xSpace, xSpace, indexing="ij")
    # Mock particle coordinates
    grid_pos = np.reshape(
        grid, (3, Nsize ** 3)
    ).T  # shape: (Nsize**3, 3) but reshape to (Nsize**3, 3) won't give correct result
    return grid_pos


def save_ann_query_pts(query_pos, file="query.pts") -> None:
    """
  Prepare query.pts for ANN. The query points are the center of each cell in the
  grid.
  """
    # Prepare query.pts
    np.savetxt(file, query_pos, delimiter="\t", fmt="%.16f")


def save_ann_pts(pos, file) -> None:
    """
  """
    np.savetxt(file, pos, delimiter="\t", fmt="%.16f")


def ann_run(
    eps,
    maxpts,
    k=1,
    data_file="data.pts",
    query_file="query.pts",
    output_file="ann_output.save",
):
    """
  Run ANN by executing command lines.
  `ann_sample [-d dim] [-max mpts] [-nn k] [-e eps] [-df data] [-qf query]`
  """
    # ANN_PATH = '/appalachia/d6/yujie/ann_1.1.2/sample'

    # Compile my_ann_sample.cpp
    ret = subprocess.run(
        [
            "g++ /appalachia/d6/yujie/Test_PowerSpec/ann/ann_sample.cpp"
            " -o /appalachia/d6/yujie/Test_PowerSpec/ann/ann_sample"
            " -I/appalachia/d6/yujie/ann_1.1.2/include"
            " -L/appalachia/d6/yujie/ann_1.1.2/lib -lANN"
        ],
        shell=True,
    )
    if ret.returncode != 0:
        raise Exception("Error occurred when compling ANN.")

    # Time
    t0 = time.perf_counter()

    # Run ANN
    ret = subprocess.run(
        [
            "time /appalachia/d6/yujie/Test_PowerSpec/ann/ann_sample"
            " -e {} -max {} -nn {}"
            " -df {} -qf {} > {}".format(
                eps, maxpts, k, data_file, query_file, output_file
            )
        ],
        shell=True,
    )
    if ret.returncode != 0:
        raise Exception("Error occurred when running ANN.")

    # Time
    t1 = time.perf_counter()
    t = t1 - t0
    print("Approximate Nearest Neighbour complete. Time taken: {:.2f} s.".format(t))


def read_ann_to_grid(f, Nsize, file):
    """
  Read the approximate nearest neighbor output to interpolate some quantity 
  (e.g. mass, velocity, etc.) at data points to query points.

  The output.save is organized in the following form:
  first column: number of nearest neighbor, 0~k-1
  second column: index of the data point
  third column: distance between the data point and the query point
  
  save is of shape (query points number*k, 3)
  e.g.
  [[ 0.    5.    0.249455]
  [ 1.    4.    0.26852 ]
  [ 0.    0.    0.332847]
  [ 1.     15.    0.35775 ]]
  """
    ann_save = np.loadtxt(file, delimiter="\t")    # Read data from output.save
    index = np.array(ann_save[:, 1], dtype=int)

    if np.ndim(f) == 1:                            # scalar field
        data_grid = f[index]                       # correlate data to query positions
        data_grid = np.reshape(data_grid, (Nsize, Nsize, Nsize))
    elif np.ndim(f) == 2:                          # vector field
        data_grid = f[index, :]                    # correlate data to query positions
        data_grid = np.reshape(data_grid, (Nsize, Nsize, Nsize, f.shape[1]))
    else:
        raise Exception("Unsupported data shape.")

    return data_grid


def fold_particles(pos, m):  # future update: incorporate with Particles class
    """
  Takes in coordinate and the folding factor m and output 
  the folded coordinates.

  Parameters
  ----------
    pos : coordinates
    f : some physical quantity (can be vector)
    m : folding factor

  Returns
  -------
    pos_fold : folded coordinates
  """
    # Initialize
    xmin, xmax = np.min(pos[:, 0]), np.max(pos[:, 0])
    ymin, ymax = np.min(pos[:, 1]), np.max(pos[:, 1])
    zmin, zmax = np.min(pos[:, 2]), np.max(pos[:, 2])
    # Get L (for each dimension, so this should work also for non-cubical box)
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    # Get r that starts from 0,0,0
    r_x, r_y, r_z = pos[:, 0] - xmin, pos[:, 1] - ymin, pos[:, 2] - zmin
    # Fold the coordinates
    pos_fold = np.zeros(np.shape(pos))  # Create the zero array for folded coordinates
    pos_fold[:, 0] = r_x % (Lx / m) + xmin
    pos_fold[:, 1] = r_y % (Ly / m) + ymin
    pos_fold[:, 2] = r_z % (Lz / m) + zmin

    return pos_fold


def _apply_phase(f, phase) -> np.ndarray:
    phi = np.array(f, dtype=np.complex128)  ### Update : complex64?
    if phi.shape == phase.shape:
        phi *= phase
    else:
        phi[:, :, :, 0] *= phase
        phi[:, :, :, 1] *= phase
        phi[:, :, :, 2] *= phase

    return phi


def _get_phase(beta, totalNsize, Nphase, x0, y0, z0) -> np.ndarray:
    Nblock = Nphase
    x = np.arange(x0, x0 + Nblock)
    y = np.arange(y0, y0 + Nblock)
    z = np.arange(z0, z0 + Nblock)
    xxx, yyy, zzz = np.meshgrid(x, y, z, indexing="ij")
    phase = np.exp(
        -1j * (2 * np.pi / totalNsize) * (beta[0] * xxx + beta[1] * yyy + beta[2] * zzz)
    )
    return phase


def fold_field(f, m):
    if m == 1:                              # m==1 means no folding
        return f

    nx = f.shape[0]
    ny = f.shape[1]
    nz = f.shape[2]
    nx1 = nx // m
    ny1 = ny // m
    nz1 = nz // m

    r = 0.0
    for i in range(m):
        for j in range(m):
            for k in range(m):
                r = (
                    r
                    + f[
                        i * nx1 : (i + 1) * nx1,
                        j * ny1 : (j + 1) * ny1,
                        k * nz1 : (k + 1) * nz1,
                        :,
                    ]
                )

    return r


def down_sample(r, n):
    if n == 1:                  # n==1 means no downsampling
        return r

    d = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                d = d + r[i::n, j::n, k::n, :]
    
    d /= n ** 3                 # for smoothing density field, needs to average the cell densities
    return d


def check_conservation(simParticles, simField3D) -> None:
    """Check mass, momentum, energy conservation before and after interpolation.
  """

    mass_0 = simParticles.total_mass()                  # mass
    mass_interpolated = simField3D.total_mass()
    print("Total mass of particles: {:.3e}".format(mass_0))
    print("Total mass after interpolation: {:.3e}".format(mass_interpolated))
    print("Total mass restored by {:.3%}".format(
        mass_interpolated / mass_0                      # +/- means over/underestimation
        )
    ) 
    print("\n")
    momentum_0 = simParticles.total_momentum()          # momentum
    momentum_interpolated = simField3D.total_momentum()
    print("Total momentum of particles:", momentum_0)   # momentum is a vector 
    print("Total momentum after interpolation:", momentum_interpolated)
    print("Total momentum restored by ({:.3%}, {:.3%}, {:.3%})".format(
        momentum_interpolated[0] / momentum_0[0],       # relative difference of each component
        momentum_interpolated[1] / momentum_0[1],
        momentum_interpolated[2] / momentum_0[2]
        )
    )
    print("\n")
    energy_0 = simParticles.total_kinetic_energy()      # kinetic energy 
    energy_interpolated = simField3D.total_kinetic_energy()
    print("Total kinetic energy of particles: {:.3e}".format(energy_0))
    print("Total kinetic energy after interpolation: {:.3e}".format(
        energy_interpolated
        )
    )
    print("Total kinetic energy restored by {:.3%}".format(
        energy_interpolated / energy_0
        )
    )
