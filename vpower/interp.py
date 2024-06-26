#--------------------------------------------------------
# File:			interp.py
# Programmer:		Yujie He
# Last modified:	29/10/23                # dd/mm/yy
# Description:	Functions for interpolation.
#--------------------------------------------------------
#
# This file contains functions related to interpolation 
# and also FFT computation.
# 
# This script includes the following classes
#  - GasParticles
#  - BoxField
#  - FoldedBox
#  - BrickInventory
#
# Functions and classes starting with '_' are for 
# intrinsic use only.
#
#--------------------------------------------------------

# For I/O
import pickle
import h5py
import os
import shutil

# For error/warning control
from sys import exit

# For computation
import subprocess
import numpy as np
import pyfftw
# from numba import njit

import pyann
from voxelize import Voxelize
Voxelize.__init__(self=Voxelize, use_gpu=False, network_dir=None)  # type: ignore

pyfftw.interfaces.cache.enable()
from spctrm import PowerSpectrum

# For plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# plt.style.use("niceplot2jay.mplstyle")

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




def load_snapshot(file, Lbox=1.0, 
                  remove_bulk_velocity=True, 
                  shift_to_origin=True):
    """ 
    Load coordinates, density, masses and velocities from a snapshot to a particles
    object.

    Parameters
    ----------
    file : str
      The HD5F snapshot file name

    Returns
    -------
    gasParticles : GasParticles
      The Particles object containing the snapshot data. See Particles class for
      more details. 

    Examples
    --------
    ```
    import utils_data as dt
    file = '/test/snapshot.hd5f'
    gasParticles = dt.LoadSnapshot(file)
    pos = gasParticles.pos
    mass = gasParticles.mass
    density = gasParticles.density
    v = gasParticles.v
    ```
    """

    f = h5py.File(file, "r")
    coordinates = f["PartType0"]["Coordinates"][:]  # type:ignore
    masses = f["PartType0"]["Masses"][:]            # type:ignore
    density = f["PartType0"]["Density"][:]          # type:ignore
    velocities = f["PartType0"]["Velocities"][:]    # type:ignore
    f.close()

    gasParticles = GasParticles(
        coordinates, masses, density, velocities, Lbox=Lbox
    )

    if remove_bulk_velocity is True:
        gasParticles.remove_bulk_velocity()
    if shift_to_origin is True:
        gasParticles.shift_to_origin()

    return gasParticles


# -----------------------GAS PARTICLES CLASS---------------------
class GasParticles:

    def __init__(self, pos, mass, density, velocity, Lbox) -> None:
        self.pos      = pos       # coordinate data of shape [Nparticles, 3]
        self.mass     = mass      # mass data of shape [Nparticles]
        self.density  = density   # density data of shape [Nparticles]
        self.velocity = velocity  # velocity data of shape [Nparticles, 3]
        self.Lbox     = Lbox      # box size is given in creation

        # Aliases
        self.r = self.h()         # radii is smoothing length with rate = 1.0
        self.v = self.velocity


    def __len__(self) -> int:
        return len(self.pos)


    def __getitem__(self, index):
        return GasParticles(
            self.pos[index],
            self.mass[index],
            self.density[index],
            self.v[index],
            self.Lbox,
        )


    def get_data(self) -> np.ndarray:
        """ All data in a single array of shape [Nparticles, 8]. If you need complex
         manipulation of the data, drop the object and use the numpy array instead. """
        return np.concatenate((self.pos, self.mass, self.density, self.v), axis=1)


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


    def density_velocity_vector(self):
        """ Return a vector of shape [Nparticles, 4] containing 
        [vx*rho, vy*rho, vz*rho, rho]. Both ANN and voxelize use this vector.
        Interpolation operation assign this vector on a regular grid and then """

        vec = np.stack(                            
            (                                      
                self.v[:, 0] * self.density,
                self.v[:, 1] * self.density,
                self.v[:, 2] * self.density,
                self.density,
            ),
            axis=1,
        )    
        return vec      


    def voxelize_padding_length(self, Nsize, smoothing_rate=1.0):
        """ For voxelize, pad up to the length that particles exceed the box 
        on each side. And to avoid dealing with rectangles we take the maximum 
        of padding on six sides and pad all sides uniformly.
        
        Note that the effect of 'particles outside of the box sticking in' is
        also counted, because any particles outside will be enclosed by padded
        box under this condition."""

        Lcell = self.Lbox / Nsize
        h = self.h(smoothing_rate=smoothing_rate)
        
        # Calculate the length that particles exceed the box on each side.
        # The calculation is vectorized.

        upper_pad = np.max(self.pos + h[..., None] - self.Lbox)
        lower_pad = np.max(h[..., None] - self.pos)
        Lpad = np.max((upper_pad, lower_pad)) / 2

        # Because Voxelize assumes periodic boundary condition, the padding can be
        # only half of the maximum padding

        if Lpad < 0:
            Lpad = 0  # keep the box size larger than specified

        __Lbox__ = self.Lbox + 2 * Lpad
        __Nsize__ = Nsize + 2 * int(Lpad / Lcell)
        return __Lbox__, __Nsize__


    def ann_interp_to_field(
        self, 
        Nsize, 
        eps        = 0.0,
        treetype   = 'kd',
        searchtype = 'standard'
    ):
        """ Interpolate velocity using ANN. This method is wrapped around 
        `ann_interpolate()` and should be the one called directly when using
        ANN interpolation.

        ANN does not need padding. Particles outside of box (if any) are counted
        directly and it does not enforce any boundary condition, unlike voxelize.
        """
        Lcell = self.Lbox / Nsize

        vec_grid = ann_interpolate(    # now run ANN
            data_pos   = self.pos, 
            query_pos  = make_grid_coords(Lbox=self.Lbox, Nsize=Nsize),
            f          = self.density_velocity_vector(),
            Nsize      = Nsize,
            eps        = eps,
            treetype   = 'kd',
            searchtype = 'standard',
        )

        v_grid = vec_grid[..., :3] / vec_grid[..., 3, None]     # divide by mass
        m_grid = vec_grid[..., 3] * Lcell**3                    # mass

        boxField = BoxField(v_grid, m_grid, Lcell=Lcell) # create a BoxField object to store the interpolated field

        return boxField


    def voxelize_interp_to_field(
            self, 
            Nsize, 
            smoothing_rate = 1.0,
            padding        = True,
            edge_removal   = False,
            nan_removal    = True):
        """Interpolate velocity using Voxelize by v = (m*v)_i/m_i.

        Issue
        -----
        - voxelize could cause the edge of a particle/cloud to fall off.
        Velocity here won't have this issue when we need only the divided
        value where the momentum and mass fall-off could cancel out.
        """
        Lcell = self.Lbox / Nsize

        if padding is True:
            __Lbox__, __Nsize__ = self.voxelize_padding_length(
                                        Nsize, 
                                        smoothing_rate
                                    )
        else:
            __Lbox__  = self.Lbox
            __Nsize__ = Nsize
        print("Padded Lbox: ", __Lbox__, "Nsize: ", __Nsize__)

        # Pulls data
        rhov_vec = self.density_velocity_vector()

        if edge_removal:
            rhov_vec = np.stack((rhov_vec, np.ones(len(self))), axis=1)

        vec_grid = Voxelize.__call__(
            self   = Voxelize,
            box_L  = __Lbox__,   # type: ignore
            coords = self.pos,
            radii  = self.h(smoothing_rate),
            field  = rhov_vec,
            box    = __Nsize__,
        )                                                   # Run Voxelize

        if edge_removal:
            vec_grid[vec_grid[..., 4] < edge_removal] = 0           # Remove cells not completely covered by particles

        v_grid = vec_grid[..., :3] / vec_grid[..., 3, None]     # divide by mass
        m_grid = vec_grid[..., 3] * Lcell**3                    # mass

        # Warning: numpy selection could haul a lot of time
        if nan_removal:
            v_grid[np.isnan(v_grid)] = 0                            # remove NaNs
            m_grid[np.isnan(m_grid)] = 0                            # remove NaNs

        if padding is True:
            v_grid = v_grid[:Nsize, :Nsize, :Nsize, :]
            m_grid = m_grid[:Nsize, :Nsize, :Nsize]

        # Create Field object
        boxField = BoxField(v_grid, m_grid, Lcell)

        return boxField
    

    #### UPDATE
    def interp_to_brick(
        self, run_output_dir, nbrick, Nbrick, eps, smoothing_rate=1.0
    ):
        """
    Interpolate velocity using Voxelize by v = (m*v)_i/m_i to `nbrick`^3 brick
    (`BrickField`) of size `Nbrick`^3 each and save them to a subfolder of
    output_dir.

    Parameters
    ----------
    output_dir : str
      Path to the output folder.
    nbrick : int
      Number of brick in each direction.
    Nbrick : int
      Size of each brick in each direction.
    smoothing_rate : float
      `Smoothing length h = particle radius * smoothing_rate`. This parameter
      only affects the padding length and have no effect on the interpolation
      which is based on nearest neighbor.

    Returns
    -------
    brickInventory : BrickInventory
      A BrickInventory object containing the information of the brick.
    
    Notes
    -----
    The output folder will be named as `[run_output_dir]Ng[Nbrick*nbrick]Nb[Nbrick]` 
    where Ng is the number of total grid points combined, Nb is the size of each 
    brick.
    """

        # initialize a Brick object
        Lbrick = self.Lbox / nbrick
        brickInventory = BrickInventory(run_output_dir, nbrick, Nbrick, Lbrick)

        brickInventory.run_output_dir = run_output_dir  # update data_folder attribute

        pos = self.pos
        h = self.h(smoothing_rate=smoothing_rate)
        for r in range(nbrick):                                # 0, 1, 2, ..., Nbrick-1
            for s in range(nbrick):                            # Nbrick * Lbox_blk = Lbox
                for t in range(nbrick):

                    selection = (                               # Potential problem: particles on the corner just outside the box with
                        (pos[:, 0] + h >= r * Lbrick)           # no part in the brick might be included.
                        & (pos[:, 0] - h < (r + 1) * Lbrick)
                        & (pos[:, 1] + h >= s * Lbrick)
                        & (pos[:, 1] - h < (s + 1) * Lbrick)
                        & (pos[:, 2] + h >= t * Lbrick)
                        & (pos[:, 2] - h < (t + 1) * Lbrick)
                    )

                    brickParticles = self[selection]
                    brickParticles.Lbox = Lbrick
                    brickParticles.pos[:, 0] -= r * Lbrick      # Shift origin to the lower left corner of the brick (not including margin)
                    brickParticles.pos[:, 1] -= s * Lbrick
                    brickParticles.pos[:, 2] -= t * Lbrick

                    subField = brickParticles.interp_to_field(
                        Nsize=Nbrick, eps=eps, auto_padding=True
                    )                                           # return a padded and trimmed field.

                    brickField = brickInventory._BrickField(    # Create a BrickField object and save for later use
                        v=subField.get_v(),
                        mass=subField.mass,
                        Lbrick=Lbrick,
                        Nbrick=Nbrick,
                        r=r,
                        s=s,
                        t=t,
                    )
                    brickField.save_field(
                        run_output_dir
                    )                                           # save to run_output_dir/brick_field_posXXX.pkl

        return brickInventory


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
    

    def specific_kinetic_energy(self) -> float:
        """ Compute the specific kinetic energy of the particles by total
        kinetic energy divided by total mass."""
        return self.total_kinetic_energy() / self.total_mass()
# -------------------END GAS PARTICLES CLASS---------------------



# -------------------------BOX FIELD CLASS-----------------------
class BoxField:

    def __init__(self, v, mass, Lcell) -> None:
        
        # Essential information for power spectrum computation
        self.Lcell = Lcell                     # cell length in length unit

        # Main data
        self.vx = v[..., 0]                         # stores a velocity field
        self.vy = v[..., 1]
        self.vz = v[..., 2]
        self.mass = mass                            # and a mass (density) field

        # Essential information for power spectrum computation
        self.Nsize = len(self.mass)
        self.Lbox = self.Nsize * self.Lcell


    def __getitem__(self, index):
        return BoxField(self.get_v()[index], self.mass[index], self.Lcell)
    

    def __array__(self):
        return self.get_data()


    def __array_wrap__(self, arr):         # data array like ouput of get_data(), [vx, vy, vz, m]
        return BoxField(arr[..., :3], arr[..., :3], self.Lcell)


    def get_v(self) -> np.ndarray:                # use a method instead of attribute to avoid unnecessary memory usage
        v = np.stack((self.vx, self.vy, self.vz), axis=3)
        return v


    def get_density(self) -> np.ndarray:          
        return self.mass / self.Lcell ** 3        # use a method instead of attribute to avoid unnecessary memory usage


    def get_data(self) -> np.ndarray:
        """ All data in a single array of shape [Nsize, Nsize, Nsize, 8]. If you need complex
         manipulation of the data, drop the object and use the numpy array instead. """
        return np.stack((self.vx, self.vy, self.vz, self.mass), axis=3)


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
            return FoldedBox(phi, m, beta, self.Lbox / m, self.Nsize // m)
        else:
            raise Exception("""Unsupported physical quantity name.""")


    def trim(self, Nmargin, Nbrick) -> None:
        n1 = Nmargin
        n2 = Nmargin + Nbrick
        self.vx = self.vx[n1:n2, n1:n2, n1:n2]
        self.vy = self.vy[n1:n2, n1:n2, n1:n2]
        self.vz = self.vz[n1:n2, n1:n2, n1:n2]
        self.mass = self.mass[n1:n2, n1:n2, n1:n2]
        # update params
        self.Nsize = self.Nsize - 2 * Nmargin
        self.Lbox = self.Lbox * Nbrick / (Nbrick + 2 * Nmargin)


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
        E = self.total_kinetic_energy() / self.total_mass()
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


    def peek(self):
        """A quick peek at the density and velocity slices."""
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
        slice = self.Nsize // 2
        self.plot_density_slice(slice, axis=2, ax=ax[0])
        self.plot_velocity_slice(0, slice, axis=2, ax=ax[1])
        plt.show()


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
# -------------------END BOX FIELD CLASS-----------------------





# -----------------------FOLDED BOX CLASS----------------------
class FoldedBox:

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
        """Save the `FoldedBox` object under `run_output_dir` using pickle."""
        filename = os.path.join(
            run_output_dir, "folded_field_b{}{}{}.pkl".format(*self.beta)
        )  # e.g. 'folded_field_b000.pkl'
        with open(filename, "wb") as file:
            pickle.dump(self, file)


    @staticmethod
    def load(run_output_dir, beta):
        """Load the saved `FoldedBox` object from `run_output_dir` using pickle."""
        filename = os.path.join(
            run_output_dir, "folded_field_b{}{}{}.pkl".format(*beta)
        )
        with open(filename, "rb") as file:
            return pickle.load(file)
# -------------------END FOLDED BOX CLASS---------------------






# ---------------------BRICK INVENTORY CLASS-------------------
class BrickInventory:

    def __init__(self, data_folder, nbrick, Nbrick, Lbrick) -> None:
        # data folder
        self.run_output_dir = data_folder
        # brick params
        self.nbrick = nbrick
        self.Nbrick = Nbrick
        self.Lbrick = Lbrick
        self.totalNsize = Nbrick * nbrick
        self.totalLbox = Lbrick * nbrick


    class _BrickField(BoxField):

        def __init__(self, v, mass, Lbrick, Nbrick, r, s, t) -> None:
            super().__init__(v, mass, Lcell=Lbrick/Nbrick)
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
                return FoldedBox(phi, m, beta, self.Lbox / m, self.Nsize // m)
            else:
                raise Exception("""Unsupported physical quantity name.""")

        def save_field(self, run_output_dir) -> None:
            loc = (self.r, self.s, self.t)
            vvvm = np.stack((self.vx, self.vy, self.vz, self.mass), axis=3)
            filename = os.path.join(
                run_output_dir, "brick_field_loc{}{}{}.npy".format(*loc)
            )
            np.save(filename, vvvm)


    def __getitem__(self, loc) -> _BrickField:
        r, s, t = loc
        filename = os.path.join(
            self.run_output_dir, "brick_field_loc{}{}{}.npy".format(*loc)
        )
        vvvm = np.load(filename)
        v = vvvm[:, :, :, 0:3]
        mass = vvvm[:, :, :, 3]

        field = self._BrickField(
            v=v, mass=mass, Lbrick=self.Lbrick, Nbrick=self.Nbrick, r=r, s=s, t=t
        )
        return field


    def fold(self, m, beta, quantity="velocity", Nresult=None) -> FoldedBox:
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
        foldedField = FoldedBox(
            f=f, m=m, beta=beta, Lbox=self.totalLbox / m, Nsize=self.totalNsize // m
        )
        for r in range(self.nbrick):
            for s in range(self.nbrick):
                for t in range(self.nbrick):
                    brick_field = self[r, s, t]
                    if n > 1:
                        brick_field.down_sample(
                            n=n
                        )  # brick_field have size Nbrick//n now

                    if m >= self.nbrick:  # fold stitch
                        newfoldedField = brick_field.fold(
                            m=m // self.nbrick,
                            beta=beta,
                            totalNsize=self.totalNsize // n,
                            quantity=quantity,
                        )
                        # m//nbrick is the folding factor of each brick
                        foldedField.f = (
                            foldedField.f + newfoldedField.f
                        )  # add to folded

                    elif m < self.nbrick:  # stitch fold
                        u = self.nbrick // m  # Each field is composed of u^3 files
                        phase = _get_phase(
                            beta=beta,
                            totalNsize=self.totalNsize // n,
                            Nphase=brick_field.Nsize,
                            x0=r * brick_field.Nsize,
                            y0=s * brick_field.Nsize,
                            z0=t * brick_field.Nsize,
                        )
                        f = _apply_phase(f=brick_field.get_v(), phase=phase)
                        foldedField.f[
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
        foldedField.f /= m ** 1.5

        return foldedField


    def save(self) -> None:
        """Save the `BrickInventory` object under `run_output_dir` using pickle."""
        filename = os.path.join(self.run_output_dir, "brick_decomp.pkl")
        with open(filename, "wb") as file:
            pickle.dump(self, file)


    @staticmethod
    def load(run_output_dir):
        """Load the saved `BrickInventory` object from `run_output_dir` using pickle."""
        filename = os.path.join(run_output_dir, "brick_decomp.pkl")
        with open(filename, "rb") as file:
            return pickle.load(file)
# -----------------END BRICK INVENTORY CLASS------------------







def _vec_to_vm_grid(vec_grid, Lcell):
    """ This function restore mass and velocity from vec_grid. Used internally in
  interp_field. This method is deprecated. 
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
    query_pos,
    f,
    Nsize,
    eps,
    treetype   = 'kd',
    searchtype = 'standard',
):
    results = pyann.nn2(
        data       = np.matrix(data_pos),
        query      = np.matrix(query_pos),
        k          = 1,
        eps        = eps,
        treetype   = treetype,
        searchtype = searchtype,
    )
    index = np.array(results.nn_idx) # pyann returns a 2d numpy matrix
    index = np.squeeze(index)        # change to 1d array
    index -= 1                       # pyann index starts from 1

    # deposit data to grid according to pyann found index
    if np.ndim(f) == 1:         # scalar field
        data_grid = f[index]    # correlate data to query positions
        data_grid = np.reshape(data_grid, (Nsize, Nsize, Nsize))
    elif np.ndim(f) == 2:       # vector field
        data_grid = f[index, :] # correlate data to query positions
        data_grid = np.reshape(data_grid, (Nsize, Nsize, Nsize, f.shape[1]))
    else:
        raise Exception("Unsupported data shape.")

    return data_grid


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
  
  save is of shape (query points number*k, 3). The three columns are No. x 
  nearest neighbor, index of the neighbor in data points, and distance.
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


def fold_particles(pos, m):
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
        phi[:, :, :, 0] *= phase # [:,None] can be used to add a new axis
        phi[:, :, :, 1] *= phase # and also :, :, :, can be shortened to ...
        phi[:, :, :, 2] *= phase

    return phi

def _get_phase(beta, totalNsize, Nphase, x0, y0, z0) -> np.ndarray:
    Nbrick = Nphase
    x = np.arange(x0, x0 + Nbrick)
    y = np.arange(y0, y0 + Nbrick)
    z = np.arange(z0, z0 + Nbrick)
    xxx, yyy, zzz = np.meshgrid(x, y, z, indexing="ij")
    phase = np.exp(
        -1j * (2 * np.pi / totalNsize) * (beta[0] * xxx + beta[1] * yyy + beta[2] * zzz)
    )
    return phase

# @njit
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

# @njit
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


def check_conservation(gasParticles, boxField) -> tuple:
    """Check mass, momentum, energy conservation before and after interpolation.
  """

    mass_0 = gasParticles.total_mass()                  # mass
    mass_interpolated = boxField.total_mass()
    mass_conservation = mass_interpolated / mass_0
    print("Total mass of particles: {:.3e}".format(mass_0))
    print("Total mass after interpolation: {:.3e}".format(mass_interpolated))
    print("Total mass restored by {:.3%}".format(mass_conservation)
    ) 
    print("\n")
    momentum_0 = gasParticles.total_momentum()          # momentum
    momentum_interpolated = boxField.total_momentum()
    momentum_conservation = momentum_interpolated / momentum_0
    print("Total momentum of particles:", momentum_0)   # momentum is a vector 
    print("Total momentum after interpolation:", momentum_interpolated)
    print("Total momentum restored by ({:.3%}, {:.3%}, {:.3%})".format(
        momentum_conservation[0],       # relative difference of each component
        momentum_conservation[1],
        momentum_conservation[2]
        )
    )
    print("\n")
    energy_0 = gasParticles.total_kinetic_energy()      # kinetic energy 
    energy_interpolated = boxField.total_kinetic_energy()
    energy_conservation = energy_interpolated / energy_0
    print("Total kinetic energy of particles: {:.3e}".format(energy_0))
    print("Total kinetic energy after interpolation: {:.3e}".format(
        energy_interpolated
        )
    )
    print("Total kinetic energy restored by {:.3%}".format(
        energy_conservation
        )
    )
    print("\n")
    specific_energy_0 = gasParticles.specific_kinetic_energy()      # specific energy
    specific_energy_interpolated = boxField.specific_kinetic_energy()
    specific_energy_conservation = specific_energy_interpolated / specific_energy_0
    print("Specific kinetic energy of particles: {:.3e}".format(specific_energy_0))
    print("Specific kinetic energy after interpolation: {:.3e}".format(
        specific_energy_interpolated
        )
    )
    print("Specific kinetic energy restored by {:.3%}".format(
        specific_energy_conservation
        )
    )

    return mass_conservation, momentum_conservation, energy_conservation, specific_energy_conservation







# --------------------------VISUALIZATION--------------------------
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
    # update someday to add flow arrow
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



def _vector_power(fx, fy, fz, Lbox, Nsize):
    """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  `np.sum(Pk*(2*np.pi/Lbox)**3)` and `0.5*np.mean(vx**2+vy**2+vz**2)` are equal
  """
    # Fourier transform
    a = (Lbox / (2 * np.pi)) ** 1.5 / Nsize ** 3
    fkx = pyfftw.interfaces.numpy_fft.fftn(fx, threads=1) * a
    fky = pyfftw.interfaces.numpy_fft.fftn(fy, threads=1) * a
    fkz = pyfftw.interfaces.numpy_fft.fftn(fz, threads=1) * a
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