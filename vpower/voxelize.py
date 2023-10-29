#----------------------------------------------------------------------
#       File:			voxelize.py
#		Programmer:		Yujie He
#		Last modified:	15/10/23                # dd/mm/yy
#		Description:    Voxelize extension for interpolation methods.
#----------------------------------------------------------------------
#
#       This file monkey patches the SimulationParticles class in
#       vpower.interp to add interpolation methods using Voxelize.
#
#       Functions and classes starting with '_' are for intrinsic use 
#       only. 
#
#----------------------------------------------------------------------


import numpy as np
import time

from voxelize import Voxelize
Voxelize.__init__(self=Voxelize, use_gpu=False, network_dir=None)  # type: ignore

from vpower.interp import (
    SimulationField3D,
    SimulationParticles,
    _vec_to_vm_grid,
    BlocksDecomposition,
)


def voxelize_interp_to_field(self, Nsize, smoothing_rate=1.0, auto_padding=True,
                             edge_removal=False):
    """Interpolate velocity using Voxelize by v = (m*v)_i/m_i.

    Issue
    -----
      - voxelize could cause the edge of a particle/cloud to fall off.
      Velocity here won't have this issue when we need only the divided
      value where the momentum and mass fall-off could cancel out.
    """
    t0 = time.perf_counter()
    print("Interpolating velocity field...")

    Lcell = self.Lbox / Nsize
    h = self.h(smoothing_rate=smoothing_rate)
    rho = self.rho(smoothing_rate=smoothing_rate)

    if auto_padding is True:
        # Calculate the length that particles exceed the box on each side.
        # The calculation is vectorized.
        upper_padding = np.max(self.pos + h[..., None] - self.Lbox)
        lower_padding = np.max(h[..., None] - self.pos)

        # Because Voxelize assumes periodic boundary condition, the padding can be
        # only half of the maximum padding
        padding = np.max((upper_padding, lower_padding)) / 2

        if padding < 0:
            padding = 0  # keep the box size larger than specified

        _Lbox_ = self.Lbox + 2 * padding
        _pos_ = self.pos
        _Nsize_ = Nsize + 2 * int(padding / Lcell)
        print("Padding: ", padding, "Lbox: ", _Lbox_, "Nsize: ", _Nsize_)

        # Log
        t = time.perf_counter() - t0
        print("Auto padding done. Time elapsed: {:.2f} s".format(t))
    else:
        _Lbox_ = self.Lbox
        _pos_ = self.pos
        _Nsize_ = Nsize

    # Interpolation
    # vec = [vx*rho, vy*rho, vz*rho, rho] -> [(vx*rho)_i, (vy*rho)_i, (vz*rho)_i, rho_i]
    if edge_removal is True:
        vec = np.stack(
            (
                self.v[:, 0] * rho, 
                self.v[:, 1] * rho, 
                self.v[:, 2] * rho, 
                rho, 
                np.ones(len(self))
            ), axis=1
        )
    else:
        vec = np.stack(
            (
                self.v[:, 0] * rho, 
                self.v[:, 1] * rho, 
                self.v[:, 2] * rho, 
                rho
            ), axis=1
        )

    vec_grid = Voxelize.__call__(
        self=Voxelize,
        box_L=_Lbox_,  # type: ignore
        coords=_pos_,
        radii=h,
        field=vec,
        box=_Nsize_,
    )                                                   # Run Voxelize

    if edge_removal is True:
        vec_grid[vec_grid[..., 4] < 0.7] = 0           # Remove cells not completely covered by particles

    v_grid = vec_grid[..., :3] / vec_grid[..., 3, None]     # divide by mass
    m_grid = vec_grid[..., 3] * Lcell**3                    # mass

    # Log
    t = time.perf_counter() - t0
    print("Interpolation done. Time elapsed: {:.2f} s".format(t))

    if auto_padding is True:
        v_grid = v_grid[:Nsize, :Nsize, :Nsize, :]
        m_grid = m_grid[:Nsize, :Nsize, :Nsize]

    # Create Field object
    simField3D = SimulationField3D(v_grid, m_grid, Lbox=self.Lbox, Nsize=Nsize)

    return simField3D


def voxelize_interp_to_blocks(
    self, run_output_dir, nblocks, Nblock, smoothing_rate=1.0
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
      `Smoothing length h = particle radius * smoothing_rate`. The total mass is kept
      constant while doing so.

    Returns
    -------
    blocksDecomp : BlocksDecomposition
      A BlocksDecomposition object containing the information of the blocks.

    Notes
    -----
    The output folder will be named as `'Ng{}Nb{}'.format(Nblock*nblocks,
    Nblock)` where Ng is the number of total grid points combined, Nb
    is the size of each block.
    """

    # initialize a Blocks object
    Lblock = self.Lbox / nblocks
    blocksDecomp = BlocksDecomposition(run_output_dir, nblocks, Nblock, Lblock)

    blocksDecomp.run_output_dir = run_output_dir  # update data_folder attribute

    pos = self.pos
    h = self.h(smoothing_rate=smoothing_rate)
    for r in range(nblocks):  # 0, 1, 2, ..., Nblock-1
        for s in range(nblocks):  # Nblock*Lbox_blk=Lbox
            for t in range(nblocks):
                # Potential problem: particles on the corner just outside the box with
                # no part in the block might be included.
                selection = (
                    (pos[:, 0] + h >= r * Lblock)
                    & (pos[:, 0] - h < (r + 1) * Lblock)
                    & (pos[:, 1] + h >= s * Lblock)
                    & (pos[:, 1] - h < (s + 1) * Lblock)
                    & (pos[:, 2] + h >= t * Lblock)
                    & (pos[:, 2] - h < (t + 1) * Lblock)
                )

                blockParticles = self[selection]
                blockParticles.Lbox = Lblock
                # Shift origin to the lower left corner of the block (not including margin)
                blockParticles.pos[:, 0] -= r * Lblock
                blockParticles.pos[:, 1] -= s * Lblock
                blockParticles.pos[:, 2] -= t * Lblock

                subField = blockParticles.voxelize_interp_to_field(
                    Nsize=Nblock, smoothing_rate=smoothing_rate, auto_padding=True
                )  # return a padded and trimmed field.

                # Create a BlockField3D object and save for later use
                blockField = blocksDecomp._BlockField3D(
                    v=subField.v(),
                    mass=subField.mass,
                    Lblock=Lblock,
                    Nblock=Nblock,
                    r=r,
                    s=s,
                    t=t,
                )
                blockField.save_field(
                    run_output_dir
                )  # save to run_output_dir/block_field_pos_XXX.pkl

    return blocksDecomp


# Monkey patch Simulation Particles
SimulationParticles.voxelize_interp_to_field = voxelize_interp_to_field  # type: ignore
SimulationParticles.voxelize_interp_to_blocks = voxelize_interp_to_blocks  # type: ignore
# Pylance report an error but this works.
