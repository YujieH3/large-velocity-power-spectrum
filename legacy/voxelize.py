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
    BlocksDecomposition,
)

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
# Pylance report an error but this works.
