# Import voxelize
import sys
import os
sys.path.append('/appalachia/d6/yujie/voxelize-master/voxelize')
sys.path.append('/appalachia/d6/yujie/voxelize-master')
import voxelize
from voxelize import Voxelize
Voxelize.__init__(self=Voxelize, use_gpu=False, network_dir=None)

import numpy as np
import utils_data as dt
import utils_pk as pk

def VoxelizeInterpolate(pos, mass, rho, velocities, Lbox, Nsize,
  smooth_rate=1., shift_to_origin=True):

  """
  Interpolate velocity using Voxelize by v = (m*v)_i/m_i.

  ## Parameters

  smoothing_rate : float
    Smoothing length h = particle radius * smoothing_rate. The total mass is kept
  constant while doing so.

  ## Returns

  vgrid, m_grid

  ## Examples

  ```
  v_grid, m_grid = VoxelizeInterpolate(pos, mass, rho, velocities Lbox=1., Nsize=256)
  vx = v_grid[:,:,:,0]
  vy = v_grid[:,:,:,1]
  vz = v_grid[:,:,:,2]
  ```

  """
  # Initialize
  v = velocities # Use a shorter name for simplicity
  Lcell = Lbox / Nsize

  if shift_to_origin == True:
    # Shift coordinates to begin at (0,0,0)
    xmin = np.min(pos[:,0])
    ymin = np.min(pos[:,1])
    zmin = np.min(pos[:,2])
    pos[:,0] -= xmin
    pos[:,1] -= ymin
    pos[:,2] -= zmin

  # mass fixed, reduce V, increase rho -> larger particles
  rho /= smooth_rate**3 # Change smoothing length while keeping mass constant
  V = mass / rho # volume = mass / density
  h = ((3*V)/(4*np.pi))**(1/3) # Smoothing length h = particle radius * smoothing_rate

  # Method 1, (mv)_i/(m)_i
  # vec = [vx*rho, vy*rho, vz*rho, rho]
  vec = np.stack([v[:,0]*rho, v[:,1]*rho, v[:,2]*rho, rho], axis=1)
  # [vx*rho, vy*rho, vz*rho, rho] -> [(vx*rho)_i, (vy*rho)_i, (vz*rho)_i, rho_i]
  vec_grid = Voxelize.__call__(self=Voxelize, box_L=Lbox, 
    coords=pos, radii=h, field=vec, box=Nsize)

  rho_grid = vec_grid[:,:,:,3] # vec = [vx*rho, vy*rho, vz*rho, rho]
  m_grid = rho_grid * Lcell**3 # rho_grid = \sum rho_alpha*V_i,alpha / V_i, V_i = Lcell*3
  rho_grid[np.where(rho_grid==0)] = 1 # Avoid divide by zero. 
  # Whereever rho_grid is 0, v_grid is 0. So this won't cause trouble.

  # v = (rho*v)_i/rho_i
  vec_grid[:,:,:,0] /= rho_grid
  vec_grid[:,:,:,1] /= rho_grid
  vec_grid[:,:,:,2] /= rho_grid
  # vec = [vx, vy, vz, rho]
  v_grid = vec_grid[:,:,:,0:3]
  v_grid[np.isnan(v_grid)] = 0 # Replace nan by zero, if any.

  return v_grid, m_grid

def SnapshotPowerSpec(file, Lbox, Nsize, smoothing_rate=1., norm=None, logging=True):
  """
  Calculate the power spectrum from snapshot without folding, using voxelize for
  interpolation. 

  ## Parameters

  ## Returns

  ## Examples

  ```
  # Calculate and plot the power spectrum.
  from utils_voxelize import *
  import matplotlib as plt
  pvk = SnapshotPowerSpec('/test/snapshot.hd5f', Lbox=1., Nsize=512)
  k = pvk[:,0]
  P = pvk[:,1]
  plt.plot(k, P)
  ```

  """
  pos, mass, rho, v = dt.PrepSnapshot(file)
  v_grid, m_grid = voxelize_interpolate(pos, mass, rho, v, Lbox, Nsize, smoothing_rate)
  vx, vy, vz = v_grid[:,:,:,0], v_grid[:,:,:,1], v_grid[:,:,:,2]
  if logging is True:
    pk.CheckConservation(mass, v, m_grid, v_grid)
  pvk = pk.PowerSpec3D([vx, vy, vz], Lbox, Nsize, logging=logging)
  return pvk
  

def InterpBlocksVelocity(pos, mass, rho, velocity, Lbox, nblocks, Nblock, 
  margin_ratio, smooth_rate=1., scheme='voxelize', output_folder='./output'):

  """
  Interpolate the full field by small blocks, each contains a fraction of the
  full box. The fundamental frequency would be Nsize_blk * Nblock * Nyquist
  frequency (2*np.pi/Lbox).

  ## Parameters

  margin : float
    The ratio of marginping size to the block size. It's best that 
  margin satisfy: 1. margin*Lbox/Nblock > h (for most h) 2. margin*Nsize_blk
  is an integer.

  ## Returns

  Nblock^3 files, each contain an array of Nsize_blk^3

  ## Examples
  """

  # Prepare parameters
  Lblock = Lbox / nblocks
  Lmargin = margin_ratio * Lblock       # margin box length (margin box length)
  fullLblock = Lblock + 2 * Lmargin
  Nmargin = int(margin_ratio * Nblock)
  fullNblock = Nblock + 2 * Nmargin
  print('Block length: {} Block Nsize: {}'.format(fullLblock, fullNblock))

  # Prepare output folder
  output_folder = os.path.join(output_folder, 'Ng{}Nb{}Nmargin{}'.\
    format(Nblock*nblocks, Nblock, Nmargin))
  if os.path.exists(output_folder) is False:
    os.mkdir(output_folder)

  pos += Lmargin # Shift coordinate

  for r in range(nblocks): # 0, 1, 2, ..., Nblock-1
    for s in range(nblocks):  # Nblock*Lbox_blk=Lbox
      for t in range(nblocks):
        selection = \
          (pos[:,0] >= r*Lblock) & (pos[:,0] < (r+1)*Lblock + 2*Lmargin) &\
          (pos[:,1] >= s*Lblock) & (pos[:,1] < (s+1)*Lblock + 2*Lmargin) &\
          (pos[:,2] >= t*Lblock) & (pos[:,2] < (t+1)*Lblock + 2*Lmargin)

        blk_pos =      pos[selection]
        blk_rho =      rho[selection]
        blk_m   =     mass[selection]
        blk_v   = velocity[selection]

        blk_pos[:,0] -= r*Lblock
        blk_pos[:,1] -= s*Lblock
        blk_pos[:,2] -= t*Lblock

        if scheme == 'voxelize':
          v_grid, m_grid = VoxelizeInterpolate(blk_pos, blk_m, blk_rho, blk_v, 
            Lbox=fullLblock, Nsize=fullNblock, smooth_rate=smooth_rate,
            shift_to_origin=False)
        else:
          raise Exception("Interpolation scheme unrecognized.")

        np.save(os.path.join(output_folder, 'r{}s{}t{}.npy'.format(r,s,t)), v_grid)
  return output_folder

