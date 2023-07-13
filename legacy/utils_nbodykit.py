import numpy as np
import pickle

import utils_pk as pk

from nbodykit.lab import *
from nbodykit import setup_logging
setup_logging()

def MeasurePowerSpectra(f_:np.ndarray, pos_:np.ndarray, Nsize:int, Lbox_:float, 
  mSpace:list, output_file=None, interp='TSC', correctShotnoise=True):
  """
  Measure the power spectra of some physical quantity f at coordinates pos
  distributed in a cube. The measurement is done by depositing f in grids of
  size Nsize^3 and a FFT. By folding multiple times the full spectra is restored.
  k ranges from the largest scale at the lowest folding to the Nyquist frequency 
  of the highest folding, from 2\pi/(Lbox/m_min) to \pi Nsize/(Lbox/m_max).

  Keyword Arguments:
  pos: coordinates
  f: some physical quantity (can be vector)
  ms: folding factors, a list of integers, e.g. [0,4,8] or [16, 16**2]
  interp_scheme: interpolation method, can be 'NGP' or 'TSC'

  Output:
  pvks: a dictionary of the power spectrum of different folding and combined.
  """

  pvks = {}
  for m in mSpace:
    # Reset
    f = f_.copy()
    pos = pos_.copy()
    Lbox = Lbox_
    Np = len(f)

    # Set coordinates to begin at (0,0,0)
    xmin = np.min(pos[:,0])
    ymin = np.min(pos[:,1])
    zmin = np.min(pos[:,2])

    pos[:,0] -= xmin
    pos[:,1] -= ymin
    pos[:,2] -= zmin

    # Fold
    if m != 0:
      pos = pk.Fold(pos, m=m)
      Lbox /= m
    
    # Assign to grid
    if interp == 'NGP':
      f_grid = pk.DepositToGrid(f, pos, Nsize, Lbox)
      fx = f_grid[:,:,:,0]
      fy = f_grid[:,:,:,1]
      fz = f_grid[:,:,:,2]
      # Normalization
      norm = 1/Np
    elif interp == 'TSC':
      # initialize the catalog
      cat = ArrayCatalog({'Position':pos, 'fx':f[:,0], 'fy':f[:,1], 'fz':f[:,2]})
      cat.attrs['BoxSize'] = Lbox

      # convert catalog to a mesh with desired window and interlacing
      mesh_vx = cat.to_mesh(value='fx', Nmesh=Nsize, resampler='tsc', compensated=True, interlaced=True)
      mesh_vy = cat.to_mesh(value='fy', Nmesh=Nsize, resampler='tsc', compensated=True, interlaced=True)
      mesh_vz = cat.to_mesh(value='fz', Nmesh=Nsize, resampler='tsc', compensated=True, interlaced=True)

      # Calculate power spectrum
      fx = mesh_vx.paint()
      fy = mesh_vy.paint()
      fz = mesh_vz.paint()

      # Normalization
      norm = 1/Nsize**3
    else:
      raise Exception("Only support NGP and TSC for interp for now.")

    # Calculate power spectra over the folded field
    pvk = pk.PowerSpec3D(quantity=[fx, fy, fz], Lbox=Lbox, Nsize=Nsize, Np=Np, correct=True)

    # Add the factor for normalization *m^3
    if m != 0:
      pvk[:,2] *= m**3
      pvk[:,1] = pvk[:,2] / pvk[:,3]

    # Save result
    pvks['m={}'.format(m)] = pvk

  if output_file is not None:
    with open(output_file, 'wb') as f:
      pickle.dump(pvks, f)

  # Combine power spectrums with different folding factor m into one
  for m in mSpace:
    pvk = pvks['m={}'.format(m)]

    # From 2*fundamental wavenumber to 1/2 Nyquist frequency 
    if m != 0:
      Lbox = Lbox_ / m
    else:
      Lbox = Lbox_
    k_f = 2*np.pi/Lbox
    k_ny = np.pi/(Lbox/Nsize)
    select = (pvk[:,1] > 0) & (pvk[:,0] >= k_f*4) & (pvk[:,0] <= k_ny/2)
    pvk = pvk[select]

    if len(pvk) > 0:
      # Combine
      if m == mSpace[0]:
        pvk_combined = pvk
      else:
        pvk_combined = pk.Combine(pvk_combined, pvk)

  pvks['combined'] = pvk_combined

  if output_file is not None:
    with open(output_file, 'wb') as f:
      pickle.dump(pvks, f)

  return pvks