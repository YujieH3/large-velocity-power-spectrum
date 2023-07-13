import numba
from numba import jit

import numpy as np
import utils_data as dt

# This script does not use voxelize.

def PowerSpec3D(quantity, Lbox, Nsize, mode='single', value_type='velocity', sampling='adaptive',
  kmin=None, kmax=None, kspacing=None, Np=0, correct=False, logging=True):
  """
  Calculate a power spectrum from a given 3D velocity field.

  Besides k and Pk, the output Pvk will also contain the information
  of sum of Power and number of points sampled in the bin.
  """
  # Calculate FFT and power
  if value_type == 'velocity':
    vx, vy, vz = quantity
    Pk = VectorPower(vx, vy, vz, Lbox, Nsize)
    if logging is True:
      print('Energy per unit mass before FFT: {:.2e}'.format(0.5*np.mean(vx**2+vy**2+vz**2)))
  elif value_type == 'density':
    Pk = ScalarPower(quantity, Lbox, Nsize)
    if logging is True:
      print('Conserved quantity before FFT: {:.2e}'.format(np.mean(quantity**2)))
  else:
    raise Exception("Unrecognized type (avaliable: velocity or density).")
  if logging is True:
    print('Conserved quantity after FFT: {:.2e}'.format(np.sum(Pk)*(2*np.pi/Lbox)**3))

  # Sampling k in concentric spheres
  Pk_pair = PkPair(Pk, Lbox, Nsize)
  Pkk = PkSample(Pk_pair=Pk_pair, Lbox=Lbox, Nsize=Nsize, kmin=kmin, 
                 kmax=kmax, kspacing=kspacing, sampling=sampling)
  if logging is True:
    int = Int(Pkk[:,0], Pkk[:,1], spacing='fixed', type='spherical')
    print('Conserved quantity after sampling: {:.2e}'.format(int))

  # Correct shot-noise power if required
  if correct is True:
    if Np == 0:
      raise Exception("Argument missing: number of particles (Np).")
    else:
      shotnoise = Lbox**3 / Np
      Pkk[:,1] -= shotnoise
      Pkk[np.where(Pkk[:,1] < 0),1] = 0 # set P(k)=0 if P(k)<0 after -P_shot
      Pkk[:,2] = Pkk[:,1] * Pkk[:,3]
  
  # Use energy spectral density of dimension ~v^2/k
  Pkk[:,1] *= 4*np.pi*Pkk[:,0]**2
  return Pkk


def VectorPower(vx, vy, vz, Lbox, Nsize):
  """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  pk.sum(Pk*(2*np.pi/Lbox)**3) and 0.5*np.mean(vx**2+vy**2+vz**2) are
  approximately equal
  """
  # Fourier transform
  a = ( Lbox/(2*np.pi) )**1.5 / Nsize**3
  vkx = np.fft.fftn(vx) * a
  vky = np.fft.fftn(vy) * a
  vkz = np.fft.fftn(vz) * a
  # Definition of velocity power spectrum
  Pk = 0.5 * (np.abs(vkx)**2 + np.abs(vky)**2 + np.abs(vkz)**2)

  return Pk


def ScalarPower(scalar, Lbox, Nsize):
  """
  Calculate FFT and power grid before sampling. This function does the main 
  math and physics in power spectrum computation.

  Default normalization is such that
  pk.sum(Pk*(2*np.pi/Lbox)**3) and np.sum(scalar**2) are 
  approximately equal
  """
  # Fourier transform
  a = ( Lbox/(2*np.pi) )**1.5 / Nsize**3
  fft_scalar = np.fft.fftn(scalar) * a
  # Energy power spectrum
  Pk = np.abs(fft_scalar)**2
  
  return Pk


def PkPair(Pk, Lbox, Nsize):
  """
  Create pairs of power and k. Sampling in concentric spherical
  shells with PkSample to get power spectrum. Called by PowerSpec3D.
  This function is independent of the physics and definition of the
  power spectrum
  """
  # Initialize k space 
  Lcell = Lbox/float(Nsize)
  kSpace = 2*np.pi*np.fft.fftfreq(Nsize, Lcell)
  # Create k and power pairs
  kx, ky, kz = np.meshgrid(kSpace, kSpace, kSpace, indexing='ij')
  k = np.sqrt(kx*kx + ky*ky + kz*kz)
  
  # Construct a (n,2) shape array
  k = np.ravel(k)
  Pk = np.ravel(Pk)
  Pk_pair = np.stack((k, Pk)) # Shape (2,n)
  Pk_pair = np.transpose(Pk_pair) # Shape (n,2)

  return Pk_pair


@jit(nopython=True)
def Sample(Pk_pair, kmin, kmax, spacing, type):
  """Calculate mean power from Pk pair with specified spacing"""
  Pvk = []
  bin_centers = np.arange(kmin, kmax + spacing, spacing) + spacing/2
  bins = len(bin_centers)
  for i in range(bins):
    bin_center = bin_centers[i]
    bin_low = bin_centers[i] - spacing/2
    bin_up = bin_centers[i] + spacing/2 # CYG
    filter = (Pk_pair[:,0] >= bin_low) & (Pk_pair[:,0] < bin_up)
    n = len(Pk_pair[filter])

    if type == 'adaptive': # Don't record if there is no sample
      if n != 0:
        Psum = np.sum(Pk_pair[filter][:,1])
        P = Psum/n # Average over sampling
        Pvk.append([bin_center, P, Psum, n]) # Save Psum and n data for update
    elif type == 'fixed': # Record even if there is no sample
      Psum = np.sum(Pk_pair[filter][:,1])
      if n != 0:
        P = Psum/n # Average over sampling
      else:
        P = 0
      Pvk.append([bin_center, P, Psum, n]) # Save Psum and n data for update

  Pvk = np.array(Pvk)
  
  return Pvk


def PkSample(Pk_pair, Lbox, Nsize, kmin=None, kmax=None, kspacing=None, 
             sampling='adaptive', Pvk=np.array([])):
  """
  Sample power spectrum from Pk_pair. If argument Pvk is given then this function
  is set to append mode. The sampled k will be combined with original ones.  The 
  Lbox and Nsize are that of the input data. This function is independent of the 
  physics and definition of the power spectrum

  default values:
  kmin = 2 pi / Lbox, kmax = 2 pi / (Lbox/Nsize), kspacing = 2 pi / Lbox
  """
  # kmin, kmax = np.min(Pk_pair[:,0]), np.max(Pk_pair[:,0])
  if kspacing is None:
    kspacing = 2*np.pi/Lbox
  if kmin is None:
    kmin = kspacing
  if kmax is None:
    Lcell = Lbox/float(Nsize)
    kmax = np.pi/Lcell # from pixel freq to Nyquist freq
  # Sample power spectrum P(k)
  Pvk_new = Sample(Pk_pair, kmin=kmin, kmax=kmax, spacing=kspacing, type=sampling)
  # Add to stored data, if any.
  if len(Pvk) == 0:
    Pvk = Pvk_new
  else:
    if len(Pvk) != len(Pvk_new):
      raise Exception('New P(k) have different k bins with stored P(k).')
    # Update Psum, n and get P = Psum/n. Leave k unchanged.
    Pvk[:,2] += Pvk_new[:,2]
    Pvk[:,3] += Pvk_new[:,3]
    select = np.where(Pvk[:,3] > 0) # Select n > 0 to avoid divided by zero
    Pvk[select,1] = Pvk[select,2] / Pvk[select,3]
  return Pvk


def Int(k,P,spacing='fixed', type='default'):
  """
  Calculate the integral of P(k)dk. For combined power spectra where the
  spacing is not constant, specify spacing='adaptive'.
  """
  a = 1
  if spacing == 'fixed':
    dk = (k[-1] - k[0]) / (len(k) - 1)
    if type == 'spherical':
      a = 4 * np.pi * k**2
    elif type != 'default':
      raise Exception("Unrecognized argument: type")
    int = np.sum(P * a * dk)
  elif spacing == 'adaptive': # dk is not constant over the spectrum
    dk =  k[1:] - k[:-1]
    if type == 'spherical':
      a = 4 * np.pi * k[1:]**2
    elif type != 'default':
      raise Exception("Unrecognized argument: type")
    int = np.sum(P[1:] * a * dk)
  else:
    raise Exception("Unrecognized argument: spacing.")
  return int


def Combine(Pvk1, Pvk2):
  """
  Combine two power spectrums, sampled according to Pvk2. Built for the case
  where Pvk2 has a higher folding factor than Pvk1. Meaning Pvk2 cover a higher k
  space and a larger k spacing.
  """
  # kspacing1 = (Pvk1[-1,0] - Pvk1[0,0])/(len(Pvk1) - 1)
  kspacing2 = (Pvk2[-1,0] - Pvk2[0,0])/(len(Pvk2) - 1)
  select = Pvk1[:,0] < Pvk2[0,0]
  Pvk = np.concatenate((Pvk1[select], Pvk2))
  for k in Pvk2[Pvk2[:,0] < Pvk1[-1,0]][:,0]:
    select = ((k - kspacing2/2) <= Pvk1[:,0]) & (Pvk1[:,0] < (k + kspacing2/2))
    addPsum = np.sum(Pvk1[select][:,2])
    addn = np.sum(Pvk1[select][:,3])
    Pvk[np.where(Pvk[:,0]==k),2] += addPsum
    Pvk[np.where(Pvk[:,0]==k),3] += addn
  select = np.where(Pvk[:,3] > 0)
  Pvk[select,1] = Pvk[select,2] / Pvk[select,3]
  
  return Pvk


def HighPassFilter2D(field, Lbox, low_k = None):
  """
  Remove the k below low_k end of the given image, supposing the low-k end is located at 
  the center of the image. Note that in practice of numpy.fft package, the low-k
  end is at the corner while high-k is in the middle. Should apply 
  np.fft.fftshift() before using this funciton.

  if low_k is not specified, we will take low_k = 2 pi / Lcell.
  """
  dk = 2*np.pi/Lbox
  Nsize = len(field)
  if low_k is None:
    Lcell = Nsize/Lbox
    low_k = 2*np.pi/Lcell
  pixel_rad = int(low_k/dk)
  grid = np.arange(0,Nsize)
  x, y = np.meshgrid(grid, grid, indexing='ij')
  x_ctr = x - Nsize//2
  y_ctr = y - Nsize//2
  filter = (x_ctr**2 + y_ctr**2 <= pixel_rad**2)
  field[filter] = 0
  return field


def DFT(f, pos, kSpace):
  """
  Calculate the direct particle-wise discrete Fourier transform given k space.
  F_k = \sum_\alpha f_\alpha \exp{-i k \cdot x_\alpha}

  In practice, this is approximated by deposit to grid and an FFT. This function
  is used for evaluation of this approximation.
  """
  N = len(kSpace)
  count = 0
  total = N**3
  F = np.zeros((N, N, N),dtype=np.complex128)
  for n in range(N):
    for l in range(N):
      for m in range(N):
        kx = complex(kSpace[n])
        ky = complex(kSpace[l])
        kz = complex(kSpace[m])
        F_k = np.sum(f * np.exp(-1j * (kx*pos[:,0] + ky*pos[:,1] + kz*pos[:,2])))
        F[n,l,m] = F_k
        count += 1
      print('{:.2%}'.format(count/total))
  return F


def DepositToGrid(f, pos, Nsize, Lbox):
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
  
  Lcell = Lbox/float(Nsize)
  index = np.array((pos // Lcell) % Nsize, dtype=int)
  index = np.transpose(index)
  np.add.at(f_grid, tuple(index), f)

  return f_grid


def FoldParticle(pos, m):
  """
  Takes in coordinate and the folding factor m and output 
  the folded coordinates.

  ## Parameters

  pos : coordinates

  f : some physical quantity (can be vector)

  m : folding factor

  ## Returns

  pos_fold : folded coordinates
  """
  # Initialize
  xmin, xmax = np.min(pos[:,0]), np.max(pos[:,0])
  ymin, ymax = np.min(pos[:,1]), np.max(pos[:,1])
  zmin, zmax = np.min(pos[:,2]), np.max(pos[:,2])
  # Get L (for each dimension, so this should work also for non-cubical box)
  Lx = xmax - xmin
  Ly = ymax - ymin
  Lz = zmax - zmin
  # Get r that starts from 0,0,0
  r_x, r_y, r_z = pos[:,0]-xmin, pos[:,1]-ymin, pos[:,2]-zmin
  # Fold the coordinates
  pos_fold = np.zeros(np.shape(pos)) # Create the zero array for folded coordinates
  pos_fold[:,0] = r_x % (Lx/m) + xmin
  pos_fold[:,1] = r_y % (Ly/m) + ymin
  pos_fold[:,2] = r_z % (Lz/m) + zmin

  return pos_fold


def ApplyPhase(f, beta, file=None, Nsize_blk=0, Nblock=0):
  if (file == None) | (Nblock == 0) | (Nsize_blk == 0):
    Nsize_all = f.shape[0]
    x = np.arange(Nsize_all)
    y = np.arange(Nsize_all)
    z = np.arange(Nsize_all)
  else:
    Nsize_all = Nblock * Nsize_blk
    r, s, t = dt.ReadrstIndex(file)
    x = np.arange(r*Nsize_blk, (r+1)*Nsize_blk)
    y = np.arange(s*Nsize_blk, (s+1)*Nsize_blk)
    z = np.arange(t*Nsize_blk, (t+1)*Nsize_blk)
  xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
  beta_x, beta_y, beta_z = beta[0], beta[1], beta[2]
  phase = np.exp(1j*(2*np.pi/Nsize_all)*(beta_x*xx + beta_y*yy + beta_z*zz))
  phi = np.array(f, dtype=np.complex128)
  phi[:,:,:,0] *= phase
  phi[:,:,:,1] *= phase
  phi[:,:,:,2] *= phase
  return phi


def FoldField(f, m):
  if m == 1: # m==1 means no folding
    return f

  nx = f.shape[0]
  ny = f.shape[1]
  nz = f.shape[2]
  nx1 = nx//m
  ny1 = ny//m
  nz1 = nz//m
  r = 0.0
  for i in range(m):
    for j in range(m):
      for k in range(m):
        r = r + f[i*nx1:(i+1)*nx1,j*ny1:(j+1)*ny1,k*nz1:(k+1)*nz1,:]

  return r


def DownSample(r, n):
  if n == 1: # n==1 means no downsampling
    return r

  d = 0.0
  for i in range(n):
    for j in range(n):
      for k in range(n):
        d = d + r[i::n,j::n,k::n,:]
  #for smoothing density field, needs to average the cell densities
  d /= n**3
  return d


def FoldBlocks(data_folder, Nsize, m, margin, Nblock, Nsize_blk):
  """
  Load by blocks, fold the field. The largest m available is Nsize_blk*Nblock/Nsize

  ## Parameters

  data_folder : str
    The output folder of the interpolation by block method, storing the interpolated 
    velocity fields.

  Nsize : int
    The grid size for power spectra computation. Make sure the total resolution
    Nsize_blk * Nblock equal to or is a multiple of m * Nsize. This gives the
    dynamical range of each folding factor m. 

  Nsize_blk : int
    The pixel size of one interpolated block.

  m : int
    The folding factor.

  ## Returns

  folded : ndarray
    Folded and down sampled field with box size Nsize.

  """
  Nsize_mgn = int(margin * Nsize_blk)
  Nsize_ext = Nsize_blk + Nsize_mgn
  n = Nsize_blk*Nblock//m//Nsize

  if n == 0:
    raise Exception('The total resolution Nsize_blk * Nblock must equal to or \
      is a multiple of m * Nsize.')

  folded = np.zeros((Nsize, Nsize, Nsize, 3))
  files = dt.LoadDataDir(data_folder)

  if m >= Nblock:
    for file in files:
      v = np.load(file)
      v = v[Nsize_mgn:Nsize_ext, Nsize_mgn:Nsize_ext, Nsize_mgn:Nsize_ext, :]
      v = DownSample(v, n) # requires n >= 1
      v = FoldField(v, m//Nblock) # requires m//Nblock >= 1
      folded += v
  elif m < Nblock:
    for file in files:
      v = np.load(file)
      v = v[Nsize_mgn:Nsize_ext, Nsize_mgn:Nsize_ext, Nsize_mgn:Nsize_ext, :]
      v = DownSample(v, n)      
      # Stitch
      r, s, t = dt.ReadrstIndex(file)
      u = Nblock//m # Each field is composed of u^3 files
      folded[(r%u)*Nsize//u:(r%u+1)*Nsize//u,
            (s%u)*Nsize//u:(s%u+1)*Nsize//u,
            (t%u)*Nsize//u:(t%u+1)*Nsize//u, :] += v

  # before folding, P(k) = (Lbox/2*pi)^3/Nsize^6 V(k)
  # after folding, P'(k) = (Lbox/m/2*pi)^3/(Nsize/m)^6 V(k)
  # P'(k) = m^3 P(k)
  # folding regions, the power spec increase by a factor of m^3
  # field needs to decrease by sqrt(m^3) to keep the same normalization
  folded /= m**1.5
  
  return folded


def CheckConservation(mass, velocities, mass_grid, velocities_grid):

  """Check mass, momentum, energy conservation before and after interpolation. """

  m = mass
  m_grid = mass_grid
  v = velocities
  vx, vy, vz = [velocities_grid[:,:,:,i] for i in range(3)]

  # Print mass before and after interpolation
  mass_0 = np.sum(m)
  mass_interpolated = np.sum(m_grid)
  print('Total mass of particles: {:.3g}'.format(mass_0))
  print('Total mass after interpolation: {:.3e}'.format(mass_interpolated))

  # Print momentum before and after interpolation
  momentum_0 = [np.sum(m*v[:,0]), np.sum(m*v[:,1]), np.sum(m*v[:,2])]
  momentum_interpolated = [np.sum(m_grid*vx), np.sum(m_grid*vy), np.sum(m_grid*vz)]
  print('Total momentum of particles:', momentum_0)
  print('Total momentum after interpolation:', momentum_interpolated)

  # Print energy before and after interpolation
  energy_0 = np.sum(0.5*m*(v[:,0]**2 + v[:,1]**2 + v[:,2]**2))
  energy_interpolated = np.sum(0.5*m_grid*(vx**2 + vy**2 + vz**2))
  print('Total kinetic energy of particles: {:.3g}'.format(energy_0))
  print('Total kinetic energy after interpolation: {:.3e}'.format(energy_interpolated))

