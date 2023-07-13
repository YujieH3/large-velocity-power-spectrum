import numpy as np
import h5py
import os

def PrepDataPts(pos, file='./data.pts'):
  """
  Prepare data.pts for ANN from simulation data.
  """
  np.savetxt(file, pos, delimiter='\t', fmt='%.16f')

def PrepQueryPts(pos, Nsize, file='./query.pts'):
  """
  Prepare query.pts for ANN, with resolution/Nsize specified. The coordinate data 
  is only used to specify the grid range.
  """
  # Initialize
  xmin, xmax = np.min(pos[:,0]), np.max(pos[:,0])
  ymin, ymax = np.min(pos[:,1]), np.max(pos[:,1])
  zmin, zmax = np.min(pos[:,2]), np.max(pos[:,2])
  xcell = (xmax - xmin) / Nsize
  ycell = (ymax - ymin) / Nsize
  zcell = (zmax - zmin) / Nsize
  # x,y,z ticks at the center of the pixels
  xSpace = np.linspace(xmin + xcell/2, xmax + xcell/2, Nsize)
  ySpace = np.linspace(ymin + ycell/2, ymax + ycell/2, Nsize)
  zSpace = np.linspace(zmin + zcell/2, zmax + zcell/2, Nsize)
  # Create grid from x,y,z ticks
  grid = np.meshgrid(xSpace,ySpace,zSpace,indexing='ij')
  grid = np.array(grid)
  # Mock particle coordinates
  pos = np.reshape(grid, (3, Nsize**3))
  pos = np.transpose(pos) # shape: (256**3, 3)
  # Prepare query.pts
  np.savetxt(file, pos, delimiter='\t', fmt='%.16f')

import subprocess
import time
def ANN(data_file, query_file, index_file, k, eps, maxpts, do_time=True):
  """
  Run ANN by executing command lines.
  `ann_sample [-d dim] [-max mpts] [-nn k] [-e eps] [-df data] [-qf query]`
  """
  ann_path = '/appalachia/d6/yujie/ann_1.1.2/sample'
  # Compile my_ann_sample.cpp
  ret = subprocess.run(['g++ my_ann_sample.cpp -o ann_sample -I/appalachia/d6/yujie/ann\
_1.1.2/include -L/appalachia/d6/yujie/ann_1.1.2/lib -lANN'], shell=True, cwd=ann_path)
  if ret.returncode != 0:
    raise Exception("Error occurred when compling ANN.")
  # Time
  if do_time is True:
    t0 = time.perf_counter()
  # Run ANN
  ret = subprocess.run(['time ./ann_sample -e {} -max {} -nn {} -df {} -qf {} \
> {}'.format(eps, maxpts, k, data_file, query_file, index_file)], shell=True,
  cwd=ann_path)
  if ret.returncode != 0:
    raise Exception("Error occurred when running ANN.")
  # Time
  if do_time is True:
    t1 = time.perf_counter()
    t = t1 - t0
    return t


def ToGrid(data=np.array, Nsize=int, index_file=str, do_weight=False):
  """
  Create the numpy array from saved data. From output.save to deposit some 
  physical quantity data to grid.

  Updated for new data. The new output.save is organized in the following form:
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
  # Read data from output.save
  ann_save = np.loadtxt(index_file, delimiter='\t')
  # Utilize searched index
  weight = 0
  k = int(np.max(ann_save[:,0])) + 1
  # Calculate sum or weighted sum
  searched = 0
  for i in range(k):
    no_i_nearest = np.abs(ann_save[:,0] - i) < 1e-5 # Number i nearest
    index = ann_save[no_i_nearest][:,1]
    index = np.array(index, dtype=int)
    if do_weight is False:
      searched += data[index]
    else:
      weight += 1/ann_save[no_i_nearest][:,2]**2 # inverse squared distance
      searched += data[index] * 1/ann_save[no_i_nearest][:,2]**2
  # Calculate mean or weighted mean
  if do_weight is False:
    searched /= k
  else:
    searched /= weight
  # transfer particle-wise data to a data cube
  if np.ndim(data) == 1:
    data_grid = np.reshape(searched, (Nsize, Nsize, Nsize))
  elif np.ndim(data) == 2: # Vector data
    data_grid = np.reshape(searched, (Nsize, Nsize, Nsize, 3))
    data_grid = np.transpose(data_grid, (3,0,1,2))
  else:
    raise Exception("Unsupported data shape.")
  
  return data_grid

def LoadGRF(p, grfNsize=512, fSol=0):
  """
  Load saved GRF data.

  ## Example
  ```
  import utils_data as dt
  v = dt.LoadGRF(p=-3)
  vx, vy, vz = v
  ```
  """

  if grfNsize == 512:
    file = './data/grf512_{}_{}.npy'.format(p, fSol)
  elif grfNsize == 256:
    file = './data/grf_{}_{}.npy'.format(p, fSol)
  else:
    raise Exception("Data matching requirement not found under ./data")

  grf = np.load(file)
  grf = np.squeeze(grf)
  grf = np.array(grf, dtype=np.float32)

  return grf

def LoadGRFParticles(p, grfNsize=512, fSol=0, Lbox=20.):
  """
  Load saved GRF data into mock particles with coordinates, velocities
  and volume.

  ## Example
  ```
  import utils_data as dt
  pos, mass, rho, v = dt.LoadGRFParticles(p=-3)
  vx, vy, vz = v[:,0], v[:,1], v[:,2]
  ```
  """

  grf = LoadGRF(p, grfNsize, fSol)

  # Construct some mock particles. Get their coorinates and velocity field
  xSpace = np.linspace(0, Lbox,   grfNsize)
  ySpace = np.linspace(1, Lbox+1, grfNsize)
  zSpace = np.linspace(2, Lbox+2, grfNsize)
  grid = np.meshgrid(xSpace, ySpace, zSpace, indexing='ij')
  grid = np.array(grid)
  # Mock coordinates (particle)
  pos = np.reshape(grid, (3, grfNsize**3))
  pos = np.transpose(pos) # shape: (grfNsize**3, 3)
  # Mock velocity (particle)
  v = np.reshape(grf, (3, grfNsize**3))
  v = np.transpose(v) # shape: (grfNsize**3, 3)
  # Volume = Lcell^3
  Lcell = Lbox/grfNsize
  V = Lcell ** 3
  # Mock mass
  mass = np.ones(grfNsize**3)
  # Mock density
  rho = mass / V

  return pos, mass, rho, v 

def LoadSnapshot(file):
  """
  Load coordinates, density, masses and velocities from a snapshot.

  ## Parameters

  file: The HD5F snapshot file name

  ## Returns

  Coordinates, density, masses and velocities (in order)
    In whatever standard units used by the snapshot. In our usual application,
  coordinates: kpc
  masses: 10^10 Msun
  density: 10^10 Msun / kpc^3
  velocities: km / s

  ## Examples
  
  ```
  import utils_data as dt
  file = '/test/snapshot.hd5f'
  pos, mass, rho, v = dt.LoadSnapshot(file)
  ```
  
  """

  f = h5py.File(file, 'r')
  coordinates = f["PartType0"]["Coordinates"][:]
  masses = f["PartType0"]["Masses"][:]
  density = f["PartType0"]["Density"][:]
  velocities = f["PartType0"]["Velocities"][:]
  f.close()
  return coordinates, masses, density, velocities

def PrepSnapshot(file, remove_bulk_velocity=True, initialize_coord=True):

  """
  Load coordinates, masses, density and velocities from a snapshot, remove bulk
  velocity to have zero total momentum and initialize the coordinates to start
  at (0,0,0).
  """

  pos, mass, rho, v = LoadSnapshot(file)

  if remove_bulk_velocity is True:
    # - center of mass velocity
    M = np.sum(mass)
    v[:,0] -= np.sum(mass * v[:,0]) / M
    v[:,1] -= np.sum(mass * v[:,1]) / M
    v[:,2] -= np.sum(mass * v[:,2]) / M

  if initialize_coord is True:
    # Initialize the coordinates to start at (0,0,0)
    pos[:,0] -= pos[:,0].min()
    pos[:,1] -= pos[:,1].min()
    pos[:,2] -= pos[:,2].min()

  return pos, mass, rho, v

def LoadDataDir(data_dir):
  filenames = []
  for root, dirs, files in os.walk(data_dir):
    for name in files:
      filenames.append(os.path.join(root, name))
  return filenames

  
def ReadrstIndex(filename):

  filename = os.path.basename(filename)

  index_r  = filename.index('r')
  index_s  = filename.index('s')
  index_t  = filename.index('t')
  index_pt = filename.index('.')
  
  r = int(filename[index_r+1:index_s]) # or index(r)+1:index(s)
  s = int(filename[index_s+1:index_t])
  t = int(filename[index_t+1:index_pt])
  return r, s, t


