# conda activate anns
import datetime
import h5py
import numpy as np
from annoy import AnnoyIndex

SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
f = h5py.File(SNAPSHOT, 'r')
coords = f["PartType0/Coordinates"][:] # of shape (N, 3)
data = f["PartType0/Velocities"][:]
f.close()

# Shift to origin. We can update to allow any origin and any shape in the future,
# should the need ever arise.
coords[:, 0] -= np.min(coords[:, 0])
coords[:, 1] -= np.min(coords[:, 1])
coords[:, 2] -= np.min(coords[:, 2])

print(f'[{datetime.datetime.now()}] Building index: {coords.shape}')
f = 3
t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
for i in range(coords.shape[0]):
    t.add_item(i, coords[i])

t.build(1) # 10 trees
print(f'[{datetime.datetime.now()}] Index built')


# u.load('test.ann') # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

result = t.get_nns_by_vector(np.array([0,0,0]), n=1, search_k=-1, include_distances=False)
print(f'[{datetime.datetime.now()}] Query result: {result[0]}')
