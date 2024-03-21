import datetime
import time
import h5py
import numpy as np
from annoy import AnnoyIndex

# ------------------------------------CONFIG------------------------------------
Nsize = 250
# -------------------------------------MAIN-------------------------------------

SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
f = h5py.File(SNAPSHOT, 'r')
coords = f["PartType0/Coordinates"][:] # of shape (N, 3)
f.close()

# Shift to origin. We can update to allow any origin and any shape in the future,
# should the need ever arise.
coords[:, 0] -= np.min(coords[:, 0])
coords[:, 1] -= np.min(coords[:, 1])
coords[:, 2] -= np.min(coords[:, 2])

# Build index
print(f'[{datetime.datetime.now()}] Building index: {coords.shape}')
t = AnnoyIndex(3, 'euclidean')  # Length of item vector that will be indexed
for i in range(coords.shape[0]):
    t.add_item(i, coords[i])
t.build(1) # 10 trees
print(f'[{datetime.datetime.now()}] Index built')

# u.load('test.ann') # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

# Create a meshgrid of query positions
x_space = np.linspace(0, 1, Nsize)
query_pos = np.meshgrid(x_space, x_space, x_space, indexing='ij')
query_pos = np.array(query_pos).reshape(3, Nsize**3).T

# Query
print(f"[{datetime.datetime.now()}] Starting query")
print(f"Data: {np.shape(coords)}")
print(f"Query: {np.shape(query_pos)}")
print(f"Resolution: {Nsize}")
t0 = time.time()

for query in query_pos:
    t.get_nns_by_vector(query, n=1, search_k=-1, include_distances=False)

t1 = time.time()
print(f"[{datetime.datetime.now()}] Finished pyann")
dt = t1 - t0
print(f"Query time taken: {dt/3600:.2f} hours ({dt:.2f} seconds)")


