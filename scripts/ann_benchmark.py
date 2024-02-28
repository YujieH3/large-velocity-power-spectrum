"""
This script benchmark the old exact NN algorithm as a reference for the new ones.
We use the real data such that the results would not be biased for different data
used.
"""

import h5py
import numpy as np
import pyann
import datetime
import time
# ------------------------------------CONFIG------------------------------------
Nsize = 1000
# -------------------------------------MAIN-------------------------------------
SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
f = h5py.File(SNAPSHOT, 'r')
coords = f["PartType0/Coordinates"][:] # of shape (N, 3)
f.close()

coords[:, 0] -= np.min(coords[:, 0])
coords[:, 1] -= np.min(coords[:, 1])
coords[:, 2] -= np.min(coords[:, 2])

# Create a meshgrid of query positions
x_space = np.linspace(0, 1, Nsize)
query_pos = np.meshgrid(x_space, x_space, x_space, indexing='ij')
query_pos = np.array(query_pos).reshape(3, Nsize**3).T

print(f"[{datetime.datetime.now()}] Starting pyann")
print(f"Data: {np.shape(coords)}")
print(f"Query: {np.shape(query_pos)}")
print(f"Resolution: {Nsize}")
t0 = time.time()

results = pyann.nn2(
    data       = np.matrix(coords),
    query      = np.matrix(query_pos),
    k          = 1,
    eps        = 0.0,
    treetype   = 'kd',
    searchtype = 'standard',
)

t1 = time.time()
print(f"[{datetime.datetime.now()}] Finished pyann")
dt = t1 - t0
print(f"Time taken: {dt/3600:.2f} hours ({dt:.2f} seconds)")

