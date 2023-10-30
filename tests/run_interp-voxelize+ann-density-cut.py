from vpower.interp import SimulationParticles
import vpower.voxelize
import numpy as np
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n","--nsize", type=int)
parser.add_argument("-r","--radii_threshold", type=float)
parser.add_argument("-d","--density_threshold_percentile",type=float)
args = parser.parse_args()

Nsize = args.nsize
radii_threshold = args.radii_threshold
density_threshold_percentile = args.density_threshold_percentile
print("Nsize: ", Nsize, "radii_threshold: ", radii_threshold, "density_threshold_percentile: ", density_threshold_percentile)

t0 = time.perf_counter()
SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
inputParticles = SimulationParticles.load_snapshot(SNAPSHOT, remove_bulk_velocity=False)

# VOXELIZE Interpolation for dense particles
denseParticles = inputParticles[inputParticles.h() <= radii_threshold]
voxelizeField = denseParticles.voxelize_interp_to_field(Nsize=Nsize)

# ANN Interpolation for all input particles
annField = inputParticles.ann_interp_to_field(Nsize=Nsize, overwrite=True)

# Combination
from vpower.interp import SimulationField3D
density_threshold = np.nanpercentile(voxelizeField.get_density(), density_threshold_percentile)

condition = np.isnan(voxelizeField.get_data()) | (voxelizeField.get_density() < density_threshold)[..., None]
array = np.where(condition, annField, voxelizeField)
outputField = SimulationField3D(array[..., :3], array[..., 3], annField.Lcell)
dt = time.perf_counter() - t0

# No need for the plot. Just check conservation.
from vpower.interp import check_conservation
check_conservation(inputParticles, outputField)

print("Done in {} s\n".format(dt))