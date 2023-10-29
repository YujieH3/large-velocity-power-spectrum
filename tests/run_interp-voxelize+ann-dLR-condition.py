from vpower.interp import SimulationParticles
import vpower.voxelize
import numpy as np

SNAPSHOT = '/appalachia/d5/DISK/from_pleiades/snapshots/gmcs0_wind0_gmc9/snapshot_550.hdf5'
Nsize = 512
radii_threshold = 1.                             # np.min(allPar.h())
# 1 for all voxelize, 0 for no voxelize

inputParticles = SimulationParticles.load_snapshot(SNAPSHOT, remove_bulk_velocity=False)

denseParticles = inputParticles[inputParticles.h() <= radii_threshold]

# VOXELIZE Interpolation for dense particles
voxelizeField = denseParticles.voxelize_interp_to_field(Nsize=Nsize)

# ANN Interpolation for all input particles
annField = inputParticles.ann_interp_to_field(Nsize=Nsize, overwrite=True)

from vpower.interp import read_ann_to_grid
def read_ann_distance_to_grid(Nsize, file):
    ann_save = np.loadtxt(file, delimiter='\t')
    distance = np.array(ann_save[:, 2], dtype=np.float32)
    distance = distance.reshape(Nsize, Nsize, Nsize)
    return distance

d = read_ann_distance_to_grid(Nsize, 'ann_output.save')
R = read_ann_to_grid(inputParticles.h(), Nsize, 'ann_output.save')

condition = (d >= (1/2*annField.Lcell - R))[..., None] | np.isnan(voxelizeField.get_data())
array = np.where(condition, annField, voxelizeField)

from vpower.interp import SimulationField3D
outputField = SimulationField3D(array[..., :3], array[..., 3], annField.Lbox, annField.Nsize)

# No need for the plot. Just check conservation.
from vpower.interp import check_conservation
check_conservation(inputParticles, outputField)