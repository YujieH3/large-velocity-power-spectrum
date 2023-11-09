from vpower.interp import SimulationParticles
import vpower.voxelize
import numpy as np
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o","--output", type=str)
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

if radii_threshold > 0.0:
    # VOXELIZE Interpolation for dense particles
    denseParticles = inputParticles[inputParticles.h() <= radii_threshold]
    voxelizeField = denseParticles.voxelize_interp_to_field(Nsize=Nsize)

    # ANN Interpolation for all input particles
    annField = inputParticles.ann_interp_to_field(Nsize=Nsize, overwrite=True)

    # Combination
    from vpower.interp import SimulationField3D
    mask = (voxelizeField.mass > 0)             # percentile of the non-zero values
    density_threshold = np.nanpercentile(voxelizeField.get_density()[mask], 10.0)
    print(density_threshold)

    condition = (voxelizeField.get_density() <= density_threshold)
    array = np.where(condition[..., None], annField, voxelizeField)
    outputField = SimulationField3D(array[..., :3], array[..., 3], annField.Lcell)
else:
    # ANN Interpolation for all input particles
    outputField = inputParticles.ann_interp_to_field(Nsize=Nsize, overwrite=True)
    dt = time.perf_counter() - t0

# No need for the plot. Just check conservation.
from vpower.interp import check_conservation
M_conserve, P_conserve, E_conserve, e_conserve = check_conservation(inputParticles, outputField)

print("Done in {} s\n".format(dt))

# Save the test results
to_save = {}
to_save['Resolution'] = Nsize
to_save['Radii threshold'] = radii_threshold
to_save['Density threshold percentile'] = density_threshold_percentile
to_save['Mass conservation'] = M_conserve
to_save['Momentum conservation'] = P_conserve
to_save['Energy conservation'] = E_conserve
to_save['Specific energy conservation'] = e_conserve
to_save['Computation time'] = dt

out = pd.DataFrame(to_save)
out.to_csv(args.output + ".csv", mode='a', index=False) # 'a' append to the end of file if it exists