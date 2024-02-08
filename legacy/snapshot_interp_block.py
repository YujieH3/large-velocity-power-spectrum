import os
from utils_folding import SimulationParticles, init_dir
from configparser import ConfigParser

conf = ConfigParser()
conf.read('params_config.ini')
OUTPUT_DIR = conf.get('file_params','OUTPUT_DIR')
SNAPSHOT = conf.get('file_params','SNAPSHOT')
nblocks = conf.getint('interp_params', 'nblocks')
Nblock = conf.getint('interp_params', 'Nblock')
smoothing_rate = conf.getfloat('interp_params', 'smoothing_rate')

# Initialize output directory for this run
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'Ng{}Nb{}'.format(Nblock*nblocks, Nblock))
init_dir(RUN_OUTPUT_DIR, auto_overwrite=True)
conf.set('file_params', 'RUN_OUTPUT_DIR', RUN_OUTPUT_DIR)
with open('params_config.ini', 'w') as f:
	conf.write(f)

# Interpolation
simParticles = SimulationParticles.load_snapshot(SNAPSHOT, remove_bulk_velocity=True, shift_to_origin=True)
blocksDecomp = simParticles.interp_to_blocks(RUN_OUTPUT_DIR, nblocks=nblocks, Nblock=Nblock,
                                          smoothing_rate=smoothing_rate) # Folded fields are saved in the process
blocksDecomp.save() # Save the decomp object
