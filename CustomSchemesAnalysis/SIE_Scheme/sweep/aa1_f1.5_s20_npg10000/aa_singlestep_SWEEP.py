"""
Step 1: get symmetric model.
Step 2: forward depletion until t=300
Step 3: generate a bad flux shape
Step 4: forward depletion until 325
Step 5: Iterate...
"""


import CustomSchemes.data.pwr_symmetric_template as pwr
import CustomSchemes
import openmc
import openmc.deplete
import numpy as np
import copy
import logging
import pickle as pkl
import glob
import os
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
openmc.deplete.pool.USE_MULTIPROCESSING=False

from CustomSchemes.AA import Anderson
from CustomSchemes.TransportMath import run_transport, run_transport_for_chain, run_transport_standard
from CustomSchemes.DumbTransport import transport as dummy_transport
from CustomSchemes.DumbTransport import flat_transport, oscillating_transport
from CustomSchemes.NuclideVectorMath import get_nuclides_for_transport, \
    make_transport_material_library, \
    get_depletion_materials_from_results_EOS, \
    chain_from_pkl, \
    depletable_mats_from_model, \
    make_transport_material_library, \
    relax_nuclides_from_files


"""
Setup logging
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filemode='a', filename="LOGGING.LOG")
logging.getLogger().setLevel(logging.DEBUG)
logging.info(f"Now running input....")


# Input stuff
micro_xs = chain_from_pkl(file=str(Path(CustomSchemes.data.__file__).parent / "FINAL_CHAIN.pkl")) # get xs from a reference file
chain_file = str(Path(CustomSchemes.data.__file__).parent / "chain_casl_pwr.xml")

dt = [25.0] # the delta t
model = pwr.get_model()

# Computing the power density to use
fuel_r=0.3975
power_density = 104
power = power_density * 366  * np.pi * fuel_r**2

# Random stuff - non-strictly-input-based
depletion_materials = depletable_mats_from_model(model=model) # get from starting model
depl_id_list = [this.id for this in depletion_materials]

# Related to the anderson acceleration scheme

THE_NUMBER_OF_SOLVES = 18

THE_ANDERSON_ORDER = 1

THE_SCALE_NPG = 1.5

THE_STARTING_NPG = 10000




nsolves = THE_NUMBER_OF_SOLVES # number of transport solves/solution
andersonOrder = THE_ANDERSON_ORDER
scale_npg = THE_SCALE_NPG # how much to scale npg by at each iteration / transport solve (10000, 15000, 22500, ....)
starting_npg = THE_STARTING_NPG # overwrite the starting histories of the model
aa = Anderson(mr=andersonOrder, tolerance=1e-15, max_solves=nsolves, dummy_transport=False, scale_npg=1.5)
model.settings.particles = starting_npg

"""Make the results folder"""
Path("results").mkdir(parents=True, exist_ok=True)
Path("depl_results").mkdir(parents=True, exist_ok=True)


"""fake dummy transport with a flat/skewed flux --- deplete until t=300 days"""
aa.initialize_bos()
LATEST_FLUX = flat_transport(N=len(depl_id_list))
this_dt = 300.0
time = 300.0
iidx=0
tidx=1
aa.deplete(iidx=iidx, tidx=tidx, x=LATEST_FLUX, depl_mats=depletion_materials, micro_xs=micro_xs, chain_file=chain_file, dt=this_dt, power=power)
# Append depletion solution: gN (corrector) or pN (predictor)
aa._append_pN_iter(time=time, pN=aa.depl_output[-1]) # append predictor number densities
aa._append_time_zero_results()
aa.finalize(time=time, x=LATEST_FLUX, fx=LATEST_FLUX, gx=None, k=1)
depletion_materials = get_depletion_materials_from_results_EOS(output_name=aa.depl_output, model=model)

# Now start with a bad flux for the predictor and iterate.
LATEST_FLUX = oscillating_transport(len(depletion_materials))

"""iterate through time"""
the_eos_time = 300.0
for _tidx, this_dt in enumerate(dt):
  # Time
  TIME_IDX = _tidx + 2
  the_eos_time += this_dt

  aa.initialize_step(time=the_eos_time)
  LATEST_FLUX = aa.solve_step(
    initial_x=LATEST_FLUX,
    tidx=TIME_IDX,
    depl_mats=depletion_materials, model=model, micro_xs=micro_xs,chain_file=chain_file,
    dt=this_dt, power=power, depl_id_list=depl_id_list, time=the_eos_time
  )
  depletion_materials = get_depletion_materials_from_results_EOS(output_name=aa.depl_output, model=model)


