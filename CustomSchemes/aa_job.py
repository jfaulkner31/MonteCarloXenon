import CustomSchemes
import CustomSchemes.data.pwr_rei_template as pwr
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

dt = [0.5, 1, 1.5, 2, 5, 10, 10, 10, 10,
25, 25, 25, 25,
25, 25, 25, 25,
25, 25, 25, 25] # up to 350 days
model = pwr.get_model()

# Computing the power density to use
fuel_r=0.3975
power_density = 104
power = power_density * 366  * np.pi * fuel_r**2

# Random stuff - non-strictly-input-based
depletion_materials = depletable_mats_from_model(model=model) # get from starting model
depl_id_list = [this.id for this in depletion_materials]

# Robbins Monro related
fake_transport = False
nsolves = 4 # number of transport solves/solution
andersonOrder = 2
aa = Anderson(mr=andersonOrder, tolerance=1e-15, max_solves=4, dummy_transport=fake_transport)

"""Make the results folder"""
Path("results").mkdir(parents=True, exist_ok=True)
Path("depl_results").mkdir(parents=True, exist_ok=True)

"""Start by performing t=0 transport"""
aa.initialize_bos()

# actual transport
if fake_transport:
  LATEST_FLUX = dummy_transport(N=len(depl_id_list))
# actual transport
else:
  RESULTS_TRANSPORT = run_transport_standard(model=model, power_tally_ids=depl_id_list) ## transport w/ batch-by-batch tally tracking
  LATEST_FLUX = aa.get_final_tally(res=RESULTS_TRANSPORT, normalize_to=1.0)

aa.finalize_bos(x=[LATEST_FLUX])
aa.dump_to_pkl(name=aa.get_pickle_name(k=0, tidx=0))

"""iterate through time"""
the_eos_time = 0.0
for _tidx, this_dt in enumerate(dt):
  # Time
  TIME_IDX = _tidx + 1
  the_eos_time += this_dt

  aa.initialize_step(time=the_eos_time)
  LATEST_FLUX = aa.solve_step(
    initial_x=LATEST_FLUX, 
    tidx=TIME_IDX, 
    depl_mats=depletion_materials, model=model, micro_xs=micro_xs,chain_file=chain_file,
    dt=this_dt, power=power, depl_id_list=depl_id_list, time=the_eos_time
  )
  depletion_materials = get_depletion_materials_from_results_EOS(output_name=aa.depl_output, model=model)

