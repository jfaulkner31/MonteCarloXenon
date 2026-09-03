import CustomSchemes.data.pwr_symmetric_template as pwr
from CustomSchemes.DumbTransport import flat_transport, oscillating_transport
import CustomSchemes
import openmc
import openmc.deplete
import numpy as np
import copy
import pickle as pkl
import glob
import os
import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

openmc.deplete.pool.USE_MULTIPROCESSING=False

"""
Setup logging
"""
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', filemode='a', filename="LOGGING.LOG")
logging.getLogger().setLevel(logging.DEBUG)

from CustomSchemes.SIE import SIE
from CustomSchemes.TransportMath import run_transport, run_transport_for_chain, run_transport_standard

from CustomSchemes.NuclideVectorMath import get_nuclides_for_transport, \
    make_transport_material_library, \
    get_depletion_materials_from_results_EOS, \
    chain_from_pkl, \
    depletable_mats_from_model, \
    make_transport_material_library, \
    relax_nuclides_from_files

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

# Robbins Monro related
nsolves = 100 # number of transport solves/solution
sie = SIE(relax_N=False, relax_F=True)

"""Make the results folder"""
Path("results").mkdir(parents=True, exist_ok=True)
Path("depl_results").mkdir(parents=True, exist_ok=True)


"""Start by performing t=0 transport"""
sie.initialize_bos()
RESULTS_TRANSPORT = run_transport_standard(model=model, power_tally_ids=depl_id_list) ## transport w/ batch-by-batch tally tracking
LATEST_FLUX = sie.get_final_tally(res=RESULTS_TRANSPORT, normalize_to=1.0)
sie.set_bos_solution(x=LATEST_FLUX)
sie.finalize_bos()
sie.dump_to_pkl(name=f'results/sie_i{0}_t{0}.pkl')


"""fake dummy transport with a skewed flux - deplete until 300 days"""
sie.initialize_bos()
LATEST_FLUX = flat_transport(N=len(depl_id_list))
this_dt = 300.0
time = 300.0
iidx=0
tidx=1
sie.deplete(iidx=iidx, tidx=tidx, x=LATEST_FLUX, depl_mats=depletion_materials, micro_xs=micro_xs, chain_file=chain_file, dt=this_dt, power=power)
sie._append_pN_iter(time=time, pN=sie.depl_output[-1])
sie._append_time_zero_results()
sie.finalize()



"""
Now iterating through time.
"""
the_eos_time = 300.0
for _tidx, this_dt in enumerate(dt):
  sie._append_x_iter(time=325.0, x=LATEST_FLUX)  
  # Time
  TIME_IDX = _tidx + 2
  the_eos_time += this_dt
  sie.initialize_step(time=the_eos_time)
  final_solve = False

  # Now iterate across solves.
  for iidx in range(0, nsolves):
    if iidx == nsolves-1:
      final_solve = True
    x = sie.solve(x=LATEST_FLUX, 
                  tidx=TIME_IDX, iidx=iidx, time=the_eos_time,
                  depl_mats=depletion_materials,
                  model=model, micro_xs=micro_xs,chain_file=chain_file,
                  dt=this_dt, power=power, depl_id_list=depl_id_list, final_solve=final_solve)
    # TODO: make sure relaxation of the fluxes is carried out correctly here. [x]
    # TODO: make sure that we are properly iterating power and the nuclide vector here. [x]
    # TODO: make sure that we are correctly doing predictor separation here [x] 
    # TODO: make sure that we can set nsolves to 1 and get the correct behavior here. [x]

    # track and update
    LATEST_FLUX = copy.deepcopy(x) 


  # Advances depletion material definitions to EOS values for the next BU step since we are now done iterating
  sie.finalize()
  sie.dump_to_pkl(name=f'results/sie_i{iidx}_t{TIME_IDX}.pkl')
  depletion_materials = get_depletion_materials_from_results_EOS(output_name=sie.depl_output, model=model)
  
    
