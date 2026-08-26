import data.pwr_rei_template as pwr
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

from SIE import SIE
from Anderson import run_transport, run_transport_for_chain, run_transport_standard

from NuclideVectorMath import get_nuclides_for_transport, \
    make_transport_material_library, \
    get_depletion_materials_from_results_EOS, \
    chain_from_pkl, \
    depletable_mats_from_model, \
    make_transport_material_library, \
    relax_nuclides_from_files

logging.info(f"Now running input....")


# Input stuff
micro_xs = chain_from_pkl(file='data/FINAL_CHAIN.pkl') # get xs from a reference file
chain_file = 'data/chain_casl_pwr.xml'
results_folder = 'results'
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
nsolves = 5 # number of transport solves/solution
sie = SIE()


"""Start by performing t=0 transport"""
RESULTS_TRANSPORT = run_transport_standard(model=model, power_tally_ids=depl_id_list) ## transport w/ batch-by-batch tally tracking
LATEST_FLUX = sie.get_final_tally(res=RESULTS_TRANSPORT, normalize_to=1.0)
sie.finalize_bos(x=LATEST_FLUX)
sie.dump_to_pkl(name=f'results/sie_i{0}_t{0}.pkl')

"""
Now iterating through time.
"""
the_eos_time = 0.0
for TIME_IDX, this_dt in enumerate(dt):
  # Time
  the_eos_time += this_dt

  # Set x and f(x)
  x = [] # X values from relaxed robbins monro algorithm
  fx = [] # Values from the actual transport solves (F(x)) --> fx

  # Solve f(x0)
  logging.info(f"Now running sie.solve, iidx={0}, TIME_IDX={TIME_IDX}, time_EOS={the_eos_time}")
  the_fx = sie.solve(x = LATEST_FLUX, 
                     tidx=TIME_IDX, iidx=0,
                     depl_mats=depletion_materials,
                     model=model, micro_xs=micro_xs,chain_file=chain_file,
                     dt=this_dt, power=power, depl_id_list=depl_id_list)

  # Track and update
  fx.append(copy.deepcopy(the_fx))  
  x.append(sie.get_relaxed_flux(fx=fx))
  LATEST_FLUX = copy.deepcopy(x[-1]) 

  # Now iterate across solves.
  for iidx in range(1, nsolves):
    the_fx = sie.solve(x=LATEST_FLUX, 
                       tidx=TIME_IDX, iidx=iidx,
                       depl_mats=depletion_materials,
                       model=model, micro_xs=micro_xs,chain_file=chain_file,
                       dt=this_dt, power=power, depl_id_list=depl_id_list)
    
    # track and update
    fx.append(copy.deepcopy(the_fx))  
    x.append(sie.get_relaxed_flux(fx=fx))
    LATEST_FLUX = copy.deepcopy(x[-1]) 
    logging.info(f"The f(x) = {the_fx}")
    logging.info(f"The x used to deplete = {x[-1]}")

  # Advances depletion material definitions to EOS values for the next BU step since we are now done iterating
  sie.finalize(time=the_eos_time, x=x, fx=fx)
  sie.dump_to_pkl(name=f'results/sie_i{iidx}_t{TIME_IDX+1}.pkl')
  depletion_materials = get_depletion_materials_from_results_EOS(output_name=sie.depl_output_name, model=model)
  
    
