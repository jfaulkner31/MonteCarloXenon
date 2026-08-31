import data.pwr_rei_template as pwr
import openmc
import openmc.deplete
import numpy as np
import copy
import pickle as pkl
import glob
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
openmc.deplete.pool.USE_MULTIPROCESSING=False
from Anderson import Anderson, run_transport, run_transport_for_chain, \
  depletable_mats_from_model, get_nuclides_for_transport, make_transport_material_library, \
  get_depletion_materials_from_results_EOS, chain_from_pkl

"""
RUN MAIN DEPLETION SEQUENCE
"""

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

# AA related
mr = 4 # number of past iterations to use in AA
max_solves = 5 # results in 6 values of x_next and 5 evaluations of the transport kernel
tol_res = 1e-6
aa = Anderson()


"""
BEGIN DEPLETION SCHEME
"""

"""Clobber results and make new results folder"""
# script_dir = Path(__file__).resolve().parent
# try:
#   shutil.rmtree(script_dir / results_folder)
# except:
#   pass
# os.makedirs(script_dir / results_folder, exist_ok=True)


"""Start by performing t=0 transport"""
RESULTS_TRANSPORT = run_transport(model=model, power_tally_ids=depl_id_list) ## transport w/ batch-by-batch tally tracking
LATEST_FLUX = aa.get_final_tally(res=RESULTS_TRANSPORT, normalize_to=1.0)
aa.finalize_bos(x=LATEST_FLUX)

the_eos_time = 0.0
for TIME_IDX, this_dt in enumerate(dt):
  # Time
  the_eos_time += this_dt
  
  # Set x and g and f(x). 
  x =  [copy.deepcopy(LATEST_FLUX)] # construct x using initial conditions
  fx = []
  g =  [] # construct g at this timestep.

  # Solve f(x1)
  fx1 = aa.solve(x = LATEST_FLUX, 
                 tidx=TIME_IDX, 
                 iidx=1,  # the resulting index of x (this will be x[1] upon solving)
                 depl_mats=depletion_materials,
                 model=model, micro_xs=micro_xs,chain_file=chain_file,
                 dt=this_dt, power=power, depl_id_list=depl_id_list)
        
  x.append(copy.deepcopy(fx1))
  fx.append(copy.deepcopy(fx1))
  g.append(x[1]-x[0])

  # Solve f(x2)
  fx2 = aa.solve(x = fx1, 
                 tidx=TIME_IDX, 
                 iidx=2,  # the resulting index of x (this will be x[2] upon solving)
                 depl_mats=depletion_materials,
                 model=model, micro_xs=micro_xs,chain_file=chain_file,
                 dt=this_dt, power=power, depl_id_list=depl_id_list)
  fx.append(copy.deepcopy(fx2))
  g.append(fx[1] - fx[0])

  # Matrices G_k ad X_k
  d = LATEST_FLUX.size
  G_k = (g[1] - g[0]).reshape(d, 1)
  X_k = (x[1] - x[0]).reshape(d, 1)
  
  breakDaLoop, k = False, int(2)
  while True:
    m_k = min(k,mr)

    # Solve least squares: min || G_k gamma - g_k ||_2
    Q, R = np.linalg.qr(G_k, mode='reduced')      # Q:(d,p), R:(p,p)
    rhs = Q.T @ g[k-1].reshape(d, 1)              # (p,1)
    gamma_k = np.linalg.lstsq(R, rhs, rcond=None)[0]  # (p,1)

    # Get intermediate x_next
    x_next = x[k-1] + g[k-1] - ((X_k + G_k) @ gamma_k).reshape(d)

    # Break the loop and update flux to x_next if so before running more transport
    if breakDaLoop:
      x.append(x_next)
      LATEST_FLUX = copy.deepcopy(x_next) # Use x_next instead of fx_next
      aa.finalize(time=the_eos_time, x=x, fx=fx, g=g, k=k)
      aa.dump_to_pkl(name=f"{results_folder}/t{TIME_IDX+1}_k{k}.pkl")
      break

    fx_next = aa.solve(x = x_next, 
                       tidx=TIME_IDX, 
                       iidx=k+1,  # the resulting index of x (this will be x[1] upon solving)
                       depl_mats=depletion_materials,
                       model=model, micro_xs=micro_xs,chain_file=chain_file,
                       dt=this_dt, power=power, depl_id_list=depl_id_list)
    g_next = fx_next - x_next

    # NOTE: fx gets fx_next (f(n_next)) and x gets x_next (so they are NOT the same after all)
    x.append(x_next)
    fx.append(fx_next)
    g.append(g_next)

    # Update increment matrices
    X_k = np.hstack([X_k, (x[k] - x[k-1]).reshape(d, 1)])
    G_k = np.hstack([G_k, (g[k] - g[k-1]).reshape(d, 1)])

    # Keep only last m_k columns
    ncols = X_k.shape[1]
    if ncols > m_k:
      X_k = X_k[:, ncols - m_k:]
      G_k = G_k[:, ncols - m_k:]

    # Convergence criteria
    nrm = np.linalg.norm(g[k], ord=2)
    if (abs(nrm) < tol_res) | (len(fx) >= max_solves): # fx is the number of transport solves up to this point
      breakDaLoop = True
    
    # Advance k
    k += 1
  #### while keepGoing:

  # Advances depletion material definitions to EOS values for the next BU step since we are now done iterating
  depletion_materials = get_depletion_materials_from_results_EOS(output_name=aa.depl_output_name, model=model)
  

# NOTE / TODO: I feel like we should really be breaking after x_next ? 
# idk, seems more logical since x_next is supposedly more stable solution


