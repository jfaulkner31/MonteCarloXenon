import glob
import copy
import logging
import os

import openmc
import openmc.deplete


from NuclideVectorMath import depletable_mats_from_model

"""
COMPUTATIONAL METHODS 
AND 
MISC FUNCTIONS
"""

"""
Gets the chain (e.g. xs that are pre-tallied)
"""
def run_transport_for_chain(model: openmc.Model, chain_file: str):
  """runs an openmc transport calculation for getting the depletion chain"""
  able_mats = depletable_mats_from_model(model) # get the depletable materials (so we can tally)
  fluxes, micros = openmc.deplete.get_microxs_and_flux(model, able_mats, chain_file=chain_file)
  return fluxes, micros

"""
Batch-by-batch transport simulation in OpenMC
"""
def run_transport(model: openmc.Model, power_tally_ids: list):
  """runs an openmc transport calculation while doing batch-by-patch tallies"""
  GARBAGE_RUN = False

  # Clear xml's
  for file in glob.glob("*.xml"):
    os.remove(file)

  if GARBAGE_RUN:
    model.settings.particles = 500

  # Export model to XML
  model.export_to_xml()

  res = {} # contains/stores power tally ids and stuff like that.

  openmc.lib.init() # initialize
  openmc.lib.simulation_init()
  for b in range(model.settings.batches):
    tallies = [openmc.lib.tallies[the_id] for the_id in power_tally_ids]
    openmc.lib.next_batch() 
    results = [tally.results for tally in tallies]
    res[b] = copy.deepcopy(results)
  openmc.lib.simulation_finalize()
  openmc.lib.finalize()
  return res

def run_transport_standard(model: openmc.Model, power_tally_ids: list):
  """
  Standard method of running transport 
  using the usual openmc kernels.

  Returns
  =======
  res : dict[int]->list[Tallies.mean()]
    res[batches]->openmc.Tallies object
  """
  GARBAGE_RUN = False
  
  # Clear xml's
  for file in glob.glob("*.xml"):
    os.remove(file)
  
  if GARBAGE_RUN:
    model.settings.particles = 500

  # Number of batches
  batches = model.settings.batches

  # How to write the source into a file once converged.
  source_name = f"source.{batches}.h5"
  model.settings.sourcepoint = {
    "batches": [batches], # write after the 1000th batch
    "write": True, # Write = true
    "separate": True # Write as a separate file
  }

  model.settings.output = {'tallies': False}

  # If the source exists, lock and load it!
  if Path(source_name).is_file():
    logging.info("Source exists so we are changing the model's starting source!")
    model.settings.source = openmc.FileSource(source_name)
    
  
  model.export_to_xml()
    
  sp_path = f"statepoint.{batches}.h5"
  
  
  logging.info(f"Now running transport with openmc.run() ... ")
  # openmc.run()
  openmc.run()
  res = {}
  
  with openmc.StatePoint(sp_path) as sp:
    tallies: list[openmc.Tally] = [sp.get_tally(id=tid) for tid in power_tally_ids]
    res[batches] = [t.mean.item() for t in tallies]
    the_str_dict = ""
    for key in sp.runtime.keys():
      the_str_dict += f"\t{str(key)}: {float(sp.runtime[key])} \n"
    logging.info(f"\tThe step keff (Combined) = {sp.keff}", )
    logging.info(f"\tThe runtime metrics are:\n {the_str_dict}")
  os.remove(sp_path)
  
  return res
