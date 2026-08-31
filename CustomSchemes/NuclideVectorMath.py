"""
Imports
"""
import openmc
import openmc.deplete
import copy 
import pickle as pkl
import numpy as np

"""
This file is dedicated to 
1. functions that do math and operations on nuclide vectors
2. functions that work with nuclide vectors from depletion
"""

def depletable_mats_from_model(model: openmc.Model) -> openmc.Materials:
  """
  Gets depletion materials from the Model object 
  as an openmc.Materials object
  get depletable materials as openmc.Materials object
  """

  depletable_mats = []
  for this in model.materials:
    if this.depletable:
      depletable_mats.append(this)
  depletable_mats = openmc.Materials(depletable_mats)
  return depletable_mats

def get_nuclides_for_transport(chain_file: str, model: openmc.Model):
  """
  Disgusting function to get nuclides to 
  use/broadcast in transport simulations (addnux functionality basically)
  """
  from openmc.deplete.coupled_operator import _find_cross_sections, _get_nuclides_with_data
  from openmc.deplete.chain import Chain
  chain = Chain.from_xml(chain_file)
  cross_sections = _find_cross_sections(model)
  nuclides_with_data = _get_nuclides_with_data(cross_sections)
  nuclides = [nuc.name for nuc in chain.nuclides
              if nuc.name in nuclides_with_data]
  return nuclides

def make_transport_material_library(output_name: str | openmc.deplete.Results | list[openmc.deplete.StepResult], model: openmc.Model, chain_file: str):
  """
  Function to take in a model, chain, and results file.

  Updates the model.materials to be the transport materials
  with the latest results from results file. Inline modification

  Only considers transport nuclides though.
  """

  # get the results based on the filename
  results = _get_results_from_output_name(output_name=output_name)

  # Make transport material library.
  transport_mats = []

  # Depletables
  trans_nuc_list = get_nuclides_for_transport(chain_file=chain_file, model=model)
  for mat in model.materials:
    if not mat.depletable:
      continue # skip if not depletable

    # Make a new material for the depletables
    new_mat = openmc.Material(mat.id, mat.name, temperature=mat.temperature)
    new_mat.volume = mat.volume
    new_mat.depletable = True
    for nuc in trans_nuc_list:
      perc = results.get_atoms(mat=mat, nuc=nuc, nuc_units='atom/b-cm')[-1][-1]
      new_mat.add_nuclide(nuclide=nuc, percent=perc, percent_type='ao')
      new_mat.set_density(units='sum')
    transport_mats.append(new_mat)

  # Nondepletables, can just append what we have already
  for mat in model.materials:
    if not mat.depletable:
      transport_mats.append(mat)

  new_lib = openmc.Materials(transport_mats)
  # new_lib.export_to_xml()
  model.materials = new_lib

def _get_results_from_output_name(output_name: str | openmc.deplete.Results | list[openmc.deplete.StepResult]) -> openmc.deplete.Results:
  if isinstance(output_name, str):
    results = openmc.deplete.Results(output_name)
  elif isinstance(output_name, openmc.deplete.Results):
    results = output_name
  elif isinstance(output_name, list):
    if isinstance(output_name[0], openmc.deplete.StepResult):
      for idx, _ in enumerate(output_name):
        assert isinstance(output_name[idx], openmc.deplete.StepResult), "Not every entry in the list was an openmc.deplete.StepResult"
    else:
      raise Exception("Output name was a list but the list did not contain openmc.deplete.StepResult's")
    results = output_name
  else:
    raise Exception("output_name must be either a string to a filename or a Results container already")
  return results

def get_depletion_materials_from_results_EOS(output_name: str | openmc.deplete.Results, model: openmc.Model): 
  """
  Function for getting materials for depletion EOS values (or BOS for the next step)
  Returns a list of materials marked depletable
  with full depletion chain.

  No inline modification of models object.
  """
  results = _get_results_from_output_name(output_name=output_name)
  
  depletion_mat_list = []
  for mat in model.materials:
    if mat.depletable:
      eos_mat = results[-1].get_material(str(mat.id))
      assert eos_mat.id == mat.id, \
        "The material obtained from results[-1].get_material()" \
          "is not the same as the material from the depletable material mat.id." \
          "Consider using a results[-1].index_mat to get the proper material index?"
      depletion_mat_list.append(eos_mat)
  return openmc.Materials(depletion_mat_list)

def chain_from_pkl(file: str):
  with open(file, 'rb') as f:
    fakeFluxes, chain = pkl.load(f)
    if len(chain) == 1:
      new_chain = []
      for this in range(16):
        new_chain.append(copy.deepcopy(chain[0]))
      return new_chain
    else:
      return chain

def relax_nuclides_from_files(files: list[str], 
                              weights: np.ndarray[1, float]) -> openmc.deplete.Results:
  """
  Put in a list of file names (depletion.h5) and a list of weights.
  Then relax the nuclides this way

  Parameters
  ==========
  files : list[str]
    list of file names
  weights : 1D np.ndarray
    weights to use for relaxation (in order of files)
  
  Returns
  =======
  relaxed_nuclides : openmc.deplete.Results
    relaxed nuclides following w1*f1 + w2*f2 + w3*f3
    where wN is the weight associated with each file where 
    ||w|| = 1
  """

  assert len(weights) == len(files), \
    "The weights and the files must have the same length!"
  
  ref_file = files[0]

  ref_results = copy.deepcopy(openmc.deplete.Results(ref_file))
  ref_bos = copy.deepcopy(ref_results[0])
  ref_eos: openmc.deplete.StepResult = copy.deepcopy(ref_results[-1])
  ref_nuc_idxs = ref_eos.index_nuc # make sure these match for every step

  file_count = len(files)
  data = ref_eos.data

  # Normalize the weights to 1.0
  weights /= sum(weights)

  # Weight the first entry.
  data *= weights[0]

  # Clear the weight entry we just used
  weights = weights[1:]

  for _, file in enumerate(files[1:]):
    this_eos_result = openmc.deplete.Results(file)[-1]
    assert (this_eos_result.index_nuc == ref_nuc_idxs), \
      "Nuclide ordering in the data array does not match " \
      "for every file being averaged when it should - otherwise" \
      " Jonathon needs to update the averaging procedure!"
    data += this_eos_result.data*weights[0]
    weights = weights[1:]

  ref_results.clear()
  ref_results.append(ref_bos)
  ref_results.append(ref_eos)
  return ref_results

"""
Down here is dedicated to interpreting and averaging some step results.
"""
