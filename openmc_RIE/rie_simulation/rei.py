import pwr_rei_template as pwr
import openmc
import openmc.deplete
import numpy as np
import copy
import pickle as pkl
import glob
import os
openmc.deplete.pool.USE_MULTIPROCESSING=False

"""
Depletion scheme for explicit euler. Meant to just get cross sections at the very end for reuse later.
"""

class Regression():
  def __init__(self):
    pass
  def get_avg(self, res: dict, val: float):
    """
    gets average from a results dict.
    Note that res must be ran through self.tally_by_gen
    before the averages can be computed.

    Parameters
    ==========
    res : dict
      batch-by-batch estimate of the flux (each batch represents a single trial)
    val : float
      what to normalize the output flux to

    Returns
    =======
    flux : np.array
      the normalized average flux (norm'd across all gens)
    """
    flux = np.zeros(len(res[0]))
    for key in res.keys():
      # Get the normalized flux (this batches estimate for flux)
      the_unnorm_flux = res[key]
      sum_unnorm = np.sum(the_unnorm_flux)
      if sum_unnorm > 0:
        the_normd_flux = the_unnorm_flux/sum_unnorm * val
        flux += the_normd_flux
    flux *= val / np.sum(flux)
    return flux

  def write_res_pkl(self, res: dict, file:str):
    with open(file, 'wb') as f:
      pkl.dump(res, f)

  def tally_by_gen(self, res: dict):
    """does a quick cleanup after running transport"""
    theLength = res[0].__len__()
    shape0 = np.zeros(theLength)
    d = {}
    """nice little function to get tallies by gen"""
    for key in res.keys():
      shape1 = np.array([ this[:,:,1][0][0] for this in res[key][0:theLength] ])
      shape = shape1 - shape0
      d[key] = shape
      # advance
      shape0 = shape1
    return d

  def normalize_res(self, res: dict, val: float):
    """normalizes res dictionary to some value"""
    for key in res.keys():
      the_sum = np.sum(res[key])
      if the_sum == 0:
        res[key] = res[key]
      else:
        res[key] = res[key] / the_sum * val
    return res

  def _get_vij(self, N: int, start: int, F: list, I: list,
              i: int, j: int, f: int):
    Vij = 0.0
    for k in range(start,start+N):
      Vij += (F[i][k][f] - I[i][f])*(F[j][k][f] - I[j][f]) / (N-1)
    return Vij

  def get_new_I(self, N: int, start: int,
                F: list, I: list,
                f: int):
    """
    Parameters
    ==========
    N : int
      number active histories

    start : int
      starting aka nSkipped

    F : list
      list of res dicts (F1, F2, F3, ...)

    I : list
      list of final I values (normalized?)

    f : int
      tally id number to operate on.

    Returns
    =======
    new_I : float
      the new value of I corresponding to tally id f
    coeff : float
      the correlation coefficient

    """
    n = len(F)
    V = np.zeros((n,n))
    for i in range(n):
      for j in range(n):
        V[i,j] = self._get_vij(N=N, start=start, F=F, I=I, i=i, j=j, f=f) # returns vectror of Vij where each index is a fuel zone

    print("the matrix V is")
    print(V)
    if n == 2:
      # print correlation a and b
      coeff = V[1,0]/V[0,0]**0.5/V[1,1]**0.5
      print("the correlation coeff = ", coeff)
    else:
      coeff = 0
    the_I = np.array([[this[f] for this in I]]).transpose() # col vec of my best estimates.
    X = np.ones((n,1)) # col vec of ones.
    M = np.linalg.inv(X.transpose() @ np.linalg.inv(V) @ X) @ X.transpose() @ np.linalg.inv(V)
    print("\nThe matrix is: ", M, "\n")
    new_I = M @ the_I
    return new_I, coeff

def run_transport_for_chain(model: openmc.Model, chain_file: str):
  """runs an openmc transport calculation for getting the depletion chain"""
  able_mats = depletable_mats_from_model(model) # get the depletable materials (so we can tally)
  fluxes, micros = openmc.deplete.get_microxs_and_flux(model, able_mats, chain_file=chain_file)
  return fluxes, micros

def run_transport(model: openmc.Model, power_tally_ids: list):
  """runs an openmc transport calculation while doing batch-by-patch tallies"""
  # Clear xml's
  for file in glob.glob("*.xml"):
    os.remove(file)
  # export model
  model.export_to_xml()
  res = {} # contains/stores power tally ids and stuff like that.
  settings = model.settings

  openmc.lib.init()
  openmc.lib.simulation_init()
  for b in range(settings.batches):
    tallies = [openmc.lib.tallies[the_id] for the_id in power_tally_ids]
    openmc.lib.next_batch()
    results = [tally.results for tally in tallies]
    res[b] = copy.deepcopy(results)
  openmc.lib.simulation_finalize()
  openmc.lib.finalize()
  return res

def depletable_mats_from_model(model: openmc.Model) -> openmc.Materials:
  """get depletable materials as openmc.Materials object"""
  depletable_mats = []
  for this in model.materials:
    if this.depletable:
      depletable_mats.append(this)
  depletable_mats = openmc.Materials(depletable_mats)
  return depletable_mats

def get_nuclides_for_transport(chain_file: str, model: openmc.Model):
  from openmc.deplete.coupled_operator import _find_cross_sections, _get_nuclides_with_data
  from openmc.deplete.chain import Chain
  chain = Chain.from_xml(chain_file)
  cross_sections = _find_cross_sections(model)
  nuclides_with_data = _get_nuclides_with_data(cross_sections)
  nuclides = [nuc.name for nuc in chain.nuclides
              if nuc.name in nuclides_with_data]
  return nuclides

def make_transport_material_library(output_name: str, model: openmc.Model, chain_file: str):
  """
  Function to take in a model, chain, and reults file.

  Updates the model.materials to be the transport materials
  with the latest results from results file. Inline modification

  Only considers transport nuclides though.
  """

  # Make transport material library.

  results = openmc.deplete.Results(output_name)
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

def get_depletion_materials_from_results_EOS(output_name: str, model: openmc.Model):
  """
  Function for getting materials for depletion EOS values (or BOS for the next step)
  Returns a list of materials marked depletable
  with full depletion chain.

  No inline modification of models object.
  """
  results = openmc.deplete.Results(output_name)
  depletion_mat_list = []
  for mat in model.materials:
    if mat.depletable:
      eos_mat = results[-1].get_material(str(mat.id))
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


"""
Some random settings
"""
fuel_r=0.3975
power_density = 104
power = power_density * 366  * np.pi * fuel_r**2
dt = [0.5, 1, 1.5, 2, 5, 10, 10, 10, 10,
25, 25, 25, 25,
25, 25, 25, 25,
25, 25, 25, 25] # up to 350 days
iterations = 1 # number of iterations we run (so +1 to this for total # transports)
Nstart = 500
Nactive = 500
iter_strategy = 'regression' # 'const_relax' OR set iterations to 0 to straight burn

"""
Get the model
"""
model = pwr.get_model()
chain_file = '../chain_casl_pwr.xml'

"""
Regression class
"""
regr =  Regression()

"""
Run explicit euler depletion scheme
"""
depletion_materials = depletable_mats_from_model(model=model) # get from starting model
depl_id_list = [this.id for this in depletion_materials]
print(depl_id_list)
# raise Exception("kill")

"""
T0 transport and get fluxes
"""
d = run_transport(model=model, power_tally_ids=depl_id_list) ## this one for res tracking...
res = regr.tally_by_gen(res=d)
res_normd = regr.normalize_res(res=res, val=1.0)
fluxes = regr.get_avg(res=res_normd, val=1.0)
# _fakefluxes, micro_xs = run_transport_for_chain(model=model, chain_file=chain_file) # todo load reference xs here instead
micro_xs = chain_from_pkl(file='../chain_gen/FINAL.pkl') # get xs from a reference file

for idx, this_dt in enumerate(dt):
  res_list = []
  flux_this_step_list = []
  I = []
  for ni in range(iterations+1):
    """Start w/ predicting forward in time with most recent flux estimate"""
    # Deplete (operator setup and then deplete)
    output_name = f"depl_step_s{idx}_i{ni}.h5"
    print("Now depleting with flux =", fluxes)
    op = openmc.deplete.IndependentOperator(depletion_materials, fluxes, micro_xs, chain_file=chain_file)
    openmc.deplete.PredictorIntegrator(op, timesteps=[this_dt], power=power, timestep_units='d').integrate(path=output_name)

    # Now update the transport materials (inline modify model.materials)
    make_transport_material_library(output_name=output_name, model=model, chain_file=chain_file)
    d = run_transport(model=model, power_tally_ids=depl_id_list) ## this one for res tracking...
    res =  regr.tally_by_gen(res=d)
    regr.write_res_pkl(res=res, file=f'res_s{idx}_i{ni}.pkl')
    res_normd = regr.normalize_res(res=res, val=1.0)
    fluxes = regr.get_avg(res=res_normd, val=1.0)
    I.append(copy.deepcopy(fluxes))
    res_list.append(res_normd)

    # Corrector fluxes or keep them as is (correct if ni is above 0)
    if ni > 0:
      if iter_strategy == 'regression':
        fluxes = []
        coefs= []
        for f in range(len(depl_id_list)): # go thorugh fList and get all the newly predicted fluxes.. note that tally id's have same ids as depl materials
          flux_f, coef = regr.get_new_I(N=Nactive, start=Nstart, F=res_list, I=I, f=f)
          coefs.append(coef)
          fluxes.append(flux_f[0][0])
        fluxes = np.array(fluxes)
        fluxes *= 1.0 / np.sum(fluxes)
      elif iter_strategy == 'const_relax': # constant relaxation of current fluxes and the previously relaxed fluxes.
        prev = flux_this_step_list[-1]
        fluxes = 0.7*prev + 0.3*fluxes # 1 iteration, relaxation factor of 0.3  # TODO fix this but this works for now for only 1 iteration
        fluxes *= 1.0 / np.sum(fluxes)
    else:
      fluxes = fluxes # we cant relax since we have only 1 solution.... (impl euler.)
    
    # fluxes at this BU step. (this will naturally be all the T1 fluxes.)
    flux_this_step_list.append(fluxes)

  # Depletion finalize with final flux values.
  output_name = f"depl_step_s{idx}_final.h5"
  print("Now doing final depletion CORRECTOR and depleting with flux =", fluxes)
  op = openmc.deplete.IndependentOperator(depletion_materials, fluxes, micro_xs, chain_file=chain_file)
  openmc.deplete.PredictorIntegrator(op, timesteps=[this_dt], power=power, timestep_units='d').integrate(path=output_name)

  # Now make the depletion materials BOS for the next depletion step...
  depletion_materials = get_depletion_materials_from_results_EOS(output_name=output_name, model=model)
