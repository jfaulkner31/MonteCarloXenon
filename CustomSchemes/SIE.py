"""
Stochastic implicit euler class for depletion
"""

import CustomSchemes.data.pwr_rei_template as pwr
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
from CustomSchemes.Colors import Colors, nice_grid, nice_legend
import logging 
from CustomSchemes.NuclideVectorMath import relax_nuclides_from_files
openmc.deplete.pool.USE_MULTIPROCESSING=False

class SIE:
  """
  Class that contains useful information 
  for doing stochastic implicit euler depletion
  """
  def __init__(self, relax_N: bool = None, relax_F: bool = None):
    self._x:     dict[float: list[np.ndarray]] = {}      #  x[time] -> x (relaxed phi values from the robbins monro algo.)
    self._fx:    dict[float: list[np.ndarray]] = {}      # fx[time] -> fx (actual solves fromt transport)

    self._N:     dict[float: list[openmc.deplete.Results]] = {} # the N a a result of relaxing
    self._gN:    dict[float: list[openmc.deplete.Results]] = {} # the N as a direct output of the depletion solution
    self._pN:    dict[float: openmc.deplete.Results] = {}       # the N as an outout of the predictor number densities.

    # Plotting and analysis settings for consistency
    self._colors = [Colors.colors()+Colors.colors2()][0]*2
    self._markers = ['s', 'x', 'd', '+', '^', '>', '<']*4

    # Depletion output name tracker
    self._latest_depletion_output: openmc.deplete.Results = None

    # Relaxation settings
    self._relax_N = relax_N
    self._relax_F = relax_F

    # initalized?
    self.initialized = False

  """
  Properties
  """
  @property
  def depl_output(self) -> openmc.deplete.Results:
    if self._latest_depletion_output is None:
      raise Exception("self._latest_depletion_output is None --- has not been set yet oh noooo!s")
    if (not isinstance(self._latest_depletion_output, openmc.deplete.Results)):
      raise Exception("The latest depletion output is not type openmc.deplete.Results!")
    return self._latest_depletion_output
  @property
  def x(self) -> dict[float: list[np.ndarray]]:
    return self._x
  @property
  def fx(self) -> dict[float: list[np.ndarray]]:
    return self._fx
  @property
  def times(self) -> list[float]:
    return [float(key) for key in self._x.keys()]

  def set_depl_output(self, name: openmc.deplete.Results):
    self._latest_depletion_output = name

  def _append_x_iter(self, time: float, x: np.ndarray[tuple[int], float]):
    """appends to x (the flux relaxed solution / guessed solution)"""
    assert self.initialized, "Step must be initialized before we can append to x"
    self._x[time].append(copy.deepcopy(x))

  def _append_fx_iter(self, time: float, fx: np.ndarray[tuple[int], float]):
    """appends to f(x) (the flux solution directly from transport)"""
    assert self.initialized, "Step must be initialized before we can append to f(x)"
    self._fx[time].append(copy.deepcopy(fx))

  def _append_N_iter(self, time: float, N: openmc.deplete.StepResult):
    """append the relaxed step result to the relaxed N dictionary"""
    assert self.initialized, "Step must be initialized before we can append to the relaxed nuclide vector array"
    assert isinstance(N, openmc.deplete.StepResult), "The data we are appending must be a openmc.deplete.StepResult"
    self._N[time].append(copy.deepcopy(N))

  def _append_gN_iter(self, time: float, gN: openmc.deplete.StepResult):
    """append to gN - the unrelaxed nuclide vector from solving Depletion(x)"""
    assert self.initialized, "Step must be initialized before we can append to the relaxed nuclide vector array"
    assert isinstance(gN, openmc.deplete.StepResult), "The data we are appending must be a openmc.deplete.StepResult"
    self._gN[time].append(copy.deepcopy(gN))

  def _append_pN_iter(self, time: float, pN: openmc.deplete.StepResult):
    """append the predictor result to pN"""
    assert self.initialized, "Step must be initialized before we can append to the relaxed nuclide vector array"
    assert isinstance(pN, openmc.deplete.StepResult), "The data we are appending must be a openmc.deplete.StepResult"
    assert time not in self._pN.keys(), f"self._pN already has an entry in self._pN[{time}] " \
      "implying that multiple predictors have been ran or " \
      " the data has been appended to the predictor data container multiple times!"
    self._pN[time] = pN

  def finalize(self):
    """
    Finalizes results after a timestep.
    """
    assert self.initialized, "Step must be initialized before it can be finalized."
    # self._x[time] = x
    # self._fx[time] = fx
    self.initialized = False

  def finalize_bos(self):
    """
    Finalizes the BOS results
    
    Parameters
    ==========
    x : np.ndarray
      1d np array for the BOS fluxes (x0)
    """
    self.finalize()

  def set_bos_solution(self, x: np.ndarray[tuple[int], float]):
    """
    Sets the solution from BOS transport (x) as f(x) and x.

    Parameters
    ==========
    x : np.ndarray[tuple[int], float]
      the flux solution to append
    Returns
    =======
    None
    """
    assert self.initialized, "Cannot set BOS solution unless we have already initialized."
    self._append_fx_iter(time=0, fx=x)
    self._append_x_iter(time=0, x=x)

  def initialize_bos(self):
    """
    INitializes the BOS
    """
    self.initialize_step(time=0.0)

  def initialize_step(self, time: float):
    assert not self.initialized, "Step is trying to be initialized but it was never finalized"
    self._x[time] = []
    self._fx[time] = []
    self._gN[time] = []
    self._N[time] = []
    self.initialized = True

  def dump_to_pkl(self, name: str):
    """
    Dumps self to a pkl file.

    Parameters
    ==========
    name : str
      name of the file to dump to
    """
    with open(name, "wb") as file:
      pkl.dump(self, file)

  def get_from_pkl(self, file: str):
    """
    Returns a SIE object from a pkl file.
    """
    with open(file, 'rb') as f:
      out: SIE = pkl.load(f)
    self._x = out._x
    self._fx = out._fx
    self._latest_depletion_output = out._latest_depletion_output
    self._relax_F = out._relax_F
    self._relax_N = out._relax_N

    self._gN = out._gN
    self._pN = out._pN
    self._N  = out._N
    return self

  def get_final_tally(self, res: dict, normalize_to: float = 1.0):
    """
    Description
    ===========
    Take in results from a batch-wise transport calculation
    
    Parameters
    ==========
    res : dict 
      results obtained from run_transport()
    normalize_to : float = 1.0
      value to normalize the tally to upon output 
    

    Outputs
    =======
    out : dict
      dictionary of tallies by generation 
    """
    maxx = max(list(res.keys()))
    shape1 = np.array(res[maxx])    
    return shape1/np.sum(shape1) * normalize_to
  
  def solve(self, 
            x: np.ndarray,
            tidx: int,
            iidx: int, 
            depl_mats: openmc.Materials,
            model: openmc.Model,
            micro_xs: list,
            chain_file: str,
            dt: float,
            power: float,
            depl_id_list: list[int],
            time: float,
            final_solve: bool) -> np.ndarray[tuple[int], float]:
    """
    Solves corrector + transport. Outputs the relaxed flux solution.

    If no relaxation of the flux, outputs just the most recent flux guess... 
    or otherwise relaxation with a factor of 1.0

    f(x) is the solution to the coupled problem (fluxes):
      f(x) = Transport(Corrector(x))

    x is the relaxed flux guess and iterates: Relax(x1, x2, ... xN)
    
    Parameters
    ==========
    x : np.ndarray
      x vector used to deplete (input x)
    tidx : int
      time index
    iidx : int
      iteration index
    depl_mats : openmc.Materials
      materials to be depleted.
    model : openmc.Model
      model object for transport and depletion
    micro_xs : list[openmc.MicroXS]
      list of openmc micro xs
    chain_file : str
      filename of the depletion chain to use in depletion
    dt : float
      delta time (days)
    power : float
      power input to deplete with
    depl_id_list : list[int]
      list of depletion ids for depletion
    predictor : bool
      tells us if the solve is a predictor calculation or not.

    Outputs
    =======
    fx : np.ndarray
      output result
    """
    assert tidx >= 1, "The time idx for the first depletion step is always greater than 0"

    # Deplete and then interally update self._latest_depletion_output
    self.deplete(iidx=iidx, tidx=tidx, 
                 x=x, depl_mats=depl_mats, 
                 micro_xs=micro_xs, chain_file=chain_file,
                 dt=dt, power=power)
    # Append depletion solution: gN (corrector) or pN (predictor)
    if iidx==0:
      self._append_pN_iter(time=time, pN=self.depl_output[-1]) # append predictor number densities
      if tidx == 1: # append time=0 isotopics
        self._append_time_zero_results()
        
    else:
      self._append_gN_iter(time=time, gN=self.depl_output[-1]) # append the number densities after solving
    
    # Relaxes the nuclide densities if applicable, sets the latest output internally as well
    if self._relax_N:
      # If iidx = 0, this is a predictor calculation and we dont want to relax nuclide densities.
      if iidx != 0:
        self.set_depl_output(self.get_relaxed_n(iidx=iidx, tidx=tidx))
        self._append_N_iter(time=time, N=self.depl_output[-1]) # append the relaxed number densities

    fx = self.transport(model=model, chain_file=chain_file, depl_id_list=depl_id_list)
    self._append_fx_iter(time=time, fx=fx)
  
    # Relax transport using the fx iterates from transport
    if self._relax_F: # relaxation
      x = self.get_relaxed_flux(fx=self.fx[time])
    else: # no relaxation
      x = copy.deepcopy(fx) 
      
    # Append the x (whether relaxed or not) to the x vec.
    self._append_x_iter(time=time, x=x)

    # If final solve, deplete forward one last time with most recent flux solution since this is basically free
    if final_solve:
      self.deplete(iidx=iidx+1, tidx=tidx, 
                  x=x, depl_mats=depl_mats, 
                  micro_xs=micro_xs, chain_file=chain_file,
                  dt=dt, power=power)
      # Append depletion solution: gN (corrector)
      self._append_gN_iter(time=time, gN=self.depl_output[-1]) # append the number densities.
      
      # Relaxes the nuclide densities if applicable, sets the latest output internally as well
      if self._relax_N:
        self.set_depl_output(self.get_relaxed_n(iidx=iidx+1, tidx=tidx))
        self._append_N_iter(time=time, N=self.depl_output[-1]) # append the relaxed number densities
    
    return x
  
  def deplete(self, iidx: int, tidx : int, 
              x : np.ndarray[tuple[int], float],
              depl_mats: openmc.Materials,
              micro_xs: list,
              chain_file: str,
              dt: float,
              power: float):
    """
    Performs depletion from BOS to EOS 
    using the entered flux.

    Parameters
    ==========
    iidx : int
      the iteration index, for Predictor this is 0, for Corrector this is iidx > 0
    tidx : int
      the timestep index. taken from extenral call
    x : np.ndarray[1, float]
      the fluxes/tallies used to deplete.
    depl_mats : list[openmc.Material]
      the list of materials to be depleted.
    micro_xs : list[xs]
      the xs used to deplete.
    chain_file : str
      the chain file used to deplete
    power : float 
      the power used to normalize the fluxes
    dt : float
      the timestep size (days)

    Returns
    =======
    None

    Updates
    =======
    1. updates self._latest_depletion_output using self.set_depl_output

    """
    # Name the depletion output
    depl_output_name = self._get_depletion_output_name(iidx=iidx, tidx=tidx)
    
    # Perform depletion until EOS
    depl_flux = copy.deepcopy(x)
    op = openmc.deplete.IndependentOperator(depl_mats, depl_flux, micro_xs, chain_file=chain_file)
    openmc.deplete.PredictorIntegrator(op, timesteps=[dt], power=power, timestep_units='d').integrate(path=depl_output_name)
    
    # Update the latest depletion output information internally
    self.set_depl_output(openmc.deplete.Results(depl_output_name))

  def transport(self,
                model: openmc.Model, 
                chain_file : str,
                depl_id_list : list[int]) -> np.ndarray[tuple[int], float]:
    """
    Runs transport only, returns fx

    Parameters
    ==========
    model : openmc.Model
      the model in openmc
    chain_file : str
      the chain file we use throughout our calculation
    depl_id_list : list[int]
      list of depletion ids. we use depletion ids for collecting tallies for fluxes from transport
    """
    from CustomSchemes.NuclideVectorMath import make_transport_material_library
    make_transport_material_library(output_name=self.depl_output, model=model, chain_file=chain_file)
    
    # Results from transport 
    from TransportMath import run_transport, run_transport_standard
    tr_dict = run_transport_standard(model=model, power_tally_ids=depl_id_list) ## this one for res tracking...
    fx = self.get_final_tally(res=tr_dict, normalize_to=1.0)
    return fx

  def _append_time_zero_results(self):
    """
    Appends time=0 results 
    in every category.
    """
    self._append_pN_iter(time=0, pN=self.depl_output[0])
    self._append_gN_iter(time=0, gN=self.depl_output[0])
    if self._relax_N:
      self._append_N_iter(time=0, N=self.depl_output[0])

  def get_relaxed_flux(self, fx: list[np.ndarray[float]]) -> np.ndarray[float]:
    """
    Get relaxed flux from a list of fluxes using the weights.
    """
    assert isinstance(fx, list), "fx must be a list"
    if len(fx) == 0:
      raise Exception("Length of fx must be 1 or more!")
    new = np.zeros(len(fx[0]), dtype=float)
    norm_to = np.sum(fx[0])
    for this in fx:
      new += this
    
    new = new / np.sum(new) * norm_to
    return new
  
  def get_relaxed_n(self, iidx: int, tidx: int):
    """
    Gets the relaxed EOS nuclide results based on previous files

    Parameters
    ==========
    iidx : int
      iteration index
    tidx : int
      iteration index

    Returns
    =======
    results : openmc.deplete.Results
      Results object representing the BOS and EOS nuclide densities where EOS have been relaxed.
    """
    from CustomSchemes.NuclideVectorMath import relax_nuclides_from_files

    assert iidx > 0, "iidx must be greater than 0 to relax the nucldies. " \
                     "We do not relax nuclides from the predictor calculations!"

    # Make the filenames from all results thus far
    # NOTE: EOS Predictor Nuclide densities use iidx=0 so we ignore these.
    # NOTE: the files we average are the depletion output name values f(x) values at the end of a depletion solve,
    #       they are NOT the relaxed values.
    # NOTE: in the robbins monro algorithm each depletion solve gets a constant weight.

    # Get all of the files
    files = self._get_corrector_depletion_solve_files(iidx=iidx, tidx=tidx)

    # Make the weights based on the RM algorithm
    weights = self._get_weights_RM(N=iidx)

    # Get the relaxed Results based on a list of filenames.
    results: openmc.deplete.Results = relax_nuclides_from_files(files=files, weights=weights)
    return results


  def _relax_nuclides_from_results():
    pass

  def _get_corrector_depletion_solve_files(self, iidx : int, tidx : int) -> list[str]:
    """
    Gets corrector depletion solve files for all
    values of iidx excluding zero.
    """
    return [self._get_depletion_output_name(iidx=the_i, tidx=tidx) for the_i in range(1, iidx+1)]


  def _get_weights_RM(self, N: int) -> np.ndarray[tuple[int], float]:
    """
    How to weight each iterate in the RM style scheme.

    Uses a uniform weight between each MC iterate/guess
    
    Parameters
    ==========
    N : int 
      number of weights

    Returns
    =======
    w : np.ndarray
      weights (normalized to 1.0)
    """
    assert N > 0, "Number of points to weight with must be more than 0!"
    return np.ones(N)/N
  
  def _get_depletion_output_name(self, iidx: int, tidx: int) -> str:
    """
    Gets depletion output name

    Parameters
    ==========
    iidx : int
      iteration index
    tidx : int
      iteration index

    Returns
    =======
    filename : str
      name of the depletion h5 file

    """
    # PREDICTOR: depl_step_s{TIME_IDX+1}_i{0}.h5 # made to align logically with the transport grid
    depl_output_name = f"depl_step_s{tidx}_i{iidx}.h5" 
    return depl_output_name
    
  """
  Plotting and analysis
  """
  def _time_flag(self, t: float):
    """
    Checks time is valid

    Parameters
    ==========
    t : float
      the time
    
    Returns
    =======
    None

    """    
    if t not in self.times:
      raise ValueError(f"Time input of {t} is not ok / found in self.times!")
    
  def plot_x(self, time: float, dpi=100):
    """
    Plot the progression of x_next. NOT
    including the initial condition

    Parameters
    ==========
    time : float
      the time 
    dpi : int   
      image dpi
    """
    self._time_flag(t=time)

    plt.figure(figsize=(5,3), dpi=dpi)
    for c, x in enumerate(self._x[time]):
      plt.plot(x, '-', markerfacecolor='none', label=f'$x_{c}$', color=self._colors[c], marker=self._markers[c], markersize=4, mew=0.7, lw=0.7)
    
    plt.xlabel('Fissionable zone index')
    plt.ylabel('Flux (normalized)')
    nice_legend()
    nice_grid()

  def plot_fx(self, time: float, dpi=100):
    """
    Plot the progression of f(x) NOT
    including the initial condition.
    f(x) is the transport tallies where x_n+1 = TRANSPORT(CORRECTOR(X_n))

    Parameters
    ==========
    time : float
      the time 
    dpi : int   
      image dpi
    """
    self._time_flag(t=time)

    plt.figure(figsize=(5,3), dpi=dpi)
    for c, x in enumerate(self._fx[time]):
      plt.plot(x, '-', markerfacecolor='none', label=f'$f(x_{c})$', color=self._colors[c], marker=self._markers[c], markersize=4, mew=0.7, lw=0.7)
    
    plt.xlabel('Fissionable zone index')
    plt.ylabel('Flux (normalized)')
    nice_legend()
    nice_grid()

  def plot_all_x(self, dpi: int = 100):
    plt.figure(figsize=(5,3), dpi=dpi)
    for c, t in enumerate(self.times):
      plt.plot(self._x[t][-1], '-', label=f"t={t}", color=self._colors[c], lw=0.7)

    plt.xlabel('Fissionable zone index')
    plt.ylabel('Flux (normalized)')
    cols = int(len(self.times) / 4)
    nice_legend(ncols=cols, fontsize=8)
    nice_grid()

  def plot_x_norm(self, time: float, dpi: int = 100, order: int = 2, yscale: str = 'linear', out: bool = True):
    """
    Plots the L2 in x.
    Note that the first L2 point is based on EOS - BOS.
    While the others are based on EOS iteration values.
    This is because the x is used to evaluate the L2 - so the
    first value will always be bad since it is the 'IC'
    
    Parameters
    ==========
    time : float
      the time  
    dpi : int = 100
      image dpi
    order : int = 2
      order to be used in np.linalg.norm()
    """
    
    self._time_flag(t=time)
    g = []
    for c, x in enumerate(self._x[time]):
      if c == 0:
        x_old = x
        continue
      g.append(np.linalg.norm(x - x_old, ord=order)/np.linalg.norm(x, ord=order))
      x_old = x
    
    plt.figure(figsize=(5,3), dpi=dpi)
    plt.plot(np.linspace(1,len(g), len(g)), g, 'ks-', label=f'L{order}', lw=0.8, markerfacecolor='none')
    nice_grid()
    plt.yscale(yscale)
    plt.xlabel('Iteration')
    plt.ylabel(f'L{order} norm for x')
    
    print(f"L{order} = {g}")
    
  def plot_fx_norm(self, time: float, dpi: int = 100, order: int = 2, yscale: str = 'linear'):
    """
    Plots fx norm using fx(N) - fx(N-1)

    Parameters
    ==========
    time : float
      the time  
    dpi : int = 100
      image dpi
    order : int = 2
      order to be used in np.linalg.norm()
    """
    self._time_flag(t=time)
    g = []
    for c, x in enumerate(self._fx[time]):
      if c == 0:
        x_old = x
        continue
      g.append(np.linalg.norm(x - x_old, ord=order)/np.linalg.norm(x, ord=order))
      x_old = x
    
    plt.figure(figsize=(5,3), dpi=dpi)
    plt.plot(np.linspace(1,len(g), len(g)), g, 'ks-', label='L2', lw=0.8, markerfacecolor='none')
    plt.yscale(yscale)
    nice_grid()
    plt.xlabel('Iteration')
    plt.ylabel(f'L{order} norm for f(x)')

    print(f"L{order} = {g}")


  """
  Getting N and gN by iteration and time.

  N is the relaxed solutions by iteration or at t=time
  gN is the unrelaxed solutions by iterations or at t=time
  """

  def get_gN_by_iteration(self, time: float, nuc: str):
    """
    Gets unrelaxed version of N by iteration at a given point in time.
    This is the nuclide vector used after relaxation is performed.

    Parameters
    ==========
    time : float
      the EOS time

    Returns
    =======
    nuclides : dict
      dict[mat_id] = ?
    """

    the_N_list: list[openmc.deplete.StepResult] = self._gN[time] # list of StepResult's : [StepResult, StepResult ...]
    assert len(the_N_list) > 0, "the length of the N list is only 0"
    assert isinstance(the_N_list[0], openmc.deplete.StepResult), "zeroth entry is not step result"
    return self.__get_mats_from_step_result_list(nuc=nuc, the_N_list=the_N_list)
  
  def get_N_by_iteration(self, time: float, nuc: str):
    """
    Gets relaxed version of N by iteration at a given point in time
    This is the nuclide vector used after relaxation is performed.

    Parameters
    ==========
    time : float
      the EOS time

    Returns
    =======
    nuclides : dict
      dict[mat_id] = ?
    """

    the_N_list: list[openmc.deplete.StepResult] = self._N[time] # list of StepResult's : [StepResult, StepResult ...]
    assert len(the_N_list) > 0, "the length of the N list is only 0"
    assert isinstance(the_N_list[0], openmc.deplete.StepResult), "zeroth entry is not step result"
    return self.__get_mats_from_step_result_list(nuc=nuc, the_N_list=the_N_list)

  def get_N_by_time(self, nuc: str):
    """
    Gets relaxed nuclides (final solution) for all timesteps

    If relax_N is not on, returns the gN solution.

    Parameters
    ==========
    time : float
      the EOS time

    Returns
    =======
    nuclides : dict
      dict[ time: dict[mat_name: value] ]
    time : list[float]
      the timesteps
    """
    time_vec = []
    the_N_list = []
    if self._relax_N:
      the_N = self._N
    else:
      the_N = self._gN
    for time in the_N.keys():
      if len(the_N[time]) > 0:
        assert isinstance(the_N[time][0], openmc.deplete.StepResult), "zeroth entry is not step result"
        time_vec.append(time)
        the_N_list.append(the_N[time][-1])
    return time_vec, self.__get_mats_from_step_result_list(nuc=nuc, the_N_list=the_N_list)

  def get_Nmat_by_time(self, nuc: str, mat_name: str) -> np.ndarray[tuple[int], np.float64]:
    """
    Gets relaxed nuclides (final solution) for all timesteps for a given material name

    If relax_N is not on, returns the gN solution.

    Parameters
    ==========
    time : float
      the EOS time

    Returns
    =======
    nuclides : dict
      dict[ time: dict[mat_name: value] ]
    time : list[float]
      the timesteps
    name : str
      the name of the material - e.g. '11'
    """
    t, n = self.get_N_by_time(nuc=nuc)
    out = []
    for step in n.keys():
      assert mat_name in n[step].keys(), f"The mat name {mat_name} is not a material key name for step = {step}"
      out.append(n[step][mat_name])
    return t, np.array(out)

  def get_Nnames(self):
    """returns the names of the materials"""
    return list(self._gN[ list(self._gN.keys())[0] ][0].index_mat.keys())

  def __get_mats_from_step_result_list(self, nuc: str, the_N_list: list[openmc.deplete.StepResult]):
    """
    Gets a dictionary of mats by iteration.

    Pass in a list of Step results - each one by iteration or ?

    Parameters
    ==========
    nuc : str
      the nuclide
    the_N_list : list[openmc.deplete.StepResult]
      the list of nuclides, each index is an iteration, 
      this is obtained from self._N[time] or self._gN[time]
    
    Returns
    =======
    nuclides : dict
      dict[ iteration: dict[mat_name: value] ]
    """
    ref_mat_ids = the_N_list[0].index_mat # mat name are the keys - np.int64 (idx) are the values

    # Compare key value pairs
    for this in the_N_list:
      assert ref_mat_ids == this.index_mat, "The index mat ids ordering does not match, need to make sure they match or Jonathon needs to implement different indexing"

    # Iteration, StepResult
    out: dict[int: list[np.float64]] = {} # dict[ iteration: [u235_mat0, u235_mat1, .... ] ]
    for it, the_N in enumerate(the_N_list):
      the_mat_list = {}
      for mat_name in ref_mat_ids.keys():
        mat_idx_in_stepresult = ref_mat_ids[mat_name]
        the_mat_list[mat_name] = the_N[mat_idx_in_stepresult, nuc]
      out[it] = the_mat_list
    return out

def load_SIE(file: str) -> SIE:
  """
  Returns a SIE object from a pkl file.
  """
  with open(file, 'rb') as f:
    out = pkl.load(f)
  return out
