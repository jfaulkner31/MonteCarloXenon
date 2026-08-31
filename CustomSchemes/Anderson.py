"""
Imports and related stuffs
"""
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
from Colors import Colors, nice_grid, nice_legend
import logging 
from SIE import SIE
from NuclideVectorMath import *


openmc.deplete.pool.USE_MULTIPROCESSING=False

"""
Anderson acceleration
depletion schemes with 
implicit Euler timestepping
"""
class Anderson():
  """
  Class that contains useful information 
  for doing anderson acceleration depletion
  """
  def __init__(self):
    self._x:     dict[float: list[np.ndarray]] = {}      #  x[time] -> x
    self._fx:    dict[float: list[np.ndarray]] = {}      # fx[time] -> fx
    self._g:     dict[float: list[np.ndarray]] = {}      #  g[time] -> g
    self._k: dict[float: int] = {}
    self._latest_depletion_output_name: str = None 

    # Plotting and analysis settings for consistency
    self._colors = [Colors.colors()+Colors.colors2()][0]*2
    self._markers = ['s', 'x', 'd', '+', '^', '>', '<']*4


  """
  Class getters
  """
  @property 
  def times(self) -> list[float]:
    return [float(key) for key in self._x.keys()]
  @property
  def x(self) -> dict[float: list[np.ndarray]]:
    return self._x
  @property
  def fx(self) -> dict[float: list[np.ndarray]]:
    return self._fx
  @property
  def g(self) -> dict[float: list[np.ndarray]]:
    return self._g
  @property
  def k(self) -> dict[float: int]:
    return self._k
  @property
  def depl_output_name(self) -> str:
    if self._latest_depletion_output_name is None:
      raise Exception("self._latest_depletion_output_name is None --- has not been set yet oh noooo!s")
    return self._latest_depletion_output_name
  
  """
  Class methods for computing
  """
  def finalize_bos(self, x: np.ndarray):
    """
    Finalizes the BOS results
    
    Parameters
    ==========
    x : np.ndarray
      1d np array for the BOS fluxes (x0)
    """
    self.finalize(time=0.0, x=[x], fx=[], g=[], k=-1)

  def finalize(self, 
               time: float, 
               x: list[np.ndarray], 
               fx: list[np.ndarray], 
               g: list[np.ndarray], 
               k: int):
    """
    Finalizes results after a timestep.
    """
    if time in self.times:
      raise Exception("Time already in self.times, cannot overwrite.")
    self._x[time] = x
    self._fx[time] = fx
    self._g[time] = g
    self._k[time] = k

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
    out : np.ndarray
      fluxes result (normalized) 
    """
    theLength = res[0].__len__()
    shape0 = np.zeros(theLength)
    d = {}
    """nice little function to get the very last tally"""
    maxx = max(list(res.keys()))
    for key in [maxx]: # Cheat code to just get the last generation tall
      shape1 = np.array([ this[:,:,1][0][0] for this in res[key][0:theLength] ])
    shape1 = shape1/np.sum(shape1) * normalize_to
    return shape1
  
  def tally_by_gen(self, res: dict):
    """
    Description
    ===========
    Take in results from a batch-wise transport calculation
    and outputs tallies by generation
    
    Parameters
    ==========
    res : dict 
      results obtained from run_transport()

    Outputs
    =======
    out : dict
      dictionary of tallies by generation 
    """
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
            depl_id_list: list[int]) -> np.ndarray:
    """
    Solves corrector + transport. Outputs f(x)

    f(x) is the solution to the coupled problem:
      f(x) = Transport(Corrector(x))

    Essentially an evaluation of f(x)
    
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
    depl_output_name : str
      file name for the depletion EOS output.

    Outputs
    =======
    fx : np.ndarray
      output result
    """
    # Name the depletion output
    depl_output_name = f"depl_step_s{tidx+1}_i{iidx}.h5" # PREDICTOR: depl_step_s{TIME_IDX+1}_i{0}.h5 # made to align logically with the transport grid
    
    # Perform depletion until EOS
    depl_flux = copy.deepcopy(x)
    op = openmc.deplete.IndependentOperator(depl_mats, depl_flux, micro_xs, chain_file=chain_file)
    openmc.deplete.PredictorIntegrator(op, timesteps=[dt], power=power, timestep_units='d').integrate(path=depl_output_name)
    
    # Update the latest depletion output name internally
    self.set_depl_output_name(depl_output_name)

    make_transport_material_library(output_name=depl_output_name, model=model, chain_file=chain_file)
    
    # Results from transport 
    tr_dict = run_transport(model=model, power_tally_ids=depl_id_list) ## this one for res tracking...
    fx = self.get_final_tally(res=tr_dict, normalize_to=1.0)
    return fx
  
  def set_depl_output_name(self, name: str):
    self._latest_depletion_output_name = name


  """
  Class methods for analysis 
  """
  def get_from_pkl(self, file: str):
    """
    Returns an Anderson object from a pickle file.
    """
    with open(file, 'rb') as f:
      out: Anderson = pkl.load(f)
    self._x = out._x
    self._fx = out._fx
    self._g = out._g
    self._k = out._k
    self._latest_depletion_output_name = out._latest_depletion_output_name
    return self

  def plot_x(self, time: float, dpi=100):
    """
    Plot the progression of x_next
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

  def plot_final_x(self, time: float, label: str, dpi=100, coloridx: int = 0):
    c = coloridx
    self._time_flag(t=time)

    plt.figure(figsize=(5,3), dpi=dpi)
    plt.plot(self._x[time][-1], '-', markerfacecolor='none', label=label, color=self._colors[c], marker=self._markers[c], markersize=4, mew=0.7, lw=0.7)
    
    plt.xlabel('Fissionable zone index')
    plt.ylabel('Flux (normalized)')
    nice_legend()
    nice_grid()


  def plot_fx(self, time: float, dpi=100, include_xfinal: bool = True, include_average: bool = True):
    """
    Plot the progression of x_next
    including the initial condition

    Parameters
    ==========
    time : float
      the time 
    dpi : int   
      image dpi
    include_xfinal : bool
      whether or not to include the final x_next value (the final guess)
    include_average : bool
      whether or not to include the average result
    """
    self._time_flag(t=time)

    plt.figure(figsize=(5,3), dpi=dpi)
    for c, x in enumerate(self._fx[time]):
      plt.plot(x, '-', markerfacecolor='none', label=f'$fx_{c}$', color=self._colors[c], marker=self._markers[c], markersize=4, mew=0.7, lw=0.7)
    if include_xfinal:
      plt.plot(self._x[time][-1], 'kx--', label=r'$x_\mathrm{final}$', mew=0.7, lw=0.7, markersize=3)
    if include_average:
      avg = np.zeros(len(self._fx[time][0]))
      for v in self._fx[time]:
        avg += v
      avg = avg / len(self._fx[time])
      plt.plot(avg / sum(avg) * sum(self._fx[time][0]), 'rs', label=r'$\mathrm{AVG(}f(x)\mathrm{)}$', mew=0.7, lw=0.7, markerfacecolor='none', markersize=2)


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
  Other
  """
  def _time_flag(self, t: float):
    if t not in self.times:
      raise ValueError(f"Time input of {t} is not ok / found in self.times!")

"""
Processing norms with time
"""
def get_norm_vs_time(res: Anderson | SIE, ref: Anderson | SIE, order: int = 2) -> np.ndarray[float]:
  
  """
  Gets the norm (difference) as a function of time.
  """
  all_norms = []
  for t in ref.times:
    try:
      res._time_flag(t)
    except:
      break

    xref = ref.x[t][-1]
    xres = res.x[t][-1]

    nrm = np.linalg.norm(xres-xref, ord=order) / np.linalg.norm(xref, ord=order)
    all_norms.append(nrm)
  return np.array(all_norms)

