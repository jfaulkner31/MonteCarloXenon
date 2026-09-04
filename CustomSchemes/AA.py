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

# TODO: make sure that we are handliong chceks and BSO and EOS properly when appending data.

class Anderson():
  def __init__(self, mr: int = None, tolerance: float = None, max_solves: int = None, dummy_transport = False, scale_npg: float = 1):
    """
    Anderson acceleration class

    mr : int
      anderson order - e.g. AA2, AA3, etc
    tolerance : float
      tolerance for convergence criteria
    max_solves : int
      max number of transport solves in a given step
    dummy_transport : bool
      replaces the transport solver with a dummy variable
    """
    assert isinstance(mr, int) | (mr is None), "mr must be an int"
    assert isinstance(tolerance, float) | isinstance(tolerance, int)  | (tolerance is None), "tolerance must be a float or an int"
    assert isinstance(max_solves, int) | (max_solves is None), "max solves must be an int"
    assert isinstance(dummy_transport, bool) | (dummy_transport is None), "dummy transport must be a bool"
    if mr is not None:
      assert mr > 1, "mr must be more than 1"
    if tolerance is not None:
      assert tolerance > 0, "tolerance must be more than 0"
    if max_solves is not None:
      assert max_solves > 2, "Max number of solves must be more than 2 when using anderson acceleration"

    # Flux solutions / versions of the flux solutions
    self._x:  dict[float: list[np.ndarray]] = {} # x is from anderson acceleration
    self._fx: dict[float: list[np.ndarray]] = {} # fx is from solving the transport equation
    self._gx: dict[float: list[np.ndarray]] = {} # gx is just differences in soln's kinda
    self._k: dict[int] = {}                      # iteration k after solving.
    self._npg: dict[list[int]] = {}                    # number of histories (npg) used to solve each iteration at each timestep

    # Nuclide vector solutions
    self._gN:    dict[float: list[openmc.deplete.Results]] = {} # the N as a direct output of the depletion solution (correctors)
    self._pN:    dict[float: list[openmc.deplete.REsults]] = {} # the N that are used as predictor N

    # Plotting and analysis settings for consistency
    self._colors = [Colors.colors()+Colors.colors2()][0]*2
    self._markers = ['s', 'x', 'd', '+', '^', '>', '<']*4

    # Depletion output name tracker
    self._latest_depletion_output: openmc.deplete.Results = None

    # initalized?
    self.initialized = False

    # AA stuff:
    self._scale_npg: float = scale_npg
    self._original_npg: int = None
    self._mr = mr
    self._tolerance = tolerance
    self._max_solves = max_solves
    self._dummy_transport = dummy_transport
    self._latest_gamma: dict[float: np.ndarray] = {}
    self._latest_alpha: dict[float: np.ndarray] = {}

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
  def gx(self) -> dict[float: list[np.ndarray]]:
    return self._gx
  @property
  def times(self) -> list[float]:
    return [float(key) for key in self._x.keys()]
  @property
  def _relax_N(self) -> bool:
    return False
  @property
  def mr(self) -> int:
    return self._mr
  @property
  def k(self) -> dict[int]:
    return self._k

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

  def _append_gx_iter(self, time: float, gx: np.ndarray[tuple[int], float]):
    """appends to g(x)"""
    assert self.initialized, "Step must be initialized before we can append to f(x)"
    self._gx[time].append(copy.deepcopy(gx))

  """appending Nuclide information"""
  def _append_N_iter(self, time: float, N: openmc.deplete.StepResult):
    """append the relaxed step result to the relaxed N dictionary"""
    assert False, "The anderson accelerations scheme has no self._N parameter currently, we have not implemented nuclide relaxation yet!"
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

  """appending npg information"""
  def _append_npg_information(self, time: float, npg: int):
    assert self.initialized, "Step must be initialized before we can append npg information"
    assert isinstance(npg, int), "npg must be an int"
    self._npg[time].append(npg)

  def finalize(self, time: float, x: list[np.ndarray[tuple[int], float]], fx: list[np.ndarray[tuple[int], float]], gx: list[np.ndarray[tuple[int], float]], k: int):
    """
    Finalizes results after a timestep.
    """
    assert self.initialized, "Step must be initialized before it can be finalized."
    self._x[time] = x
    self._fx[time] = fx
    self._gx[time] = gx
    self._k[time] = k
    self.initialized = False

  def finalize_bos(self, x: np.ndarray[tuple[int], float]):
    """
    Finalizes the BOS results

    Parameters
    ==========
    None

    Returns
    =======
    None
    """
    self.finalize(time=0, x=copy.deepcopy(x), fx=copy.deepcopy(x), gx=None, k=-1)

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
    self._append_gx_iter(time=0, gx=None)

  def initialize_bos(self):
    """
    INitializes the BOS
    """
    self.initialize_step(time=0.0)

  def initialize_step(self, time: float):
    assert not self.initialized, "Step is trying to be initialized but it was never finalized"
    self._x[time] = []
    self._fx[time] = []
    self._gx[time] = []
    self._gN[time] = []
    self._k[time] = None
    self._npg[time] = []
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

  def get_pickle_name(self, k: int, tidx: int) -> str:
    return f'results/aa{self._mr}_i{k}_t{tidx}.pkl'

  def get_from_pkl(self, file: str):
    """
    Returns a SIE object from a pkl file.
    """
    with open(file, 'rb') as f:
      out: Anderson = pkl.load(f)
    self._x = out._x
    self._fx = out._fx
    self._gx = out._gx
    self._k = out._k

    self._latest_depletion_output = out._latest_depletion_output

    self._mr = out._mr
    self._tolerance = out._tolerance
    self._max_solves = out._max_solves

    self._gN = out._gN
    self._pN = out._pN

    self._latest_alpha = out._latest_alpha
    self._latest_gamma = out._latest_gamma

    try:
      self._npg = out._npg
    except:
      self._npg = None
    try:
      self._scale_npg = out._scale_npg
    except:
      self._scale_npg = None
    try:
      self._original_npg = out._original_npg
    except:
      self._original_npg = None
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

  def _solve_iteration(self,
            x: np.ndarray, # x value to use for depletion
            tidx: int,
            iidx: int,
            depl_mats: openmc.Materials,
            model: openmc.Model,
            micro_xs: list,
            chain_file: str,
            dt: float,
            power: float,
            depl_id_list: list[int],
            time: float) -> np.ndarray[tuple[int], float]:
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
    # Assertations

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

    # Run transport and get fx for this transport solve
    self._append_npg_information(time=time, npg=model.settings.particles) # append before solving
    fx = self.transport(model=model, chain_file=chain_file, depl_id_list=depl_id_list)
    self._increase_npg_in_model(model=model)


    return fx

  def solve_step(self,
            initial_x: np.ndarray, # initial conditon for fluxes
            tidx: int,
            depl_mats: openmc.Materials,
            model: openmc.Model,
            micro_xs: list,
            chain_file: str,
            dt: float,
            power: float,
            depl_id_list: list[int],
            time: float) -> np.ndarray[tuple[int], float]:
    """
    Solves a depletion step using anderson acceleration

    Input is initial_x which is the fluxes to use for the predictor calculation
    """
    assert tidx >= 1, "The time idx for the first depletion step is always greater than 0"

    # If original NPG hasnt been set yet.
    if self._original_npg is None:
      self._original_npg = model.settings.particles
    else: # reset the npg in the model to the original value before we do any solving at this timestep
      self._reset_model_npg(model=model)

    x = [copy.deepcopy(initial_x)]
    fx = []
    g = []

    fx1 = self._solve_iteration(x=x[-1], tidx=tidx, iidx=0, depl_mats=depl_mats, model=model, micro_xs=micro_xs, chain_file=chain_file, dt=dt, power=power, depl_id_list=depl_id_list, time=time)
    x.append(copy.deepcopy(fx1))
    fx.append(copy.deepcopy(fx1))
    g.append(x[1]- x[0])

    fx2 = self._solve_iteration(x=fx1, tidx=tidx, iidx=1, depl_mats=depl_mats, model=model, micro_xs=micro_xs, chain_file=chain_file, dt=dt, power=power, depl_id_list=depl_id_list, time=time)
    fx.append(copy.deepcopy(fx2))
    g.append(fx[1]-fx[0])

    # Make Matrices G_k and X_k
    d = len(initial_x)
    G_k = (g[1] - g[0]).reshape(d, 1)
    X_k = (x[1] - x[0]).reshape(d, 1)

    breakTheLoop = False
    k = int(2)

    while True:
      m_k = min(k, self.mr)
      x_next = self._solve_lst_sq(G_k=G_k, X_k=X_k, d=d, k=k, g=g, x=x, time=time)

      # Loop breakage
      if breakTheLoop:
        # append since we computed a new x_next
        x.append(x_next)
        self._set_latest_alpha(time=time, p=G_k.shape[1])

        # Final depletion calculation using x_next (basically free)
        self.deplete(iidx=k, tidx=tidx, x=x_next, depl_mats=depl_mats, micro_xs=micro_xs, chain_file=chain_file, dt=dt, power=power)
        self.finalize(time=time, x=x, fx=fx, gx=g, k=k)
        self.dump_to_pkl(name=self.get_pickle_name(k=k, tidx=tidx))
        return x_next

      fx_next = self._solve_iteration(x=x_next, tidx=tidx, iidx=k, depl_mats=depl_mats, model=model, micro_xs=micro_xs, chain_file=chain_file, dt=dt, power=power, depl_id_list=depl_id_list, time=time)

      g_next = fx_next - x_next

      x.append(x_next)
      fx.append(fx_next)
      g.append(g_next)

      X_k, G_k = self._update_matrices(G_k=G_k, X_k=X_k, d=d, k=k, m_k=m_k, g=g, x=x)

      # Converged or finished?
      breakTheLoop = self._did_converge(gk=g[k],numFX=len(fx))

      # Advance
      k += 1

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
    if not self._dummy_transport:
      from CustomSchemes.NuclideVectorMath import make_transport_material_library
      make_transport_material_library(output_name=self.depl_output, model=model, chain_file=chain_file)

      # Results from transport
      from CustomSchemes.TransportMath import run_transport, run_transport_standard
      tr_dict = run_transport_standard(model=model, power_tally_ids=depl_id_list) ## this one for res tracking...
      fx = self.get_final_tally(res=tr_dict, normalize_to=1.0)
    else: # dummy transport
      from CustomSchemes.DumbTransport import transport
      fx = transport(N=len(depl_id_list))

    return fx

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

  def _append_time_zero_results(self):
    """
    Appends time=0 results
    in every category.
    """
    self._append_pN_iter(time=0, pN=self.depl_output[0])
    self._append_gN_iter(time=0, gN=self.depl_output[0])
    if self._relax_N:
      self._append_N_iter(time=0, N=self.depl_output[0])

  def _did_converge(self, gk: np.ndarray, numFX: list) -> bool:
    """check if it converged"""
    # Convergence criteria
    nrm = np.linalg.norm(gk, ord=2)

    converged = False
    if (abs(nrm) < self._tolerance) | (numFX >= self._max_solves): # fx is the number of transport solves up to this point
      converged = True
    return converged

  def _update_matrices(self, G_k: np.ndarray, X_k: np.ndarray, d: tuple, k: int, m_k: int, g: list[ np.ndarray[tuple[int], float] ], x: list[ np.ndarray[tuple[int], float] ]) -> tuple[np.ndarray, np.ndarray]:
    """updates/trims matrices and outputs them"""
    # Stacking operation
    X_k = np.hstack([X_k, (x[k] - x[k-1]).reshape(d, 1)])
    G_k = np.hstack([G_k, (g[k] - g[k-1]).reshape(d, 1)])

    # Keep only last m_k columns
    ncols = X_k.shape[1]
    if ncols > m_k:
      X_k = X_k[:, ncols - m_k:]
      G_k = G_k[:, ncols - m_k:]

    return X_k, G_k

  def _set_latest_gamma(self, time: float, gamma_k: np.ndarray):
    self._latest_gamma[time] = gamma_k

  def _set_latest_alpha(self, time: float, p: int):
    alpha = self._compute_alpha(time=time, p=p)
    self._latest_alpha[time] = alpha

  def _solve_lst_sq(self, G_k: np.ndarray, X_k: np.ndarray, d: tuple, k: int, g: list[ np.ndarray[tuple[int], float] ], x: list[ np.ndarray[tuple[int], float] ], time: float):
    """solves the least squares problem"""
    # Solve least squares: min || G_k gamma - g_k ||_2
    Q, R = np.linalg.qr(G_k, mode='reduced')      # Q:(d,p), R:(p,p)
    rhs = Q.T @ g[k-1].reshape(d, 1)              # (p,1)
    gamma_k = np.linalg.lstsq(R, rhs, rcond=None)[0]  # (p,1)
    self._set_latest_gamma(time=time, gamma_k=gamma_k)

    # Get intermediate x_next
    x_next = x[k-1] + g[k-1] - ((X_k + G_k) @ gamma_k).reshape(d)
    return copy.deepcopy(x_next)

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
    depl_output_name = f"depl_results/depl_step_s{tidx}_i{iidx}.h5"
    return depl_output_name

  def _compute_alpha(self, time: float, p: int = None):
    """
    Convert the last Anderson 'difference-form' coefficients gamma_k (length p)
    from your update
        x_next = f(x_{k-1}) - sum_j gamma_j (f_{j}-f_{j-1})
    into the conventional affine weights alpha (length p+1) such that
        x_next = sum_{i=0}^p alpha_i f(x_{k-1-p+i}),
    with sum(alpha)=1.

    Parameters
    ==========
    gamma_k : array_like, shape (p,) or (p,1)
        The most recent gamma returned by your least-squares solve.
    p : int or None
        Optional: number of columns used (usually G_k.shape[1]).
        If None, inferred from gamma_k length.

    Returns
    =======
    alpha : ndarray, shape (p+1,)
        Conventional Anderson weights in time order:
        [weight for oldest f(x), ..., weight for newest f(x)].
    """
    gamma_k = copy.deepcopy(self._latest_gamma[time])
    gamma = np.asarray(gamma_k, dtype=float).reshape(-1)
    if p is None:
        p = gamma.size
    if gamma.size != p:
        raise ValueError(f"gamma has length {gamma.size} but p={p}")

    alpha = np.empty(p + 1, dtype=float)
    alpha[0] = gamma[0]
    if p > 1:
        alpha[1:p] = gamma[1:] - gamma[:-1]
    alpha[p] = 1.0 - gamma[-1]
    return alpha

  def reconstruct_final_x_next_from_alpha(self, time: float):
    assert time in self._latest_alpha.keys(), "time is invalid"
    assert time in self.fx.keys(), "time is invalid"
    alpha = self._latest_alpha[time]

    starting_fx_idx = int(-1) - len(alpha) + 1

    reconstructed_x = np.zeros(len(self.fx[time][0]))
    for a in alpha:
      reconstructed_x += self.fx[time][starting_fx_idx] * a
      starting_fx_idx += 1
    return reconstructed_x

  """
  Getting x and fx by iteration and time
  """
  def get_x_by_iteration(self, time: float) -> list[np.ndarray[tuple[int], float]]:
    """
    Gets x by iteration for a given point in time

    Parameters
    ==========
    time : float
      the time

    Returns
    =======
    fx : list[np.ndarray]
      the fx solutions for a given point in time
    """
    self._time_flag(t=time)

    return self.x[time]

  def get_x_by_time(self) -> tuple[list[float], list[np.ndarray[tuple[int], float]]]:
    """
    Gets last value of x as a function of time

    Parameters
    ==========
    None

    Returns
    =======
    time : list[float]
      self.times
    x : list[np.ndarray]

    """
    x = []
    time = []
    for t in self.times:
      time.append(t)
      x.append(self.x[t][-1])
    return time, x

  def get_fx_by_iteration(self, time: float) -> list[np.ndarray[tuple[int], float]]:
    """
    Gets fx by iteration for a given point in time

    Parameters
    ==========
    time : float
      the time

    Returns
    =======
    fx : list[np.ndarray]
      the fx solutions for a given point in time
    """
    self._time_flag(t=time)
    return self.fx[time]


  """
  NPG related
  """
  def _set_model_npg(self, npg: int, model: openmc.Model):
    """updates npg inline"""
    model.settings.particles = npg

  def _reset_model_npg(self, model: openmc.Model):
    assert self._original_npg is not None, "self._original_npg must have been set!!!"
    model.settings.particles = self._original_npg

  def _increase_npg_in_model(self, model: openmc.Model):
    """scales the npg in the model - inline modification of model object"""
    npg = model.settings.particles
    new = round(npg*self._scale_npg)
    self._set_model_npg(npg=new, model=model)


  """
  General functions / Other
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
