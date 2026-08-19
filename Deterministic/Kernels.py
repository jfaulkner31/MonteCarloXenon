"""
Implements a diffusion equation solver.
"""
from Fields import Field, ScalarField
from Meshing import Mesh_1D
import numpy as np
import Eigenvalue as Eigenvalue

""" 
KERNEL BASE DEFINITION
"""

class Kernel:
  def __init__(self, field: Field, mesh: Mesh_1D):
    self.field = field
    self.mesh = mesh
    self.b = None
    self.aC = None # matrix aF
    self.aF = None # matrix aC
    self._dt = None

    # initialize zeros --- really just initializing size of matrices
    self.b = np.zeros(self.mesh.nz)
    if self.mesh.nz > 1:
      for idx in range(self.mesh.nz):
        try:
          self.aC = np.vstack([self.aC, np.zeros(self.mesh.nz)])
          self.aF = np.vstack([self.aF, np.zeros(self.mesh.nz)])
        except:
          self.aC = np.zeros(self.mesh.nz)
          self.aF = np.zeros(self.mesh.nz)
    else:
      self.aC = np.zeros((1,1))
      self.aF = np.zeros((1,1))

  def update_coeffs(self):
    self.get_aC()
    self.get_aF()
    self.get_b()
  def get_aC(self):
    pass
  def get_aF(self):
    pass
  def get_b(self):
    pass
  def update_dt(self, _dt: float):
    self._dt = _dt
  def plot_matrix(self):
    self.update_coeffs()
    A = abs(self.aC + self.aF)
    mask = A < 1e-10
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # plt.axis('off')  # Optional: hides axis ticks
    plt.show()

  def plot_b(self):
    self.update_coeffs()
    plt.plot(self.b, 'ks-', markerfacecolor='w')

"""
Diffusion Kernel
"""
class DiffusionKernel(Kernel):
  def __init__(self, field, mesh, Gamma: float):
    super().__init__(field, mesh)
    self.Gamma = Gamma # diffusion coefficient

  def get_aC(self):
    # assigns diagonal coefficients
    # reset aC
    self.aC *= 0.0

    # iterate for every cell now
    for cid in self.mesh.cidList:
      if self.mesh.cells[cid].upperType == 'f': # if not boundary
        self.aC[cid, cid] -= -1.0*(self.Gamma * self.mesh.cells[cid].upperNeighborGdiff)

      if self.mesh.cells[cid].lowerType == 'f': # if not boundary
        self.aC[cid, cid] -= -1.0*(self.Gamma * self.mesh.cells[cid].lowerNeighborGdiff)

  def get_aF(self):
    # assigns offdiagonal coefficients
    for cid in self.mesh.cidList:

      # assign upper
      if self.mesh.cells[cid].upperType == 'f':
        self.aF[cid, self.mesh.cells[cid].upperNeighborCid] = -self.Gamma * self.mesh.cells[cid].upperNeighborGdiff

      # assign lower
      if self.mesh.cells[cid].lowerType == 'f':
        self.aF[cid, self.mesh.cells[cid].lowerNeighborCid] = -self.Gamma * self.mesh.cells[cid].lowerNeighborGdiff

  def get_b(self):
    pass ## do nothing for this.

class ImplicitReactionKernel(Kernel):
  """
  Implicit Reaction Source term: lambda * PHI
  """
  def __init__(self, field: ScalarField, mesh: Mesh_1D, lam: float):
    super().__init__(field, mesh)
    self.lam = lam

  def get_aC(self):
    self.aC *= 0.0
    for cid in self.mesh.cidList:
      self.aC[cid,cid] = self.lam * self.mesh.cells[cid].vol

  def get_aF(self):
    pass
  def get_b(self):
    pass

class ExplicitSourceKernel(Kernel):
  """
  Explicit Source Kernel: Q
  Source is Q in terms of source/m3
  Beta is a multiplier - e.g. delayed neutron fraction that just scales the source.
  """
  def __init__(self, field: ScalarField, mesh: Mesh_1D, Q: float | np.ndarray | ScalarField, beta: float):
    super().__init__(field=field, mesh=mesh)
    self.Q = Q
    self.beta = beta
    self.scaling_factor = 1.0

  def get_aC(self):
    pass
  def get_aF(self):
    pass
  def get_b(self):
    try:
      self.scaling_factor + 1
    except:
      self.scaling_factor = 1.0

    # RESET b
    self.b *= 0.0

    # ITERATE THROUGH b
    for cid in self.mesh.cidList:

      # SCALAR FIELD SOURCE
      if isinstance(self.Q, ScalarField):
        self.b[cid] = self.Q.T[cid] * self.mesh.cells[cid].vol * self.beta * self.scaling_factor

      # NP ARRAY SOURCE
      elif isinstance(self.Q, np.ndarray):
        self.b[cid] = self.Q[cid] * self.mesh.cells[cid].vol * self.beta * self.scaling_factor

      # FLOAT SOURCE
      elif isinstance(self.Q, float):
        self.b[cid] = self.Q * self.mesh.cells[cid].vol * self.beta * self.scaling_factor
      elif isinstance(self.Q, int):
        self.b[cid] = float(self.Q) * self.mesh.cells[cid].vol * self.beta * self.scaling_factor
      # EXCEPTION UNKNOWN TYPE
      else:
        raise Exception("Unknown ExplicitSource type for self.Q")

  def get_integrated_source(self):
    """
    Returns int(self.Q * dV * scaling_factor)
    """
    try:
      self.scaling_factor + 1
    except:
      self.scaling_factor = 1.0

    out = 0.0
    v_total = 0.0
    # if it is just a float --
    if isinstance(self.Q, float):
      for cid in self.mesh.cidList:
        _v = self.mesh.cells[cid].vol
        out += _v * self.Q
        v_total += _v

    elif isinstance(self.Q, np.ndarray):
      for cid in self.mesh.cidList:
        _v = self.mesh.cells[cid].vol
        out += _v * self.Q[cid]
        v_total += _v

    elif isinstance(self.Q, Field):
        _v = self.mesh.cells[cid].vol
        out += _v * self.Q.T[cid]
        v_total += _v

    else:
      raise ValueError("Q type must be np array, float, or a Field for explicit source kernel")

    return out * self.scaling_factor

  def get_averaged_src_density(self):
    """
    Gets averaged source density ---
    int(self.Q * dV) / int(dV)
    """
    try:
      self.scaling_factor + 1
    except:
      self.scaling_factor = 1.0

    out = 0.0
    v_total = 0.0
    # if it is just a float --
    if isinstance(self.Q, float):
      for cid in self.mesh.cidList:
        _v = self.mesh.cells[cid].vol
        out += _v * self.Q
        v_total += _v

    elif isinstance(self.Q, np.ndarray):
      for cid in self.mesh.cidList:
        _v = self.mesh.cells[cid].vol
        out += _v * self.Q[cid]
        v_total += _v

    elif isinstance(self.Q, Field):
        _v = self.mesh.cells[cid].vol
        out += _v * self.Q.T[cid]
        v_total += _v

    else:
      raise ValueError("Q type must be np array, float, or a Field for explicit source kernel")

    return out / v_total * self.scaling_factor

  def get_Q_as_array(self) -> np.ndarray:
    """returns source Q as a np array."""
    if isinstance(self.Q, float):
      return np.ones(len(self.mesh.cidList)) * self.Q
    elif isinstance(self.Q, list):
      return np.array(self.Q)
    elif isinstance(self.Q, np.ndarray):
      return self.Q
    else:
      raise Exception("Not allowed type")

  def set_Q(self, Q: float | np.ndarray | ScalarField, scaling_factor: float):
    self.Q = copy.deepcopy(Q)
    self.scaling_factor = scaling_factor

  def set_scaling_factor(self, scaling_factor: float):
    self.scaling_factor = scaling_factor


class FissionSource(Kernel):
  """
  Source for the fission term in the eigenvalue equation.

  field : Field
    flux field
  keff : Eigenvalue
    the eigenvalue object
  keff

  """
  def __init__(self, field: Field, mesh: Mesh_1D, keff: Eigenvalue.Eigenvalue):
    super().__init__(field, mesh)
    if not isinstance(keff, Eigenvalue.Eigenvalue):
      raise Exception("keff in the fission source term was not an eigenvalue kernel")
    self.keff = keff

  def get_b(self):
    """
    Gets fission source NSF
    """
    


