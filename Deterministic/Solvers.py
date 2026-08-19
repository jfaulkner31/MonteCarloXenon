from Fields import *
from Kernels import *
from Meshing import *
import scipy.linalg

class BasicSolver():
  def __init__(self, kernels: list[Kernel], bcs: list, field: Field):
    """
    Pass in BC objects and Kernels.
    Updates the coefficients of these objects.
    Solves Ax = b
    """
    self.kernels: list[Kernel] = kernels
    self.bcs = bcs
    self.field = field
    self.A = self.kernels[0].aC * 0.0
    self.b = self.kernels[0].b * 0.0

  def solve(self, _dt: float):
    self.A *= 0.0
    self.b *= 0.0
    for k in self.kernels:
      k.update_dt(_dt=_dt)
      k.update_coeffs()
      self.A += k.aC
      self.A += k.aF
      self.b += k.b
    for bc in self.bcs:
      bc.update_coeffs()
      self.A += bc.aC
      self.A += bc.aF
      self.b += bc.b
    self.field.T = scipy.linalg.solve(self.A, self.b)

  def plot_matrix(self):
    # first setup matrix
    self.A *= 0.0
    self.b *= 0.0
    for k in self.kernels:
      k.update_coeffs()
      self.A += k.aC
      self.A += k.aF
      self.b += k.b
    for bc in self.bcs:
      # bc.update_coeffs()
      self.A += bc.aC
      self.A += bc.aF
      self.b += bc.b
    mask = abs(self.A) < 1e-10
    plt.imshow(self.A, cmap='jet')
    plt.colorbar()
    plt.axis('off')  # Optional: hides axis ticks
    plt.show()

  def plot_b(self):
    # first setup matrix
    self.A *= 0.0
    self.b *= 0.0
    for k in self.kernels:
      k.update_coeffs()
      self.A += k.aC
      self.A += k.aF
      self.b += k.b
    for bc in self.bcs:
      # bc.update_coeffs()
      self.A += bc.aC
      self.A += bc.aF
      self.b += bc.b
    plt.plot(self.b, 'ks--', markerfacecolor='w')

class icoSIMPLE:
  """
  A solver for the pressure-momentum equations.
  """
  def __init__(self, vel: FaceField, p: Field, mu: float, mesh: Mesh_1D):
    self.mu = mu
    self.vel = vel
    self.p = p
    self.mesh = mesh

    diff_kernel = DiffusionKernel(field=vel, mesh=mesh, Gamma=self.mu)
    conv_kernel = AdvectionKernel(field=vel)
