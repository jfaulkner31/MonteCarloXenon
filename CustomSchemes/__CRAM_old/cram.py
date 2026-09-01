import numpy as np
from scipy.linalg import expm

ALPHA16 = [+2.1248537104952237480e-16 + 0.0*1j,
         +5.0901521865224915650e-7 -2.4220017652852287970e-5*1j,
         -2.1151742182466030907e-4 +4.3892969647380673918e-3*1j,
         -4.1023136835410021273e-2 -1.5743466173455468191e-1*1j,
         +1.4793007113557999718e+0 +1.7686588323782937906e0*1j,
         -1.5059585270023467528e+1 -5.7514052776421819979e0*1j,
         +6.2518392463207918892e+1 -1.1190391094283228480e1*1j,
         -1.1339775178483930527e+2 +1.0194721704215856450e2*1j,
         +6.4500878025539646595e+1 -2.2459440762652096056e2*1j
         ]

THETA16 = [+1.0843917078696988026e+1 + 1.9277446167181652284e1*1j,
         +5.2649713434426468895e+0 + 1.6220221473167927305e1*1j,
         +1.4139284624888862114e+0 + 1.3497725698892745389e1*1j,
         -1.4193758971856659786e+0 + 1.0925363484496722585e1*1j,
         -3.5091036084149180974e+0 + 8.4361989858843750826e0*1j,
         -4.9931747377179963991e+0 + 5.9968817136039422260e0*1j,
         -5.9481522689511774808e+0 + 3.5874573620183222829e0*1j,
         -6.4161776990994341923e+0 + 1.1941223933701386874e0*1j]

ALPHA16 = np.array(ALPHA16)
THETA16 = np.array(THETA16)

ALPHA4 = np.array([
  8.652240695288853e-5 +  0.0j,
  -7.339595716394208e-2 + 4.499999224740631e-1j,
  6.168677956783283e-2 - 1.905040979303084e0j,
])

THETA4 = np.array([
  3.678453861815378e-1 + 3.658121298678667e0j,
  -1.548393223297123e0 + 1.191822946627425e0j
])

class CoupledDepleter():
  """
  Performs depletion of multiple mixtures at once using a block matrix.
  """
  def __init__(self):
    pass
  def _solve():
    """Solver"""
    pass
  def solve():
    """
    Public solve function
    """


class Depleter():
  """
  Performs depletion based on input.

  Links to a CRAM solver.

  Parameters
  ==========
  A : np.array (2D)
    2D depletion matrix

  dt : float
    Time step size in seconds
  """
  def __init__(self, A: np.ndarray, dt: float):
    self.A = A
    self.dt = dt
    self._crammer = CRAM(order=16)
    self._Crammed = self._crammer.solve_C(A=A, dt=dt, method="inversion")

  def solve(self, N0: np.ndarray):
    """
    Performs a solve.

    Parameters
    ==========
    N0 : np.array() (1D)
      Initial nuclide vector

    Returns
    =======
    N : np.array() (1D)
      Final nuclide vector
    """
    self.N1 = self._Crammed @ N0

    return self.N1


class CRAM():
  def __init__(self, order=16):
    if order == 16:
      self.alpha = ALPHA16
      self.theta = THETA16
    elif order == 4:
      self.alpha = ALPHA4
      self.theta = THETA4
    else:
      raise ValueError("Only 16th order CRAM is implemented")
    self.order = order
    self.F_vec = None
    self._get_F_l_K()
    self._C = None

  def _get_F_l_K(self):
    F_vec = np.zeros(int(self.order/2))
    for L in range(int(self.order/2)):
      if L == 0:
        F_vec[L] += self.alpha[0].real
      F_vec[L] -= 2.0 * [np.sum(self.alpha[1:]/self.theta**(L+1))][0].real
    self.F_vec = F_vec

  def solve_C(self, A: np.ndarray, dt: float, method: str):
    """Solve the matrix exponential using CRAM.

    Args:
      A: The matrix to exponentiate. Must be square.

    Returns:
      exp(A)
    """

    if method == "sum":
      n = A.shape[0]
      C_new = np.zeros([n,n])
      for L in range(int(self.order/2)):
        C_new += self.F_vec[L] * np.linalg.matrix_power(A * dt, L)
        self._C = C_new
      return C_new
    elif method == "inversion":
      n = A.shape[0]
      I = np.eye(n)
      C_new = np.zeros([n,n], dtype=complex)
      for i in range(int(self.order/2)):
        C_new += self.alpha[i+1] * np.linalg.inv(A * dt - self.theta[i] * I)
      C_new = 2.0 * C_new.real + I * self.alpha[0].real
      self._C = C_new
      return C_new.real
    else:
      raise ValueError("Method must be 'sum' or 'inversion'")

  def get_loss(self, A: np.ndarray, lam: np.ndarray, n0: np.ndarray, way: str):

    if way == 'me':
      # Very poor for small eigenvalues due to large terms in inv(A)
      n = A.shape[0]

      return (self._C - np.eye(n)) @ np.linalg.inv(A)

    elif way == 'chat':
      n = A.shape[0]
      # Build Van Loan matrix
      Mblk = np.block([[A, np.eye(n)],
                      [np.zeros((n, n)), np.zeros((n, n))]])
      E = expm(Mblk * 3600)
      return E[:n, n:]



    # return dN.real
