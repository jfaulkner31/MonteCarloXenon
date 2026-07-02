"""
A python function with the 'normal' aa scheme. 
Validates x_next given outputs of solve().

Basically validates that the scheme I implemented in aa_job.py is doing the right thing and properly solving.
"""

"""
Imports
"""
from Anderson import Anderson
from Colors import Colors, nice_grid, nice_legend

import matplotlib.pyplot as plt
import copy
import numpy as np

"""
Function that 'solves'
Interrogates output of the openmc version to grab solutions for the flux.
This basically feeds solutions from openmc back in - so each should yield the same solution.
"""
def solve(t: float, solveNumber: int, file: str):
  # Get the Anderson object
  aa: Anderson = Anderson().get_from_pkl(file=file)

  # Check
  if t not in aa.times:
    raise Exception("Unknown time!")
  if aa.k[t] <= 0:
    raise Exception("Bad k (did you enter 0.0 as the time? 0.0 doesnt iterate so it's k value is -1 and has no iterations to check!)")
  
  if solveNumber == 0:
    return aa.x[t][0] # get the initial conditions
  else:
    return aa.fx[t][solveNumber-1] # return the solve number
  # Get the solve

def get_MC_soln(t: float, file: str):
  """
  Gets the last x_next value from the MC run.
  """
  aa: Anderson = Anderson().get_from_pkl(file=file)
  # Check
  if t not in aa.times:
    raise Exception("Unknown time!")
  if aa.k[t] <= 0:
    raise Exception("Bad k (did you enter 0.0 as the time? 0.0 doesnt iterate so it's k value is -1 and has no iterations to check!)")
  
  return aa.x[t][-1]

def get_all_fx(t: float, file: str) -> list[np.ndarray[float]]:
  aa: Anderson = Anderson().get_from_pkl(file=file)
  if t not in aa.times:
    raise Exception("Unknown time!")
  if aa.k[t] <= 0:
    raise Exception("Bad k (did you enter 0.0 as the time? 0.0 doesnt iterate so it's k value is -1 and has no iterations to check!)")

  return aa.fx[t]


def get_second_last_xnext(t: float, file: str):
  """
  Gets the second to last xnext so we can observe how substantially it changes.
  """
  aa: Anderson = Anderson().get_from_pkl(file=file)
  # Check
  if t not in aa.times:
    raise Exception("Unknown time!")
  if aa.k[t] <= 0:
    raise Exception("Bad k (did you enter 0.0 as the time? 0.0 doesnt iterate so it's k value is -1 and has no iterations to check!)")
  
  return aa.x[t][-2]

def get_last_fx(t: float, file: str):
  """
  Gets the last fx value from the MC run. This is NOT the x_next we validate.
  """
  aa: Anderson = Anderson().get_from_pkl(file=file)
  # Check
  if t not in aa.times:
    raise Exception("Unknown time!")
  if aa.k[t] <= 0:
    raise Exception("Bad k (did you enter 0.0 as the time? 0.0 doesnt iterate so it's k value is -1 and has no iterations to check!)")
  
  return aa.fx[t][-1]

def anderson_alpha_from_last_gamma(gamma_k, p=None):
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


"""
USER INPUT
"""  

TIME = 0.5 
FILE = 'cluster_results/t1_k4.pkl' 
k_max = 3 # number of solves?
m=4 # 

def validate_case(time: float, file: str, k_max: int, m: int, dpi: int = 100):
  
  FILE = file
  TIME = time

  """ 
  Vector-capable version of the Wikipedia/MATLAB Anderson acceleration code.
  - x can be scalar or vector.
  - f must return same shape as x (for vectors, elementwise maps are fine).
  - Stops when ||g_k|| <= tol_res, where g_k = f(x_k) - x_k.
  """

  # Get initial conditions
  x0 = solve(t=TIME, solveNumber=0, file=FILE) 

  x0 = np.asarray(x0, dtype=float)
  scalar: bool = (x0.ndim == 0)
  if scalar:
    x0 = x0.reshape(1)  # treat scalar as 1-vector internally

  d = x0.size

  # Cache f(x) values to avoid duplicate evaluations
  fx0 = np.asarray(solve(t=TIME, solveNumber=1, file=FILE), dtype=float).reshape(-1) # NOTE: SOLVE
  if fx0.size != d:
    raise ValueError(f"f(x0) returned size {fx0.size}, expected {d}")

  x = [x0.copy(), fx0.copy()]          # X0 initial condition and the first solution
  fx = [fx0.copy(), np.asarray(solve(t=TIME, solveNumber=2, file=FILE), 
                              dtype=float).reshape(-1)] # NOTE: SOLVE again
  g  = [x[1] - x[0], fx[1] - x[1]]     # g0 = f(x0)-x0 = x1-x0, g1 = f(x1)-x1

  # Increment matrices (d x p)
  G_k = (g[1] - g[0]).reshape(d, 1)
  X_k = (x[1] - x[0]).reshape(d, 1)

  k, breakDaLoop, andersonAlphas = 2, False, None
  while True:
    m_k = min(k, m)

    # Solve least squares: min || G_k gamma - g_k ||_2
    Q, R = np.linalg.qr(G_k, mode='reduced')      # Q:(d,p), R:(p,p)
    rhs = Q.T @ g[k-1].reshape(d, 1)              # (p,1)
    gamma_k = np.linalg.lstsq(R, rhs, rcond=None)[0]  # (p,1)

    # Wikipedia update
    x_next = x[k-1] + g[k-1] - ((X_k + G_k) @ gamma_k).reshape(d)

    if breakDaLoop:
      x.append(x_next)
      andersonAlphas  = anderson_alpha_from_last_gamma(gamma_k=copy.deepcopy(gamma_k), p=G_k.shape[1])
      break

    fx_next = np.asarray(solve(t=TIME, solveNumber=k+1, file=FILE), 
                        dtype=float).reshape(-1) # this is the solve function.
    if fx_next.size != d:
      raise ValueError(f"f(x) returned size {fx_next.size}, expected {d}")
    g_next = fx_next - x_next

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
    
    # CHeck if we keep iterating
    if (len(fx) >= k_max):
      breakDaLoop = True
      


    k += 1
    

  x_star = x[-1]
  if scalar:
    x_star =  float(x_star[0])

  # ---- Example: your f(x), but vector IC ----
  print("Computed fixed point (last x_next) = ", x_star)
  print("Last n_next value from vector x = ", get_MC_soln(file=FILE, t=TIME))
  print("Iterations:", k)
  
  
  # Anderson Alpha related stuff (weights for each iterate)
  alphaVec = np.zeros(len(x_star))
  if andersonAlphas is not None:
    print(f"Alpha sums are == {sum(andersonAlphas)}")
    print("Alphas =", andersonAlphas)
    all_fxs = get_all_fx(t=TIME, file=FILE)
    for ai, alpha in enumerate(andersonAlphas):
      alphaVec += alpha * all_fxs[ai]
    


  from Colors import Colors as col
  from Colors import nice_grid, nice_legend
  plt.figure(figsize=(5,3), dpi=dpi)
  plt.plot(x_star, 'ks--', label='validated', markersize=7, markerfacecolor='none', mew=0.5) # the last soln from the validator (x_next final)
  plt.plot(get_MC_soln(file=FILE, t=TIME), 'rx-', label=r'MC $x_\mathrm{next}$', markersize=5, lw=0.5, mew=0.5) # the last x_next from MC
  plt.plot(get_second_last_xnext(file=FILE, t=TIME), 'g+--', label=r'MC $x_\mathrm{next-1}$', markersize=5, lw=0.5, mew=0.5)
  plt.plot(get_last_fx(file=FILE, t=TIME), 'b^-', label=r'MC $f(x_\mathrm{next})$', markersize=5, lw=0.5,markerfacecolor='none', mew=0.5)
  plt.plot(alphaVec / np.sum(alphaVec) * np.sum(get_last_fx(file=FILE, t=TIME)), 
           'rd--', label=r'$\mathrm{SUM(} \alpha f(x)  \mathrm{)}$', markersize=3, lw=0.5, markerfacecolor='none', markeredgecolor='k', markeredgewidth=0.5)
  nice_legend()
  nice_grid()
  plt.title('Validated result vs. MC result', fontsize=9)


  plt.figure(figsize=(5,3), dpi=dpi)
  plt.plot( 
    np.array(x_star)/np.array(get_MC_soln(file=FILE, t=TIME)) - 1.0,
    'ks--', markerfacecolor='none', markersize=5, label='rel diff this script vs MC script'
    )
  
  theAlphaComparison = alphaVec / np.sum(alphaVec) * np.sum(get_last_fx(file=FILE, t=TIME))
  plt.plot( 
    theAlphaComparison/np.array(get_MC_soln(file=FILE, t=TIME)) - 1.0,
    'rs--', markerfacecolor='none', markersize=5, label=r'reconstructed $\alpha$ rel diff.'
    )
  
  nice_legend()
  nice_grid()
  plt.ylabel('rel diff.')
  plt.title('Validated result vs. MC result', fontsize=9)



