import numpy as np
def transport(N: int) -> np.ndarray[tuple[int], float]:
  """
  Runs transport. Returns a flat flux with 10% noise.
  
  Parameters
  ==========
  N : int
    the number of depletion zones.

  """
  return __noisy_transport(N=N, noise=0.10)

def flat_transport(N: int) -> np.ndarray[tuple[int], float]:
  """
  Runs transport. Returns a flat flux with zero noise.
  
  Parameters
  ==========
  N : int
    the number of depletion zones.
  
  Parameters
  ==========
  N : int
    the number of depletion zones.

  noise : float
    the noise level - 0.10 would be 10% noise from the flat 'true' solution
  """
  return __noisy_transport(N=N, noise=0)

def __noisy_transport(N: int, noise: float) -> np.ndarray[tuple[int], float]:
  """
  transport flat with a given noise level
  """
  true = 1/N
  randMaxmin = true*noise # the interval rand shall be normalized ot : e.g. [-0.01 0.01]
  rands = np.random.random(N)*2 - 1
  rands *= randMaxmin # -0.05 to 0.05
  flux = np.ones(N) / N + rands
  for this in flux:
    assert this > 0, "Flux in dummy transport kernel was found to be negative or zero - this should not be possible"
  flux = flux / np.sum(flux)
  return flux
