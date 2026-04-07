import numpy as np
import matplotlib.pyplot as plt
import copy
class Material:
  """
  A material with cross sections.
  """
  def __init__(self, s: float, a: float, f: float, nu: float):
    self.sigS = s
    self.sigA = a
    self.sigF = f
    self.sigNu = nu
    self.sigC = self.sigA - self.sigF
    self.sigT = s+a
    self.collV = np.array([self.sigS, self.sigC, self.sigF])/self.sigT
    self.cdf = np.cumsum(self.collV)
    if self.sigF > 0:
      self.fissile = True
    else:
      self.fissile = False
  def get_collision_type(self, xi):
    """0=scatter, 1=capture, 2=fission"""
    return int(np.searchsorted(self.cdf, xi, side='right'))

class ZPlane:
  def __init__(self, z0: float):
    self.z0 = z0

  def dist(self, z: float, wZ: float, **kwargs):
    return (self.z0 - z)/wZ

class Element:
  """
  A 1D element.
  """
  def __init__(self, material: Material, dz: float):
    self.material = material
    self.dz = dz
    self._eid = None
    self._flux = 0.0
    self._surfs: list[ZPlane] = []

  @property
  def fissile(self):
    return self.material.fissile

  @property
  def eid(self):
    return self._eid

  @property
  def flux(self):
    return self._flux

  @property
  def sigT(self):
    return self.material.sigT

  def inside(self, z: float) -> bool:
    return (z>=self._left) & (z<=self._right)

  def add_flux_score(self, w: float, l: float, nbank: float, score: bool):
    if not score:
      return
    else:
      self._flux += w*l/nbank/self.vol

  def divide_flux(self, nactive: int):
    """divides flux by number active generations"""
    self._flux /= nactive


  def set_eid(self, eid):
    self._eid = eid

  def set_surfs(self, left: float, right: float):
    self._right = right
    self._left = left
    area = 1.0
    self.vol = (right-left)*area
    self._surfs = [ZPlane(right), ZPlane(left)]

  def get_surf_crossing(self, wZ: float, z: float) -> tuple[float, float]:
    """
    Returns distance to surface and surface z value
    """
    sur = None
    for this in self._surfs:
      s = this.dist(z=z, wZ=wZ)

      # Discard behind us
      if s < 0:
        continue

      # Take first one
      if sur is None:
        sur = this
        dMin = s

      # Compare and take min
      elif s < dMin:
        sur = this
        dMin = s
    return dMin, sur.z0

class Mesh1D:
  """
  A mesh consisting of elements arranged in an adjacent fashion.
  """
  def __init__(self, elements: list[Element], left: str, right: str):
    self.elements: list[Element] = elements
    for idx, this in enumerate(elements):
      this.set_eid(eid=idx)
    self.left = left
    self.right = right
    self.dzList = [this.dz for this in self.elements]
    self.L = sum(self.dzList)

    self.boundaryList = [0.0]
    self.leftList = [] # rightside boundaries values
    self.rightList = [] # leftside boundaries values
    # List of  boundary values
    for this in self.dzList:
      self.boundaryList.append(self.boundaryList[-1] + this)
    self.boundaryList = np.array(self.boundaryList)
    self.leftList = np.array(self.boundaryList[0:-1])
    self.rightList = np.array(self.boundaryList[1:])
    for idx, this in enumerate(elements):
      this.set_surfs(left=self.leftList[idx], right=self.rightList[idx])

    self.zMin = self.boundaryList[0]
    self.zMax = self.boundaryList[-1]

    # Fission matrix
    nele = len(self.elements)
    self._fma = np.zeros((nele,nele))

  def _where_am_i(self, z: float) -> Element:
    """
    Put in z and return where (element id) what element we are inside of
    """
    if z > self.zMax:
      if self.right == 'v': # leak
        return -1 #
      if self.right == 'r': # refl
        return -2
    if z < self.zMin:
      if self.left == 'v': # leak
        return -1 #
      if self.right == 'r': # refl
        return -2

    for idx, this in enumerate(self.boundaryList[0:-1]):
      if (z > self.boundaryList[idx]) & (z <= self.boundaryList[idx+1]):
        return self.elements[idx]
    raise Exception("Particle lost!")
  def plot_flux(self):
    plt.plot([this.flux for this in self.elements])

  def add_FMA_score(self, start: int, end: int, score: float, scoreBool: bool):
    """
    Add a score to the fission matrix
    """
    if not scoreBool:
      return

    self._fma[start, end] += score

  def divide_fma(self, nactive: int):
    """divides fma by number active generations"""
    self._fma /= nactive

class Neutron:
  def __init__(self, element_start: Element, z0: float, w0 = 1.0):
    self.e0 = element_start
    self.w0 = 1.0
    self.z0 = z0
    self.z1: float = None
    self.e1: Element = None
    self.w1: float = None
  def _set_final(self, z1: float, e1: Element, w1: float):
    self.e1 = e1
    self.w1 = w1
    self.z1 = z1

class Bank:
  """A bank of particles"""
  def __init__(self, bank: list[Neutron]):
    self.bank = bank
    self.nbank = sum([this.w0 for this in self.bank]) # starting score - N - number particles
    self.nbank_e = {}
    for this in self.bank:
      try:
        self.nbank_e[this.e0.eid] += this.w0
      except:
        self.nbank_e[this.e0.eid] = this.w0
  def get_bank_z0(self):
    return [this.z0 for this in self.bank]

class FluxTally:
  """
  Flux tally implementation

  Currently just flux in a material.

  Has important methods for scoring between generations.
  """
  def __init__(self, tid: int):
    self.tid = tid
    self.score = 0.0

class MonteCarlo:
  def __init__(self, mesh: Mesh1D, npg: int, nsk: int, ngen: int):
    self.mesh = mesh
    self.npg = npg
    self.nsk = nsk
    self.ngen = ngen

  def _init_bank(self):
    """
    Initializer for the fission bank. Based on randomly sampling fissionable material
    """
    # Compute random number boundaries for each channel.
    bank = np.empty(self.npg, dtype=Neutron)
    counter = 0

    while counter < self.npg:
      # if ((counter % 1000) == 0):
      #   print("Now banking n =",counter)
      r = np.random.random()
      z = r * self.mesh.L
      ele = self.mesh._where_am_i(z=z)
      if ele.fissile:
        neu = Neutron(element_start=ele, z0 = z, w0=1.0)
        bank[counter] = neu
        counter += 1

    return Bank(bank)

  def _get_w_iso(self)->tuple[float,float,float]:
    """
    Sample a directional flight path/direction
    """
    # Sample directional flights
    xidir = np.random.random(2) # xidir !!! not xipos
    wtheta = np.arccos(1-2*xidir[0])  # teta
    wphi = 2*np.pi*xidir[1]         # phi angle

    # WX, WY, WZ
    wx = np.sin(wtheta) * np.cos(wphi)
    wy = np.sin(wtheta) * np.sin(wphi)
    wz = np.cos(wtheta)
    return wx, wy, wz

  def _solveST(self, bank: Bank, score = True):
    """
    Monte Carlo surface tracking solver.

    Implemented russian roulette and implicit fission/capture.
    """
    next_bank: list[Neutron] = []
    kScore = 0.0

    # Go through bank (npg)
    for neuIdx, n in enumerate(bank.bank):
      # if (neuIdx % 1000) == 0:
      #   print("Now solving n =",neuIdx)
      # Neutron starting information
      w = n.w0 # current weight
      z = n.z0 # current start pos
      x = 0.0
      y = 0.0
      e = n.e0 # current element

      # Sample omegas
      wx,wy,wz = self._get_w_iso()

      # Tracking
      while 1:
        xi = np.random.random() # sample a random number for distance determination
        Si = -1.0 * np.log(xi) / e.sigT #

        # Get x1,y1,z1
        x1=x + Si*wx
        y1=y + Si*wy
        z1=z + Si*wz

        # If it crossed surface, compute L and score:
        if not e.inside(z1):
          eOriginal = e
          l, z = e.get_surf_crossing(wZ=wz, z=z)
          z += 1e-10*np.sign(wz) # nudge in direction wz if it crossed the surface
          e.add_flux_score(w=w, l=l, nbank=bank.nbank, score=score)
          e = self.mesh._where_am_i(z=z)
          if e == -1: # Leakage condition
            break
          elif e == -2: # reflective
            z -= 2e-10*np.sign(wz) # nudge in opposite direction :)
            wz *= -1.0 # change direction
            e = eOriginal


        else: # collide within current volume
          l = (x1-x)**2 + (y1-y)**2 + (z1-z)**2
          e.add_flux_score(w=w, l=l, nbank=bank.nbank, score=score)

          # 1. Implicit Fission
          if e.material.fissile:
            prod_w = w * e.material.sigNu * (e.material.sigF / e.material.sigT)
            kScore += prod_w
            next_bank.append(Neutron(z0=z1, element_start=e, w0=prod_w))
            self.mesh.add_FMA_score(start=n.e0.eid, end=e.eid, score=prod_w / bank.nbank_e[n.e0.eid], scoreBool=score)

          # 2. Implicit Capture (Survival Biasing)
          w = w * (e.material.sigS / e.material.sigT)

          # 3. Russian Roulette
          keepGoing, wNew = self._oh_those_russians(w0=n.w0, currWeight=w, cutoff=0.5)
          if not keepGoing:
            break
          else:
            w = wNew
            # 4. Scattering (Surviving particles always scatter)
            wx,wy,wz = self._get_w_iso()
            z = z1

    # Compute eigenvalue: new / starting
    kestimate = kScore/bank.nbank
    # print("kgen =", kestimate)

    # Resize fission bank
    resized_bank = self._resample_to_const_N_equal_weight(next_bank=next_bank, NPG=self.npg)
    resized_bank = Bank(bank=resized_bank)
    return resized_bank, kestimate

  def _oh_those_russians(self, w0: float, currWeight: float, cutoff=0.25) ->  tuple[bool, float]:
    """
    Plays russian roulette.

    Parameters
    ==========

    w0 : float
      neutron starting weight
    currWeight : float
      neutron current weight
    cutoff : float
      russian roulette cutoff

    Returns
    =======
    tuple[keepGoing, newWeight]
    """
    if currWeight < cutoff*w0:
      # Play russian roulette
      p = np.random.rand()
      if p < currWeight/w0: # survived RR
        return True, w0
      else:
        return False, 0.0
    else:
      return True, currWeight

  def _resample_to_const_N_equal_weight(self, next_bank: list[Neutron], NPG: int, rng=np.random):
      """
      Put in a fission bank and get out a new fission bank (list of neutrons).

      Const N based on NPG with equal (1.0) starting weight values.
      """
      if NPG <= 0:
          return []
      if len(next_bank) == 0:
          raise ValueError("No fission sites to resample from.")

      w = np.array([n.w0 for n in next_bank], dtype=float)
      W = w.sum()
      if not np.isfinite(W) or W <= 0.0:
          raise ValueError("Nonpositive/invalid total fission weight; cannot resample.")

      cdf = np.cumsum(w / W)

      # systematic points in (0,1)
      u0 = rng.random() / NPG
      us = u0 + np.arange(NPG) / NPG

      # indices of selected sites
      idx = np.searchsorted(cdf, us, side="left")

      # build new generation source, all weight 1
      new_source = []
      for i in idx:
          parent = next_bank[int(i)]
          nn = copy.copy(parent)   # copy position/starting element, etc.
          nn.w0 = 1.0
          # reset any “final state” from previous tracking if needed:
          nn.z1 = None; nn.e1 = None; nn.w1 = None
          new_source.append(nn)

      return new_source

  def solve(self):
    # Generation number is the key, kGen is the value
    kResults: dict[int, float] = {}
    fluxResults: dict[int, dict[int, float]] = {} # gen -> element -> generation-wise-element-wise flux

    bank = self._init_bank()

    totGen = 0
    nactive = self.ngen - self.nsk
    print("\nNow solving inactive generations:")
    print("======================================")
    # Skipped generations
    for gen in range(self.nsk):
      bank, kgen = self._solveST(bank=bank, score=False)
      # kResults[totGen] = kgen
      self._print_keff_line(skipped=True, kgen=kgen, gen=totGen)
      totGen += 1

    # Now active generations
    print("\nNow solving active generations:")
    print("======================================")
    for gen in range(nactive):
      bank, kgen = self._solveST(bank=bank, score=True)
      kResults[totGen] = kgen
      fluxResults[totGen] = {} # store fluxes of each
      for this in self.mesh.elements:
        fluxResults[totGen][this.eid] = this.flux / (gen+1)

      self._print_keff_line(skipped=False, kgen=kgen, gen=totGen)
      totGen += 1

    """Normalize scoring based on number active generations"""
    for this in self.mesh.elements:
      this.divide_flux(nactive)
    self.mesh.divide_fma(nactive)

    """print final line / variance"""
    m, var = self._compute_keff(k=kResults)
    self.kest = m
    self.std_dev = var**0.5
    self.kByGeneration = kResults
    self.genFluxResults = fluxResults
    print("======================================")
    print(f"Final keff est: {m:.6f} +/- {var**0.5:.6f}")

  def _print_keff_line(self, skipped: bool, kgen: float, gen: int):
    the_str = f"Solving keff at Generation {gen}: {kgen}"
    print(the_str)

  def _compute_keff(self, k: dict[int,float]):
    """
    Computes mean and variance of keff

    Variance is "variance of the mean" see lectures by forrest brown.
    rather than population variance.

    """
    vals = list(k.values())
    pop_var = np.var(vals) # population variance
    var_mean = pop_var/len(vals)
    m = np.mean(vals)
    return m, var_mean

  """
  Postprocessing
  """
  def flux_by_generation(self, elements: list[int]):
    """plots flux in each element by generation"""
    gens = list(self.genFluxResults.keys())
    for eid in elements:
      vals = [self.genFluxResults[key][eid] for key in gens]
      plt.plot(gens, vals, 'x-', markersize=3, label=f'eid {eid}')
  def keff_by_generation(self, dpi: int=300):
    """plots keff by active generation"""
    plt.figure(figsize=(5,3), dpi=dpi)
    gens = list(self.kByGeneration.keys())
    vals = list(self.kByGeneration.values())
    plt.plot(gens, vals, 'x-', markersize=3)
    plt.ylabel(r'$\mathrm{k_{eff}}$')
    plt.xlabel(r'$\mathrm{Generation}$')


