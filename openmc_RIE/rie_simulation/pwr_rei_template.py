"""
Generates a pincell and returns the model.
"""

import openmc
import numpy as np
import matplotlib.pyplot as plt
import openmc.deplete

class UO2Material():
  """
  UO2 openmc material
  """
  def __init__(self, temp: float, name: str):
    """Uo2 generated from scale 3.5 wt% u235 composition"""
    mat = openmc.Material(temperature=temp, name=name)
    mat.add_nuclide('U235', percent=8.147143e-04, percent_type='ao')
    mat.add_nuclide('U238', percent=2.217910e-02, percent_type='ao')
    mat.add_nuclide('O16', percent=4.587589e-02, percent_type='ao')
    mat.add_nuclide('O17', percent=1.747530e-05, percent_type='ao')
    # mat.add_nuclide('O18', percent=9.427465e-05, percent_type='ao')
    mat.set_density(units='sum')
    mat.volume = 366/16 * np.pi * 0.3975**2
    mat.depletable = True
    self.mat = mat
  def get_mat(self) -> openmc.Material:
    return self.mat

class GarbageMaterial():
  """material with very low xs - meant to be helium gap"""
  def __init__(self, name: str, temp: float = 600):
    mat = openmc.Material(temperature=temp, name=name)
    mat.add_nuclide('He4', percent=1, percent_type='ao')
    mat.set_density(units='g/cm3', density=1e-20)
    self.mat = mat
  def get_mat(self) -> openmc.Material:
    return self.mat

class ZircMaterial():
  """Zircaloy 4 openmc material"""
  def __init__(self, temp: float, name: str):
    mat = openmc.Material(temperature=temp, name=name)
    mat.add_nuclide('O16',   percent=1.19276E-03, percent_type='wo')
    mat.add_nuclide('Cr50',  percent=4.16117E-05, percent_type='wo')
    mat.add_nuclide('Cr52',  percent=8.34483E-04, percent_type='wo')
    mat.add_nuclide('Cr53',  percent=9.64457E-05, percent_type='wo')
    mat.add_nuclide('Cr54',  percent=2.44600E-05, percent_type='wo')
    mat.add_nuclide('Fe54',  percent=1.12572E-04, percent_type='wo')
    mat.add_nuclide('Fe56',  percent=1.83252E-03, percent_type='wo')
    mat.add_nuclide('Fe57',  percent=4.30778E-05, percent_type='wo')
    mat.add_nuclide('Fe58',  percent=5.83334E-06, percent_type='wo')
    mat.add_nuclide('Zr90',  percent=4.97862E-01, percent_type='wo')
    mat.add_nuclide('Zr91',  percent=1.09780E-01, percent_type='wo')
    mat.add_nuclide('Zr92',  percent=1.69646E-01, percent_type='wo')
    mat.add_nuclide('Zr94',  percent=1.75665E-01, percent_type='wo')
    mat.add_nuclide('Zr96',  percent=2.89038E-02, percent_type='wo')
    mat.add_nuclide('Sn112', percent=1.27604E-04, percent_type='wo')
    mat.add_nuclide('Sn114', percent=8.83732E-05, percent_type='wo')
    mat.add_nuclide('Sn115', percent=4.59255E-05, percent_type='wo')
    mat.add_nuclide('Sn116', percent=1.98105E-03, percent_type='wo')
    mat.add_nuclide('Sn117', percent=1.05543E-03, percent_type='wo')
    mat.add_nuclide('Sn118', percent=3.35688E-03, percent_type='wo')
    mat.add_nuclide('Sn119', percent=1.20069E-03, percent_type='wo')
    mat.add_nuclide('Sn120', percent=4.59220E-03, percent_type='wo')
    mat.add_nuclide('Sn122', percent=6.63497E-04, percent_type='wo')
    mat.add_nuclide('Sn124', percent=8.43355E-04, percent_type='wo')
    mat.set_density(units='g/cm3', density=6.56)
    self.mat = mat
  def get_mat(self) -> openmc.Material:
    return self.mat

class WaterMaterial():
  """water material"""
  def __init__(self, temp: float, name: str, density: float):
    mat = openmc.Material(temperature=temp, name=name)
    mat.add_nuclide('O16',   percent=3.330861e-01, percent_type='ao')
    mat.add_nuclide('H1',  percent=6.663259e-01, percent_type='ao')
    mat.add_nuclide('B10',  percent=7.186970e-05, percent_type='ao')
    mat.add_nuclide('B11',  percent=2.892846e-04, percent_type='ao')
    mat.add_s_alpha_beta('c_H_in_H2O', fraction=1.0)
    mat.set_density(units='g/cm3', density=density)
    self.mat = mat
  def get_mat(self) -> openmc.Material:
    return self.mat

class LWRPincell():
  """
  Universe for an LWR pincell
  """
  def __init__(self, name: str,
              fuel_r: float, clad_ir: float, clad_or: float, pitch: float, dz: float,
              Water: WaterMaterial, UO2: UO2Material, Zirc: ZircMaterial, Helium: GarbageMaterial):
    # Dimensions
    self.fuel_r = fuel_r
    self.clad_ir = clad_ir
    self.clad_or = clad_or
    self.pitch = pitch
    self.xPitch = pitch
    self.yPitch = pitch
    self.dz = dz

    centroid = (0.0,0.0,0.0)

    # Make planes.
    xPlu = openmc.XPlane(x0=centroid[0] + self.xPitch/2.0)
    xNeg = openmc.XPlane(x0=centroid[0] - self.xPitch/2.0)
    yPlu = openmc.YPlane(y0=centroid[1] + self.yPitch/2.0)
    yNeg = openmc.YPlane(y0=centroid[1] - self.yPitch/2.0)
    zPlu = openmc.ZPlane(z0=centroid[2] + self.dz/2.0)
    zNeg = openmc.ZPlane(z0=centroid[2] - self.dz/2.0)

    # Make cylinders
    fuel_cyl = openmc.ZCylinder(r=fuel_r)
    clad_inner_cyl = openmc.ZCylinder(r=clad_ir)
    clad_outer_cyl = openmc.ZCylinder(r=clad_or)

    # Fuel cell, cladding cell, nothingness cell, water cell
    fuel_cell = openmc.Cell(fill=UO2.get_mat(), region=(-fuel_cyl & -zPlu & +zNeg))
    helium_cell = openmc.Cell(fill=Helium.get_mat(),
                              region=(+fuel_cyl & -clad_inner_cyl & -zPlu & +zNeg))
    zirc_cell = openmc.Cell(fill=Zirc.get_mat(), region=(+clad_inner_cyl & -clad_outer_cyl & -zPlu & +zNeg))
    water_cell = openmc.Cell(fill=Water.get_mat(), region=(-xPlu & +xNeg & -yPlu & +yNeg & -zPlu & +zNeg & +clad_outer_cyl))

    # Universe
    self.uni = openmc.Universe(name=name, cells=[fuel_cell, helium_cell, zirc_cell, water_cell])

    # Store the plane objects in case we want to modify them later.
    self.xPlu, self.xNeg, self.yPlu, self.yNeg, self.zPlu, self.zNeg = xPlu, xNeg, yPlu, yNeg, zPlu, zNeg
  def get_uni(self) -> openmc.Universe:
    return self.uni
  def set_refl_z(self):
    self.zPlu.boundary_type='reflective'
    self.zNeg.boundary_type='reflective'

class LWRControlRod():
  """
  Universe for an LWR control rod cell aka
  empty instrumentation thimble
  """
  def __init__(self, name: str,
              clad_ir: float, clad_or: float, pitch: float, dz: float,
              Water: WaterMaterial, Zirc: ZircMaterial):
    # Dimensions
    self.clad_ir = clad_ir
    self.clad_or = clad_or
    self.pitch = pitch
    self.xPitch = pitch
    self.yPitch = pitch
    self.dz = dz

    centroid = (0.0,0.0,0.0)

    # Make planes.
    xPlu = openmc.XPlane(x0=centroid[0] + self.xPitch/2.0)
    xNeg = openmc.XPlane(x0=centroid[0] - self.xPitch/2.0)
    yPlu = openmc.YPlane(y0=centroid[1] + self.yPitch/2.0)
    yNeg = openmc.YPlane(y0=centroid[1] - self.yPitch/2.0)
    zPlu = openmc.ZPlane(z0=centroid[2] + self.dz/2.0)
    zNeg = openmc.ZPlane(z0=centroid[2] - self.dz/2.0)

    # Make cylinders
    clad_inner_cyl = openmc.ZCylinder(r=clad_ir)
    clad_outer_cyl = openmc.ZCylinder(r=clad_or)

    # Fuel cell, cladding cell, nothingness cell, water cell
    water_cell0 = openmc.Cell(fill=Water.get_mat(), region=(-clad_inner_cyl & -zPlu & +zNeg))
    zirc_cell = openmc.Cell(fill=Zirc.get_mat(), region=(+clad_inner_cyl & -clad_outer_cyl & -zPlu & +zNeg))
    water_cell = openmc.Cell(fill=Water.get_mat(), region=(-xPlu & +xNeg & -yPlu & +yNeg & -zPlu & +zNeg & +clad_outer_cyl))

    # Universe
    self.uni = openmc.Universe(name=name, cells=[zirc_cell, water_cell, water_cell0])

    # Store the plane objects in case we want to modify them later.
    self.xPlu, self.xNeg, self.yPlu, self.yNeg, self.zPlu, self.zNeg = xPlu, xNeg, yPlu, yNeg, zPlu, zNeg
  def get_uni(self) -> openmc.Universe:
    return self.uni
  def set_refl_z(self):
    self.zPlu.boundary_type='reflective'

    self.zNeg.boundary_type='reflective'


def get_model() -> openmc.Model:
  densCurve = np.array([[13.039305966937505, 0.7546030708094662],
  [17.971143135616785, 0.7541532649090789],
  [22.902980304296058, 0.7536839022304138],
  [27.834817472975335, 0.7532145395517489],
  [32.76665464165463, 0.7527451768730838],
  [37.6984918103339, 0.7520215760768086],
  [42.63032897901316, 0.7512979752805334],
  [47.56216614769243, 0.7504765905928696],
  [52.49400331637171, 0.7497725465748721],
  [57.42584048505098, 0.748912048330653],
  [62.357677653730285, 0.7479537661950453],
  [67.28951482240954, 0.7469368137246044],
  [72.22135199108881, 0.7459394180324412],
  [77.15318915976809, 0.7449420223402782],
  [82.08502632844736, 0.7438663995350042],
  [87.01686349712662, 0.7427125496166194],
  [91.94870066580592, 0.7415586996982346],
  [96.88053783448518, 0.7404048497798498],
  [101.81237500316448, 0.7392314430831872],
  [106.74421217184374, 0.7380189228299694],
  [111.676049340523, 0.7368064025767515],
  [116.60788650920227, 0.7355547687669781],
  [121.53972367788154, 0.7343226917354825],
  [126.47156084656082, 0.7330906147039868],
  [131.4033980152401, 0.7318585376724911],
  [136.3352351839194, 0.7305677903061624],
  [141.26707235259866, 0.729316156496389],
  [146.19890952127793, 0.7280645226866157],
  [151.13074668995722, 0.7268324456551201],
  [156.0625838586365, 0.7256003686236244],
  [160.99442102731578, 0.7243878483704065],
  [165.92625819599505, 0.7231557713389108],
  [170.8580953646743, 0.7219236943074152],
  [175.78993253335358, 0.7207502876107527],
  [180.72176970203284, 0.7196159944706456],
  [185.65360687071214, 0.7184425877739831],
  [190.5854440393914, 0.7172691810773206],
  [195.51728120807067, 0.7161740014937689],
  [200.44911837674996, 0.7150788219102172],
  [205.38095554542926, 0.7140031991049433],
  [210.3127927141085, 0.7129666898562247],
  [215.24462988278776, 0.7118910670509507],
  [220.17646705146706, 0.7108936713587876],
  [225.10830422014632, 0.7099549460014576],
  [230.0401413888256, 0.7089379935310167],
  [234.97197855750488, 0.707979711395409],
  [239.90381572618415, 0.7070214292598013],
  [244.83565289486341, 0.7061609310155822],
  [249.7674900635427, 0.7053199895496407],
  [254.69932723222198, 0.7044594913054215],
  [259.63116440090124, 0.7035794362829246],
  [264.56300156958054, 0.7028362787083717],
  [269.49483873825983, 0.7020540075772633],
  [274.42667590693907, 0.7013108500027104],
  [279.35851307561836, 0.7006068059847129],
  [284.29035024429766, 0.6999027619667154],
  [289.22218741297695, 0.6992769450618287],
  [294.1540245816562, 0.6987097984917752],
  [299.0858617503354, 0.6981230951434438],
  [304.0176989190147, 0.6975559485733904],
  [308.94953608769407, 0.6970083587816145],
  [313.8813732563733, 0.6965585528812273],
  [318.81321042505255, 0.6961478605373953],
  [323.74504759373184, 0.6957371681935635],
  [328.6768847624111, 0.6953069190714539],
  [333.6087219310904, 0.6949744538407329],
  [338.54055909976967, 0.6947202157231227],
  [343.4723962684489, 0.6944659776055124],
  [347.99324700640494, 0.6942469416888021],
  [349.637192729298, 0.6970865858947254],
  [349.637192729298, 0.6956784978587304],
  [349.637192729298, 0.6928623217867403]])
  bounds = np.linspace(0,366,17)
  xNew = bounds[0:-1]/2 + bounds[1:]/2
  yNew = np.interp(xNew, densCurve[:,0], densCurve[:,1])
  plt.figure(figsize=(5,3))
  plt.plot(xNew, yNew, 'ks--', markerfacecolor='white')
  plt.grid()
  print(xNew)
  densValues = yNew


  # Water densities.
  densities = densValues
  temps = [600]*16
  zirc = ZircMaterial(temp=600, name='zirc')
  helium = GarbageMaterial(name='helium')

  # Geometry:
  fuel_r=0.3975
  clad_ir=0.4125
  clad_or=0.4750
  pitch=1.26
  dz=366/16

  fuel_mats = []
  water_mats = []
  pins = []

  # Generate unique materials
  for n in range(16):
    fuel_mats.append(UO2Material(temp=600, name=f"fuel_{n}"))
    water_mats.append(WaterMaterial(temp=temps[n], name=f"water_{n}", density=densities[n]))
  fuel_mats_list = [this.get_mat() for this in fuel_mats]
  water_mats_list = [this.get_mat() for this in water_mats]
  mats_all = openmc.Materials(materials=fuel_mats_list + water_mats_list + [zirc.get_mat(), helium.get_mat()])
  # mats_all.export_to_xml()

  # Generate pincells
  for idx, fuel in enumerate(water_mats):
    pins.append(
      LWRPincell(name=f'cell_{idx}',
                              fuel_r=fuel_r, clad_ir=clad_ir, clad_or=clad_or, pitch=pitch, dz=dz,
                              Water=water_mats[idx], UO2=fuel_mats[idx], Zirc=zirc, Helium=helium)
    )

  # Pincell stack.
  lowest = 0.0
  planes = []
  planes.append(openmc.ZPlane(lowest))
  pinUnis = [this.get_uni() for this in pins]
  lat3d = openmc.RectLattice()
  lat3d.lower_left = (-pitch/2, -pitch/2, 0)
  lat3d.pitch = (pitch, pitch, dz)
  lat3d.universes = [
      [[pinUnis[0]]],
      [[pinUnis[1]]],
      [[pinUnis[2]]],
      [[pinUnis[3]]],
      [[pinUnis[4]]],
      [[pinUnis[5]]],
      [[pinUnis[6]]],
      [[pinUnis[7]]],
      [[pinUnis[8]]],
      [[pinUnis[9]]],
      [[pinUnis[10]]],
      [[pinUnis[11]]],
      [[pinUnis[12]]],
      [[pinUnis[13]]],
      [[pinUnis[14]]],
      [[pinUnis[15]]],
  ]

  # Make planes.
  xPlu = openmc.XPlane(x0=pitch/2.0)
  xNeg = openmc.XPlane(x0=-pitch/2.0)
  yPlu = openmc.YPlane(y0=pitch/2.0)
  yNeg = openmc.YPlane(y0=-pitch/2.0)
  zPlu = openmc.ZPlane(z0=366)
  zNeg = openmc.ZPlane(z0=0.0)
  zPlu.boundary_type = 'vacuum'
  zNeg.boundary_type = 'vacuum'
  yPlu.boundary_type = 'reflective'
  yNeg.boundary_type = 'reflective'
  xPlu.boundary_type = 'reflective'
  xNeg.boundary_type = 'reflective'

  # Make prism cell
  prism = openmc.Cell(fill=lat3d, region=(-xPlu & +xNeg & -yPlu & +yNeg & -zPlu & +zNeg))
  final_universe = openmc.Universe(cells=[prism])

  # Export geometry to xml
  geom = openmc.Geometry()
  geom.root_universe = final_universe
  # geom.export_to_xml()

  # Plot the universe! Look at all those unique materials/cells!
  # Double check the thimbles are correctly laid out as well!
  final_universe.plot(basis='xy', pixels=50000, origin=(0.0,0.0,366/2), color_by='material')

  """Tallies"""
  talls = []
  for this in fuel_mats_list:
    the_t = openmc.Tally(name=f'flux{this.id}', tally_id=this.id)
    the_t.scores = ['flux']
    the_t.filters = [
      openmc.MaterialFilter(bins=[this,])
    ]
    talls.append(the_t)

  tallies = openmc.Tallies(tallies=talls)
  # tallies.export_to_xml()


  """Starting source and settings"""
  # Make a point source at the center of the problem
  # point = openmc.stats.Point((0.0,0.0,0.0))
  spatial_dist = openmc.stats.Box((-pitch/2,-pitch/2,0.0), (pitch/2,pitch/2,375), only_fissionable=True)

  # Define the starting source
  source = openmc.IndependentSource(space=spatial_dist)
  settings = openmc.Settings()
  settings.source = source
  settings.batches = 1000
  settings.inactive = 500
  settings.particles = 300000
  #settings.temperature['method'] = 'interpolation'
  # settings.export_to_xml()

  """Setup chain and depletion"""
  model = openmc.Model(geom, mats_all, settings, tallies)
  return model

  """

    chain = 'chain_casl_pwr.xml'
    openmc.config['chain_file'] = chain

    op = openmc.deplete.CoupledOperator(model)

    #fluxes,micros = openmc.deplete.get_microxs_and_flux(model, fuel_mats_list)

    power_density=104
    volume_total = 366*np.pi * fuel_r**2
    power = power_density*volume_total
    timesteps = [0.5, 1.0, 1.5, 2., 5, 10] ### days 0->20


    openmc.deplete.PredictorIntegrator(op, timesteps, power, timestep_units='d').integrate()


    #import pickle as pkl
    #with open('micros.pkl', 'wb') as file:
    #  pkl.dump((fluxes,micros), file)
  """
