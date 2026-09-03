"""
Generates a symmetric pincell
and returns the model.
"""

import openmc
import numpy as np
import matplotlib.pyplot as plt
import openmc.deplete

from CustomSchemes.data.pwr_rei_template import UO2Material, GarbageMaterial, ZircMaterial, WaterMaterial, \
 LWRControlRod, LWRPincell, \
 get_density_curve, \
 PincellGeometry

def get_model(do_low_fidelity: bool = False) -> openmc.Model:
  DO_PLOT = False
  # TEMP
  temperature = 600

  # Geometry:
  cell_geom = PincellGeometry()
  fuel_r=cell_geom.fuel_r
  clad_ir=cell_geom.clad_ir
  clad_or=cell_geom.clad_or
  pitch=cell_geom.pitch
  dz=cell_geom.dz
  height = cell_geom.height


  DENSITY_X_VALUE = cell_geom.height/2.0

  densCurve = get_density_curve()
  bounds = np.linspace(0,cell_geom.height,cell_geom.nzones+1)
  xNew = bounds[0:-1]/2 + bounds[1:]/2
  yNew = np.interp(DENSITY_X_VALUE, densCurve[:,0], densCurve[:,1])
  yAll = np.interp(xNew, densCurve[:,0], densCurve[:,1])
  densValue = yNew
  if DO_PLOT:
    plt.figure(figsize=(5,3))
    plt.plot(xNew, yAll, 'ks--', markerfacecolor='white')
    plt.plot(xNew, yNew*np.ones(len(xNew)), 'rx--', markerfacecolor='white')
    plt.grid()
  print(xNew)



  fuel_mats = []
  pins = []

  # Generate materials.
  for n in range(cell_geom.nzones):
    fuel_mats.append(UO2Material(temp=temperature, name=f"fuel_{n}"))

  water = WaterMaterial(temp=temperature, name="water", density=densValue)
  zirc = ZircMaterial(temp=temperature, name='zirc')
  helium = GarbageMaterial(name='helium')

  fuel_mats_list = [this.get_mat() for this in fuel_mats]

  mats_all = openmc.Materials(materials=fuel_mats_list+[water.get_mat(), zirc.get_mat(), helium.get_mat()])

  # Generate pincells
  for idx, _ in enumerate(fuel_mats):
    pins.append(
      LWRPincell(name=f'cell_{idx}',
                              fuel_r=fuel_r, clad_ir=clad_ir, clad_or=clad_or, pitch=pitch, dz=dz,
                              Water=water, UO2=fuel_mats[idx], Zirc=zirc, Helium=helium)
    )

  # Pincell stack
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
  zPlu = openmc.ZPlane(z0=cell_geom.height)
  zNeg = openmc.ZPlane(z0=0.0)
  zPlu.boundary_type = 'reflective'
  zNeg.boundary_type = 'reflective'
  yPlu.boundary_type = 'reflective'
  yNeg.boundary_type = 'reflective'
  xPlu.boundary_type = 'reflective'
  xNeg.boundary_type = 'reflective'

  # Make lattice outer universe (ultra thin helium)
  the_outer_cell = openmc.Cell(fill=None, region=(-xPlu & +xNeg & -yPlu & +yNeg & -zPlu & +zNeg))
  lattice_outer = openmc.Universe(cells=[the_outer_cell])
  lat3d.outer = lattice_outer
    
  # Make prism cell
  prism = openmc.Cell(fill=lat3d, region=(-xPlu & +xNeg & -yPlu & +yNeg & -zPlu & +zNeg))
  final_universe = openmc.Universe(cells=[prism])

  # Export geometry to xml
  geom = openmc.Geometry()
  geom.root_universe = final_universe
  # geom.export_to_xml()

  # Plot the universe! Look at all those unique materials/cells!
  # Double check the thimbles are correctly laid out as well!
  if DO_PLOT:
    final_universe.plot(basis='xy', pixels=50000, origin=(0.0,0.0,cell_geom.height/2), color_by='material')
    final_universe.plot(basis='xz', pixels=50000, origin=(0.0,0.0,cell_geom.height/2), color_by='material')

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
  spatial_dist = openmc.stats.Box((-pitch/2,-pitch/2,0.0), (pitch/2,pitch/2,cell_geom.height), only_fissionable=True)

  # Define the starting source
  source = openmc.IndependentSource(space=spatial_dist)
  settings = openmc.Settings()
  settings.source = source
  settings.batches = 1000
  settings.inactive = 500
  if do_low_fidelity:
    settings.particles = 500
  else:
    settings.particles = 100000
  #settings.temperature['method'] = 'interpolation'
  # settings.export_to_xml()

  """Setup chain and depletion"""
  model = openmc.Model(geom, mats_all, settings, tallies)
  return model          
