import openmc
import matplotlib.pyplot as plt
import numpy as np
openmc.Materials.cross_sections = '/home/jonathon/openmc_xs/endfb-vii.1-hdf5/cross_sections.xml'

def mat_from_stdcmp(mat: openmc.Material, file: str):
  """fills a material from stdcmp"""
  with open(file, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
      splitted = line.split(' ')
      nuc = splitted[0]
      nuc = nuc.split('-')[0] + nuc.split('-')[1]
      nuc = nuc.capitalize()
      if nuc[-1] == 'm':
        continue
      if (nuc == 'C12') | (nuc == 'C13'):
        nuc = 'C0'
      val = float(splitted[3])
      mat.add_nuclide(nuclide=nuc, percent=val, percent_type='ao')


"""Materials"""
f101 = openmc.Material(material_id=101, name='f101', temperature=900.0)
f102 = openmc.Material(material_id=102, name='f102', temperature=900.0)
f103 = openmc.Material(material_id=103, name='f103', temperature=900.0)
f104 = openmc.Material(material_id=104, name='f104', temperature=900.0)
f105 = openmc.Material(material_id=105, name='f105', temperature=900.0)
f106 = openmc.Material(material_id=106, name='f106', temperature=900.0)
f107 = openmc.Material(material_id=107, name='f107', temperature=900.0)
f108 = openmc.Material(material_id=108, name='f108', temperature=900.0)
f109 = openmc.Material(material_id=109, name='f109', temperature=900.0)
f110 = openmc.Material(material_id=110, name='f110', temperature=900.0)
f111 = openmc.Material(material_id=111, name='f111', temperature=900.0)
f112 = openmc.Material(material_id=112, name='f112', temperature=900.0)
f113 = openmc.Material(material_id=113, name='f113', temperature=900.0)
f114 = openmc.Material(material_id=114, name='f114', temperature=900.0)
f115 = openmc.Material(material_id=115, name='f115', temperature=900.0)
f116 = openmc.Material(material_id=116, name='f116', temperature=900.0)
mats_list = [f101,f102,f103,f104,f105,f106,f107,f108,f109,f110,f111,f112,f113,f114,f115,f116]
for mat in mats_list:
  mat_from_stdcmp(mat=mat, file=f"i3_p2_d2_triton_run_step_20_iter_0/StdCmpMix{mat.id}_")
  mat.set_density('sum')
  mat.add_s_alpha_beta(name='c_Graphite', fraction=1.0)

matPlena = openmc.Material(material_id=201, name='matPlena', temperature=900.0)
matPlena.add_nuclide('C0', percent=0.04512582, percent_type='ao')
matPlena.add_nuclide('Li6', percent=0.000000691507, percent_type='ao')
matPlena.add_nuclide('Li7', percent=0.0118566, percent_type='ao')
matPlena.add_nuclide('Be9', percent=0.00592865, percent_type='ao')
matPlena.add_nuclide('F19', percent=0.02371455, percent_type='ao')
matPlena.set_density('sum')

matFlibe = openmc.Material(material_id=301, name='matFlibe', temperature=900.0)
matFlibe.add_nuclide('Li6', percent=0.000001383014, percent_type='ao')
matFlibe.add_nuclide('Li7', percent=0.0237132, percent_type='ao')
matFlibe.add_nuclide('Be9', percent=0.0118573, percent_type='ao')
matFlibe.add_nuclide('F19', percent=0.0474291, percent_type='ao')
matFlibe.set_density('sum')

mats = openmc.Materials(materials= mats_list + [matPlena,matFlibe])
mats.export_to_xml()


"""Geometry"""
h = 550.0/16
# TODO currently we just have fuel hexagons w/ no flibe need to compute side lengths of hexagons and then properly use box2 below as well.
edge_1 = 20
edge_2 = 21
uList = []

fuelPlanes = [openmc.ZPlane(z0=0.0)]
theH = 0.0
for this in range(16):
  fuelPlanes.append(openmc.ZPlane(z0=theH+34.375))
  theH += 34.375

box1 = openmc.model.HexagonalPrism(edge_length=edge_1, boundary_type='reflective')
box2 = openmc.model.HexagonalPrism(edge_length=edge_2, boundary_type='reflective')

plUpperPlena = openmc.ZPlane(z0=575.0)
plLowerPlena = openmc.ZPlane(z0=-25.0)

plBottom = openmc.ZPlane(z0=-75.0)
plUpper = openmc.ZPlane(z0=625.0)

plUpper.boundary_type = 'vacuum'
plBottom.boundary_type = 'vacuum'

flibeLower = openmc.Cell(cell_id=99, fill=matPlena, region=(-box1 & +plBottom & -plLowerPlena))
unfueledLower = openmc.Cell(cell_id=100, fill=matPlena, region=(-box1 & +plLowerPlena & -fuelPlanes[0]))
c101 = openmc.Cell(cell_id=101, fill=f101, region=(-box1 & +fuelPlanes[0] &  -fuelPlanes[1]))
c102 = openmc.Cell(cell_id=102, fill=f102, region=(-box1 & +fuelPlanes[1] &  -fuelPlanes[2]))
c103 = openmc.Cell(cell_id=103, fill=f103, region=(-box1 & +fuelPlanes[2] &  -fuelPlanes[3]))
c104 = openmc.Cell(cell_id=104, fill=f104, region=(-box1 & +fuelPlanes[3] &  -fuelPlanes[4]))
c105 = openmc.Cell(cell_id=105, fill=f105, region=(-box1 & +fuelPlanes[4] &  -fuelPlanes[5]))
c106 = openmc.Cell(cell_id=106, fill=f106, region=(-box1 & +fuelPlanes[5] &  -fuelPlanes[6]))
c107 = openmc.Cell(cell_id=107, fill=f107, region=(-box1 & +fuelPlanes[6] &  -fuelPlanes[7]))
c108 = openmc.Cell(cell_id=108, fill=f108, region=(-box1 & +fuelPlanes[7] &  -fuelPlanes[8]))
c109 = openmc.Cell(cell_id=109, fill=f109, region=(-box1 & +fuelPlanes[8] &  -fuelPlanes[9]))
c110 = openmc.Cell(cell_id=110, fill=f110, region=(-box1 & +fuelPlanes[9] & -fuelPlanes[10]))
c111 = openmc.Cell(cell_id=111, fill=f111, region=(-box1 & +fuelPlanes[10] & -fuelPlanes[11]))
c112 = openmc.Cell(cell_id=112, fill=f112, region=(-box1 & +fuelPlanes[11] & -fuelPlanes[12]))
c113 = openmc.Cell(cell_id=113, fill=f113, region=(-box1 & +fuelPlanes[12] & -fuelPlanes[13]))
c114 = openmc.Cell(cell_id=114, fill=f114, region=(-box1 & +fuelPlanes[13] & -fuelPlanes[14]))
c115 = openmc.Cell(cell_id=115, fill=f115, region=(-box1 & +fuelPlanes[14] & -fuelPlanes[15]))
c116 = openmc.Cell(cell_id=116, fill=f116, region=(-box1 & +fuelPlanes[15] & -fuelPlanes[16]))
unfueledUpper = openmc.Cell(cell_id=117, fill=matPlena, region=(-box1 & -plUpperPlena & +fuelPlanes[-1]))
flibeUpper = openmc.Cell(cell_id=118, fill=matPlena, region=(-box1 & -plUpper & +plUpperPlena))

fuelCells = [c101,c102,c103,c104,c105,c106,c107,c108,c109,c110,c111,c112,c113,c114,c115,c116]
uCombined = openmc.Universe(universe_id=1, cells=fuelCells + [flibeLower, unfueledLower, flibeUpper, unfueledUpper])
uCombined.plot(basis='yz', width=[2.5,800.0], origin=(0.0, 0.0, 300) )

geom = openmc.Geometry()
geom.root_universe = uCombined
geom.export_to_xml()

"""Starting source and settings"""
# Make a point source at the center of the problem
# point = openmc.stats.Point((0.0,0.0,0.0))
spatial_dist = openmc.stats.Box((-5,-5,0.0), (5,5,550), only_fissionable=True)

# Define the starting source
source = openmc.IndependentSource(space=spatial_dist)
settings = openmc.Settings()
settings.source = source
settings.batches = 475
settings.inactive = 275
settings.particles = 300000
settings.temperature['method'] = 'interpolation'
settings.export_to_xml()

"""Tallies"""
talls = []
for this in fuelCells:
  the_t = openmc.Tally(name=f'flux{this.id}', tally_id=this.id)
  the_t.scores = ['heating-local']
  the_t.filters = [
    openmc.CellFilter(bins=[this,])
  ]
  talls.append(the_t)

tallies = openmc.Tallies(tallies=talls)
tallies.export_to_xml()

# Do the following to init() run() and finalize() results.
res = {}
import copy
import openmc.lib
openmc.lib.init()
openmc.lib.simulation_init()
for b in range(settings.batches):
  tallies = [openmc.lib.tallies[this.id] for this in talls]
  openmc.lib.next_batch()
  results = [tally.results for tally in tallies]
  res[b] = copy.deepcopy(results)
openmc.lib.simulation_finalize()
openmc.lib.finalize()

"""
Open and save pickle file
Save generation-wise-tallies.
"""
def tally_by_gen(res):
  shape0 = np.zeros(16)
  d = {}
  """nice little function to get tallies by gen"""
  for key in res.keys():
    shape1 = np.array([ this[:,:,1][0][0] for this in res[key][0:16] ])
    shape = shape1 - shape0
    d[key] = shape
    # advance
    shape0 = shape1
  return d

d = tally_by_gen(res)
for key in range(10,15):
  plt.plot(d[key])

import pickle as pkl
with open("data_step0.pkl", "wb") as file:
    pkl.dump(d, file)
