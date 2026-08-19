import numpy as np

class Cell:
  def __init__(self, dz: float, upperZ: float, lowerZ: float, upperType: str, lowerType: str, upperArea: float, lowerArea: float, cid: int):
    self.cid = cid # cell id
    self.upperZ = upperZ # upper face position
    self.lowerZ = lowerZ # lower face position
    self.dz = self.upperZ - self.lowerZ # size of cell aka the cell delta value
    self.centroid = (self.upperZ + self.lowerZ) / 2.0
    self.upperType = upperType # type of boundary - f for internal face and b for boundary
    self.lowerType = lowerType
    self.upperArea = upperArea # upper face area
    self.lowerArea = lowerArea # lower face area
    self.dFaceUpper = self.upperZ - self.centroid # absolute distance to upper FACE
    self.dFaceLower = self.centroid - self.lowerZ # absolute distance to lower FACE

    self.vol = self.dz * (self.lowerArea + self.upperArea)/2.0

    self.dCellUpper = None # absolute distance to lower or upper face
    self.dCellLower = None

    self.lowerNeighborCid = None # stays at None if no lower neighbor - otherwise is lower neighbor cell id
    self.upperNeighborCid = None


    # d_CF vector --- = scalar value in 1D
    self.lowerNeighborDist = None # stays at None if no dist - otherwise is lower neighbor distance to centroid
    self.upperNeightborDist = None

    # geoemtric diffusion coefficients
    self.upperNeighborGdiff = None
    self.lowerNeighborGdiff = None

    # geometric weighting factors
    self.geoUpper = None # geometric weighting factors --- equation 9.6
    self.geoLower = None

    # e_CF unit vector --- 1.0 in 1D
    self.ecfUpper = 1.0 # always 1 for 1D - defined in equation 9.34 --- e.g. just a unit vector
    self.ecfLower = 1.0

class Mesh_1D:
  """
  Creates a one dimensional mesh object.
  """
  def __init__(self, nodeCoords: np.ndarray, faceAreas: list, doStaggered=True):

    # proofreading / exceptions
    self.nodeCoords = np.unique(np.sort(nodeCoords)) # nodeCoords = locations of zFaces starting at boundary
    if not np.array_equal(self.nodeCoords, nodeCoords):
      raise Exception("Node coords are not sorted or has repeating values in it!")
    if len(faceAreas) != len(self.nodeCoords):
      raise Exception("Face areas are not the same length as node coords input!")
    self.faceAreas = faceAreas
    self.cells = {} # dict of cells  - key is cell id and value is the cell object
    self.nz = len(self.nodeCoords)-1 # number
    self.cidList = []
    self.centroids = []

    # now assign cells and stuff
    cid = int(0)
    for idx, node in enumerate(self.nodeCoords):
      if idx == 0:
        continue
      else:
        upperType = 'f'
        lowerType = 'f'
        if idx == len(self.nodeCoords)-1: # last so upper face is a boundary
          upperType = 'b'
        if idx == 1: # lower face is a boundary
          lowerType = 'b'

        # assign cell and its data
        self.cells[cid] = Cell(cid=cid,
                               dz=self.nodeCoords[idx] - self.nodeCoords[idx-1],
                               upperZ=self.nodeCoords[idx],
                               lowerZ=self.nodeCoords[idx-1],
                               upperType=upperType,
                               lowerType=lowerType,
                               upperArea=self.faceAreas[idx],
                               lowerArea=self.faceAreas[idx-1])
        self.cidList.append(cid)


        cid += 1 # progress cid

    self.cidLast = cid - 1
    self.cidFirst = 0

    # now we iterate across all cells and define distance to neighbors and neighbor ids
    # ids and distances will be None if the neighbor does not exist
    for cid in self.cells.keys():
      this_cell = self.cells[cid]
      # lower cell try
      try:
        # lower neighbor cid
        lwr_n_cid = self.cells[this_cell.cid - 1].cid # lower neighbor cid
        this_cell.lowerNeighborCid = lwr_n_cid


        # absolute distance to neighbor - d_CF
        this_cell.lowerNeighborDist = abs(this_cell.centroid - self.cells[this_cell.cid - 1].centroid)

        # Geometric diffusion coefficientss
        this_cell.lowerNeighborGdiff = this_cell.lowerArea / this_cell.lowerNeighborDist

        # geometric weighting factor for lower
        this_cell.geoLower = abs((self.cells[lwr_n_cid].dz/2) / this_cell.lowerNeighborDist)

      except:
        pass

      # upper cell try
      try:
        # lower neighbor cid
        upr_n_cid = self.cells[this_cell.cid + 1].cid # lower neighbor cid
        this_cell.upperNeighborCid = upr_n_cid

        # absolute distance to neighbor - d_CF
        this_cell.upperNeighborDist = abs(this_cell.centroid - self.cells[this_cell.cid + 1].centroid)

        # Geometric diffusion coefficientss
        this_cell.upperNeighborGdiff = this_cell.upperArea/this_cell.upperNeighborDist

        # geometric weighting factor for lower
        this_cell.geoUpper = abs((self.cells[upr_n_cid].dz/2) / this_cell.upperNeighborDist)

      except:
        pass

    # finally make a z vector
    for cid in self.cells.keys():
      self.centroids.append(self.cells[cid].centroid)

    # setup for the staggered mesh variables.
    if doStaggered:
      self.setup_staggered_mesh()


  def setup_staggered_mesh(self):
    """
    Takes in a Mesh_1D and outputs a corresponding staggered mesh.
    Output mesh:
      Node coords should the be original mesh's faces
      faceAreas should be averaged between faces.

    Original Mesh:
    |   .   |   .   |   .   |   .   |    .    |

    New Mesh:
    |.  |   .   |   .   |   .   |   .    |    .|
    """

    # setup for node coords for the staggered mesh ( faces )
    dx0 = self.cells[self.cidFirst].dz / 2
    dxLast = self.cells[self.cidLast].dz / 2
    centroid0 = self.nodeCoords[0]
    centroidLast = self.nodeCoords[-1]
    face0 = centroid0 - dx0
    faceLast = centroidLast + dxLast

    firstPart = np.append([face0], self.centroids)
    secondPart = [faceLast]
    self.stagNodeCoords = np.append(firstPart, secondPart)

    # setup the face areas for the staggered mesh.
    stagFaceAreas = [self.faceAreas[0]]
    stagFaceAreas = np.append([stagFaceAreas[0]], np.array(self.faceAreas[0:-1])/2 + np.array(self.faceAreas[1::])/2)
    stagFaceAreas = np.append(stagFaceAreas, [stagFaceAreas[-1]])

    # make staggered mesh now:
    self.stagMesh = Mesh_1D(nodeCoords=self.stagNodeCoords, faceAreas=stagFaceAreas, doStaggered=False)

    # now make a dict in the original mesh where you put in CID and east/west face
    # and you get the CID in the staggered mesh associated with that face.
    self.stagCellEast = {} # pass in a CID on the main mesh and return the CID from the staggered mesh that overlaps east face
    self.stagCellWest = {}
    for cid in self.cells.keys():
      self.stagCellWest[cid] = cid
      self.stagCellEast[cid] = cid + 1
