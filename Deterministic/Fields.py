"""
General field class
"""
import numpy as np
from Meshing import Mesh_1D
import copy

# General field class
class Field:
  def __init__(self, name: str, initial_value: np.ndarray | float, mesh: Mesh_1D):
    self.name = name
    self.mesh = mesh
    self.centroids = self.mesh.centroids

class ScalarField(Field):
  """
  Cell centered field at the element centroid - represented by T
  """
  def __init__(self, name: str, initial_value: np.ndarray | float, mesh: Mesh_1D):
    #################################################
    # super from Field class
    super().__init__(initial_value=initial_value, name=name, mesh=mesh,)
    # Now to make initial values for this field based on if array or float
    if isinstance(initial_value, np.ndarray):
      if len(initial_value) == self.mesh.nz:
        self.initial_value = initial_value
      else:
        raise Exception("nz on variable input mesh must be the same as the length of the initial condition vector!")
    elif isinstance(initial_value, float):
      self.initial_value = np.ones(self.mesh.nz)*initial_value
    elif isinstance(initial_value, int):
      self.initial_value = np.ones(self.mesh.nz)*float(initial_value)
    else:
      raise Exception("Initial value must be of np.ndarray or float.")
    #################################################

    # Assigns T as initial values based on deepcopy
    self.T = copy.deepcopy(self.initial_value)
    self.T_old = copy.deepcopy(self.initial_value)

    # Gradient stuff:

    # interpolate_to_faces() -> uses Equation 9.5 to compute field values at faces instead of ecntroid
    #   g_C * T_C + (1-g_C)*T_F = T_f
    #   TODO make sure BC returns correct get_face_value

    # get_only_face_gradient -> uses 9.33 in 1D to compute face gradient
    #   TODO - make sure BC returns correct gradient method

    #   get_gradient_at_centroid() -> Computes gradient at centroid based on face values using gauss green
    #   1. get face values
    #   2. do gauss green gradient to get grad_C

    # get_gradient_at_faces() -> Computes gradient at faces.

  def set_T(self, vals: np.ndarray | list):
    for idx, val in enumerate(vals):
      self.T[idx] = val

  def interpolate_to_faces(self, upper_bc, lower_bc):
    """
    - interpolates field from centroid to faces using Equation 9.5:
      phi_f = g_C phi_C + (1-g_C)phi_F
    - This is second order accurate as long as the centroid lies on the same vector as the face centroid
    - In 1D this is always second order accurate.
    - for face values we use the values from the associated boundary condition
    """
    vals = np.array([])
    for idx, cid in enumerate(self.mesh.cidList):
      this_T_value = self.T[idx]
      try:
        upper_neighbor_T_value = self.T[idx+1]
        g = self.mesh.cells[cid].geoUpper
        upperFaceValue = g * this_T_value + (1-g) * upper_neighbor_T_value
        vals = np.append(vals,upperFaceValue)
      except:
        ### if this hits then it means there is no upper neighbor - e.g. last cell so the upper face
        ### value is now an outer cell - so we do nothing.
        pass

    # now handle bc's - these must have a method called get_face_value.
    lowest_face_value = lower_bc.get_face_value()
    highest_face_value = upper_bc.get_face_value()
    vals = np.append(lowest_face_value, vals)
    vals = np.append(vals, highest_face_value)
    return vals ### returns field interpolated to the faces using interpolation + bc's

  def get_gradient_at_centroid(self, upper_bc, lower_bc):
    face_values = self.interpolate_to_faces(upper_bc=upper_bc, lower_bc=lower_bc)
    grad_C_vec = np.array([])
    for idx, cid in enumerate(self.mesh.cidList):
      this_cell = self.mesh.cells[cid]
      lower_area = this_cell.lowerArea
      upper_area = this_cell.upperArea
      vol = this_cell.dz * (lower_area + upper_area) / 2.0
      upper_phi_f = face_values[idx+1]
      lower_phi_f = face_values[idx]
      grad_C = ( lower_phi_f * lower_area + upper_area * upper_phi_f ) / vol
      grad_C_vec = np.append(grad_C_vec, grad_C)
    return grad_C_vec # returns gradient at every single cell centroid

  def get_gradient_at_faces(self, upper_bc, lower_bc):
    """
      First we get barred gradient from 9.34 for inner faces.
      Then we get non-barred gradient using equation 9.33.
      For upper and lower boundary we can get the gradient by passing in the BC.
    """
    grad_C_vec = self.get_gradient_at_centroid(upper_bc=upper_bc, lower_bc=lower_bc) # gets gradient at centroids of cells for this variable
    grad_f_bar_vec = np.array([]) # barred gradient
    grad_f_vec = np.array([]) # real gradient
    # Get grad_f_bar at all the INNER faces.
    # then get grad_f at all the INNER faces.
    for idx, cid in enumerate(self.mesh.cidList):
      grad_C = grad_C_vec[idx]
      try:
        this_cell = self.mesh.cells[cid]
        upper_neighbor_cell = self.mesh.cells[cid+1]
        gC = this_cell.geoUpper
        gF = upper_neighbor_cell.geoLower
        grad_F = grad_C_vec[idx+1]

        # grad_f_bar
        grad_f_bar = gC * grad_C + gF*grad_F
        grad_f_bar_vec = np.append(grad_f_bar_vec, grad_f_bar)

        # grad_f
        phi_C = self.T[idx]
        phi_F = self.T[idx]
        dCF = this_cell.upperNeighborDist
        grad_f = (phi_F - phi_C) / dCF # equation 9.33 ---> uses e = unit vector = 1.0
        grad_f_vec = np.append(grad_f_vec, grad_f)

      except:
        # last cell implies no upper neighbor
        pass

    # now get lower and upper face gradient based on boundary conditions.
    upper_real_face_gradient = upper_bc.get_face_gradient()
    lower_real_face_gradient = lower_bc.get_face_gradient()
    grad_f_vec = np.append(lower_real_face_gradient, grad_f_vec)
    grad_f_vec = np.append(grad_f_vec, upper_real_face_gradient)
    return grad_f_vec ## returns gradient at ALL faces in the problem

  def get_only_face_gradient(self, upper_bc, lower_bc):
    """
      Returns gradient at faces for the 1D problem using equation 9.33.
      For upper and lower boundary we can get the gradient by passing in the BC.
    """
    grad_f_vec = np.array([]) # real gradient
    # Get grad_f_bar at all the INNER faces.
    # then get grad_f at all the INNER faces.
    for idx, cid in enumerate(self.mesh.cidList):
      try:
        this_cell = self.mesh.cells[cid]
        # grad_f
        phi_C = self.T[idx]
        phi_F = self.T[idx+1]
        dCF = this_cell.upperNeighborDist
        grad_f = (phi_F - phi_C) / dCF # equation 9.33 ---> uses e = unit vector = 1.0
        grad_f_vec = np.append(grad_f_vec, grad_f)

      except:
        # last cell implies no upper neighbor -- we get exception when idx+1 runss
        pass

    # now get lower and upper face gradient based on boundary conditions.
    upper_real_face_gradient = upper_bc.get_face_gradient()
    lower_real_face_gradient = lower_bc.get_face_gradient()
    grad_f_vec = np.append(lower_real_face_gradient, grad_f_vec)
    grad_f_vec = np.append(grad_f_vec, upper_real_face_gradient)

  def field_volume_integral(self):
    """
    Integrates field variable over volume: self.T * dV
    """
    integral = 0.0
    for cid in self.mesh.cidList:
      integral += self.T[cid] * self.mesh.cells[cid].vol
    return integral

class XS(Field):
  def __init__(self, name, initial_value, mesh):
    super().__init__(name, initial_value, mesh)
  