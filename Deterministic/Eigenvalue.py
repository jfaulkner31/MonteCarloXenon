"""
Object that holds a float keff value
"""
class Eigenvalue():
  def __init__(self, keff = 1.0):
    self._keff = keff
  def _get(self):
    return self.keff
  @property
  def keff(self):
    return self._get()
