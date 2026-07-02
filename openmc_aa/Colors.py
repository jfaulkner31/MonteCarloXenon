import matplotlib.pyplot as plt

"""
Colors class
"""
class Colors():
  """
  Stores colors to be used in plotting and whatnot.
  """
  def __init__(self):
    pass

  @staticmethod
  def colors():
    colors = ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#8C564B', "#000000", '#E377C2', '#17BECF']
    return colors

  @staticmethod
  def colors2():
    colors = ["#478DE9", "#FF8D1A", "#41DD22", "#FF1E1A", "#A565D6", "#8B2E1B", "#000000", "#DD1EA4", "#008E9E"]
    return colors

  @staticmethod
  def manilla():
    return "#FFFDD7"


"""
Convenience
"""
def nice_grid():
  plt.grid(lw=1.7, alpha=0.3)

def frameless_legend(loc: str = 'upper left', **kwargs):
  plt.legend(frameon=False, **kwargs)

def nice_legend(**kwargs):
  plt.legend(edgecolor='black', facecolor=Colors.manilla(), **kwargs)
