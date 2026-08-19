#include "Mesh.h"

// Constructor
Mesh::Mesh(int nz, double L, double area) 
  : _nz(nz) 
  , _L(L)
  , _area(area)
{
  // Setup the mesh data
  double spacing = L / _nz;
  double L0 = 0.0;
  double Lmax = L;

  // Push back the nodes
  nodes.push_back(L0);
  for (int idx = 0; idx < _nz; idx++)
  {
    nodes.push_back(spacing + nodes.back());
  }
  
  // Push back the centroids
  for (int idx = 1; idx < nodes.size(); idx++)
  {
    centroids.push_back(nodes[idx] + nodes[idx-1]);
  }

}

// Destructor
Mesh::~Mesh() {}
