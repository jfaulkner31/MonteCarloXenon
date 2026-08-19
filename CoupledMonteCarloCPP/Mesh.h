#include <vector>
class Mesh
{ 
  public:
    Mesh(int nz, 
         double L,
         double area);
    ~Mesh();
  
  private:
    int _nz;
    double _L;
    double _area;
    std::vector<double> nodes;
    std::vector<double> centroids;

};
