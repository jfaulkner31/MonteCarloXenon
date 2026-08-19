#include <vector>

class DoubleVec
{
  public:
    DoubleVec(double v, int n);
    DoubleVec(std::vector<double> v);
    ~DoubleVec();
  

  private:
    std::vector<double> _T;
    int _size;
};
