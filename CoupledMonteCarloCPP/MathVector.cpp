#include "MathVector.h"
#include <vector>

// Destructor
DoubleVec::~DoubleVec() {}

DoubleVec::DoubleVec(double v, int N)
{
  // assign to T as N entries of value v
  _T.assign(N, v);
  _size = N;
}

DoubleVec::DoubleVec(std::vector<double> v)
{
  _T = v;
  _size = _T.size();
}


