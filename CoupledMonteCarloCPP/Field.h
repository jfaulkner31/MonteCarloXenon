#include <vector>
class Field
{ 
  public:
    Field(double initial_value);
    ~Field();
  
  private:
    std::vector<double> _T;
};
