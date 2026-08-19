#include "Eigenvalue.h"

// Constructor
Eigenvalue::Eigenvalue(double ev) 
  : _keff(ev)
{
  // body --- do stuff here
}

// Destructor
Eigenvalue::~Eigenvalue() {}

// Methods


// Returns a reference to the eigenvalue
double* Eigenvalue::get()
{
  return &_keff;
}

// Sets the eigenvalue
void Eigenvalue::set(double keff)
{
  _keff = keff;
}
