import numpy as np
print("--------------------------------------------")
failures = []

"""
Testing cram
"""
from cram import *

"""Test 1 - Test depletion"""
print("NOW RUNNING TEST 1")
print("--------------------------------------------")
removal_f = 1.15741E-07
decay_x = np.log(2)/9.14/3600
A = -1.0 * np.array([[-removal_f-decay_x, 0.0], [removal_f, -decay_x]])
dt = 3600*100 # seconds -> 100 hours

N0 = np.array([1.0, 0.0])
D = Depleter(A=A, dt=dt)
N1 = D.solve(N0=N0)
print("RESULTS --- Test 1")
print("--------------------------------------------")
print("result is:",N1)
answer =  np.array([4.87932275e-04, 2.07600557e-05])
print("answer is:",answer)
print("diff is:  ", 1.0-answer/N1)
test1Failed = False
for this in (1.0-answer/N1):
  if abs(this) > 1e-8:
    test1Failed = True
failures.append(test1Failed)
print("--------------------------------------------")
print()

"""
Testing MC
"""

"""Test 2 - critical slab"""
from MonteCarlo import *
print("NOW RUNNING TEST 2")
print("--------------------------------------------")
matCritical = Material(s=0.01, a=1, nu=1, f=1)
ele1 = Element(matCritical, 5.12312)
msh = Mesh1D(elements=[ele1], left='r', right='r')
solver = MonteCarlo(mesh=msh, npg=50000, nsk=10, ngen=100)
solver.solve()

print("RESULTS --- Test 2")
print("--------------------------------------------")
print("result is:", solver.kest, "+/-", solver.std_dev)
print("answer is:", 1.0, "+/-", 0.0)
print("diff is:  ", 1.0-solver.kest)
if abs(1.0 - solver.kest) > 10e-5:
  test2Failed = True
else:
  test2Failed = False
failures.append(test2Failed)
print("--------------------------------------------")
print()











"""Final summary of test failures"""
print("--------------------------------------------")
print("Summary of tests:")
print("If any of these are true, please investigate.")
print("--------------------------------------------")
for idx, this in enumerate(failures):
  print("Test 1 Failed = ", this)
print("--------------------------------------------")
