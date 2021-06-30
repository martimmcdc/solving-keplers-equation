"""
Grid generator to compare values obtained by other methods
"""

### imports
from sympy_solver import sympy_solver
import numpy as np

N = 200
array = np.empty([N,N],float)
M = np.linspace(0,pi,N)
e = np.arange(0,1,1/N)

for i in range(N):
	ei = e[i]
	array[i,:] = np.vectorize(sympy_solver)(ei,M)

np.savetxt('sympy_200x200grid.txt',array)