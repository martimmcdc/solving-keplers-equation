# Grid generator to compare values obtained by other methods

### imports
from sympy_solver import sympy_solver
from numpy import pi,arange,linspace,empty,savetxt

N = 200
array = empty([N,N],float)
M = linspace(0,pi,N)
e = arange(0,1,1/N)

for i in range(N):
	ei = e[i]
	for j in range(N):
		array[i,j] = sympy_solver(ei,M[j])

savetxt('sympy_200x200grid.txt',array)