### imports
from sympy import Symbol,nsolve,sin

### Solução com Sympy
def sympy_solver(e,M):
	x = Symbol('E')
	return float(nsolve(x-e*sin(x)-M,x,M))