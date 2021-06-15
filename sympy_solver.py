### imports
from sympy import Symbol,nsolve,sin

### Sympy solver method
def sympy_solver(e,M):
	x = Symbol('E')
	return float(nsolve(x-e*sin(x)-M,x,M,prec=20))

if __name__ == '__main__':
	E1 = sympy_solver(0.5,1)
	print(E1)