### imports
from numpy import sin

### Método iterativo
def iterative_solver(e,M,epsilon=1e-9):
	x0 = M                  # initial guess
	x = M + e*sin(x0)       # primeira iteração
	while abs(x0-x)>x*epsilon: # defini o erro como erro relativo entre iterações sucessivas
		x0 = x                 # atualizar último valor calculado
		x = M + e*sin(x0)   # calcular seguinte
	return x