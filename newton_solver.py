### imports
from numpy import sin,cos

### Método de Newton-Raphson
def newton_solver(e,M,epsilon=1e-9,iter_counter=True):
	x0 = M                                        # initial guess
	x = x0 - (x0 - e*sin(x0) - M)/(1 - e*cos(x0)) # primeira iteração
	counter = 1
	while abs(x0-x)>x*epsilon: # defini o erro como erro relativo entre iterações sucessivas
		x0 = x                 # atualizar último valor calculado
		x = x0 - (x0 - e*sin(x0) - M)/(1 - e*cos(x0)) # calcular seguinte
		counter += 1
	if iter_counter==True:
		return x,counter
	return x