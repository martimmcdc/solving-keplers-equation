### imports
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from time import time

### Exemplo de órbita de excentricidade e em fase aleatórias
P = 365.25*24*3600                     # período orbital da Terra
n = 2*np.pi/P                          # velocidade angular
t = np.arange(0,P)                     # tempo de uma órbita em array, tp = 0
M = n*t[np.random.randint(0,len(t)-1)] # valor aleatório de M(t)
e = np.random.random()                 # valor aleatório de e em [0,1)
E = np.linspace(0,2*np.pi,1000)             # valores de E para verificar soluções graficamente
print('e = ',e,' ; M = ',M)

### Solução com Sympy
x = sp.Symbol('E')
# Timer
start = time()
zero_sympy = sp.nsolve(x-e*sp.sin(x)-M,x,1)
stop = time()
zero_sympy = float(zero_sympy)
print('Sympy solution = ',zero_sympy,' in ',stop-start,'s')

### Método iterativo
def Kepler_iter(e,M,epsilon):
	x0 = 1                     # initial guess
	x = M + e*np.sin(x0)       # primeira iteração
	while abs(x0/x-1)>epsilon: # defini o erro como erro relativo entre iterações sucessivas
		x0 = x                 # atualizar último valor calculado
		x = M + e*np.sin(x0)   # calcular seguinte
	return x
#Timer
start = time()
zero_iter = Kepler_iter(e,M,1e-12)
stop = time()
print('Iterative solution = ',zero_iter,' in ',stop-start,'s')

### Método de Newton-Raphson
def Kepler_Newton(e,M,epsilon):
	x0 = 1                                              # initial guess
	x = x0 - (x0 - e*np.sin(x0) - M)/(1 - e*np.cos(x0)) # primeira iteração
	while abs(x0/x-1)>epsilon: # defini o erro como erro relativo entre iterações sucessivas
		x0 = x                 # atualizar último valor calculado
		x = x0 - (x0 - e*np.sin(x0) - M)/(1 - e*np.cos(x0)) # calcular seguinte
	return x
# Timer
start = time()
zero_Newton = Kepler_Newton(e,M,1e-12)
stop = time()
print('Newton-Raphson solution = ',zero_Newton,' in ',stop-start,'s')

### Erros relativos
print('Erro iterativo = ',abs(zero_iter/zero_sympy-1)*100,'%')
print('Erro Newton = ',abs(zero_Newton/zero_sympy-1)*100,'%')

### Gráfico
plt.plot(E,E-e*np.sin(E)-M)
plt.axvline(zero_sympy,color='k',ls='--')
plt.xlabel('$E$')
plt.ylabel('$E(t) - e\cdot\sin{E(t)}$')
plt.xlim(0,2*zero_sympy)
plt.tick_params(direction='in')
plt.grid()
plt.show()

