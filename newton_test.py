### imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from newton_solver import newton_solver

### Valores de anomalia média e excentricidade
N = 200
M = np.linspace(0,2*np.pi,N) # valores de M(t)
e = np.arange(0,1,1/N)       # valores de e em [0,1)
eticks = e[::40]*N
elabels= e[::40]
Mticks = np.linspace(0,N,5)
Mlabels= ['0','$\\frac{\pi}{2}$','$\\pi$','$\\frac{3\pi}{2}$','$2\pi$']
zeros = np.zeros([len(e),len(M)],float)
iterations = np.zeros(zeros.shape,float)
times = np.zeros(zeros.shape,float)

for j in range(len(e)):
		e_val = e[j]
		for k in range(len(M)):
			start = time()
			zeros[j,k],iterations[j,k] = newton_solver(e_val,M[k],epsilon=1e-6)
			times[j,k] = time() - start

### Gráfico
fig = plt.figure()
plt.imshow(np.log(times),cmap=plt.cm.hot)
plt.xticks(ticks=eticks,labels=elabels)
plt.yticks(ticks=Mticks,labels=Mlabels)
plt.xlabel('$e$')
plt.ylabel('$M$')
plt.title('Newton-Raphson solver')
plt.colorbar(label='$\\log{\Delta t}$')
plt.show()