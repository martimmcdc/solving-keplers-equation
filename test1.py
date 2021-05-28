### imports
import numpy as np
import matplotlib.pyplot as plt
from time import time
from iterative_solver import iterative_solver
from newton_solver import newton_solver

N = 300                    # number of grid points for e and M
M = np.linspace(0,np.pi,N) # mean anomaly values in [0,pi]
e = np.arange(0,1,1/N)     # eccentricity values in [0,1)

### Cálculo
zeros = np.zeros([2,len(e),len(M)],float)  # array to store the roots
iterations = np.zeros(zeros.shape,float)   # array to store the number of iterations
times = np.zeros(zeros.shape,float)        # array to store the run time
methods = [newton_solver,iterative_solver] # methods to compare
for i in range(2):
	method = methods[i]
	for j in range(len(M)):
		Mj = M[j]
		for k in range(len(e)):
			start = time()
			zeros[i,j,k],iterations[i,j,k] = method(e[k],Mj,epsilon=1e-9,iter_counter=True) 
			times[i,j,k] = time() - start

### Gráfico

eticks = (e[::N//10]*N)[1:]
elabels= np.round(e[::N//10],1)[1:]
Mticks = np.linspace(0,N,5)
Mlabels= ['0','$\\frac{\pi}{4}$','$\\frac{\pi}{2}$','$\\frac{3\pi}{4}$','$\\pi$']

fig,axs = plt.subplots(2,3,figsize=(6,6),
	gridspec_kw={'width_ratios':[1,1,0.05],
	             'hspace':0.05,
	             'wspace':0.05})
ax1 = axs[0,0].imshow(np.log(times[0,:,:]),cmap=plt.cm.hot,origin='lower')
ax2 = axs[0,1].imshow(np.log(times[1,:,:]),cmap=plt.cm.hot,origin='lower')
ax3 = axs[1,0].imshow(np.log(iterations[0,:,:]),origin='lower')
ax4 = axs[1,1].imshow(np.log(iterations[1,:,:]),origin='lower')
ax5 = axs[0,2]
ax6 = axs[1,2]

axs[1,0].set_xticks(Mticks)
axs[1,0].set_xticklabels(Mlabels)
axs[1,1].set_xticks(Mticks)
axs[1,1].set_xticklabels(Mlabels)
axs[0,0].set_yticks(eticks)
axs[0,0].set_yticklabels(elabels)
axs[1,0].set_yticks(eticks)
axs[1,0].set_yticklabels(elabels)
axs[0,0].tick_params(direction='in',labelbottom=False)
axs[0,1].tick_params(direction='in',labelbottom=False,labelleft=False)
axs[1,0].tick_params(direction='in')
axs[1,1].tick_params(direction='in',labelleft=False)

axs[1,0].set_xlabel('$M$')
axs[1,1].set_xlabel('$M$')
axs[0,0].set_ylabel('$e$')
axs[1,0].set_ylabel('$e$')

axs[0,0].set_title('Newton-Raphson')
axs[0,1].set_title('Iterative')
fig.colorbar(ax2,cax=ax5,label='$\\log(\Delta t)$ , $\Delta t$ in seconds')
fig.colorbar(ax4,cax=ax6,label='\n$\\log(n)$ , $n = $number of iterations')
plt.show()