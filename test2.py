### imports ###
import numpy as np
import matplotlib.pyplot as plt
from time import time
from iterative_solver import iterative_solver
from newton_solver import newton_solver
from goat_herd_solver import goat_herd_solver

N = 200                    # number of grid points for e and M
M = np.linspace(0,np.pi,N) # mean anomaly values in [0,pi]
e = np.arange(0,1,1/N)     # eccentricity values in [0,1)

### Calculation ###
zeros = np.zeros([3,len(e),len(M)],float)  # array to store the roots
iterations = np.zeros(zeros.shape,float)   # array to store the number of iterations
times = np.zeros(zeros.shape,float)        # array to store the run time
methods = [newton_solver,iterative_solver,goat_herd_solver] # methods to compare
for i in range(3):
	method = methods[i]
	for j in range(len(M)):
		Mj = M[j]
		for k in range(len(e)):
			start = time()
			zeros[i,j,k] = method(e[k],Mj) 
			times[i,j,k] = time() - start

### Sympy calculated values to compare with
sympy_solutions = np.loadtxt('sympy_200x200grid.txt',dtype=float)
errors = zeros.copy()
errors = np.abs(sympy_solutions-errors)

### Plot ###
eticks = (e[::N//10]*N)[1:]
elabels= np.round(e[::N//10],1)[1:]
Mticks = np.linspace(0,N,5)
Mlabels= ['0','$\\frac{\pi}{4}$','$\\frac{\pi}{2}$','$\\frac{3\pi}{4}$','$\\pi$']

# same relative values for all plots
min_val1 = np.min(np.log10(times))
max_val1 = np.max(np.log10(times))
min_val2 = np.min(np.log10(errors))
max_val2 = np.max(np.log10(errors))

fig,axs = plt.subplots(2,4,figsize=(6,6),
	gridspec_kw={'width_ratios':[1,1,1,0.05],
				 'height_ratios':[1,1],
	             'hspace':0.05,
	             'wspace':0.05})
ax1 = axs[0,0].imshow(np.log10(times[0,:,:]),vmin=min_val1,vmax=max_val1,cmap=plt.cm.hot,origin='lower')
ax2 = axs[0,1].imshow(np.log10(times[1,:,:]),vmin=min_val1,vmax=max_val1,cmap=plt.cm.hot,origin='lower')
ax3 = axs[0,2].imshow(np.log10(times[2,:,:]),vmin=min_val1,vmax=max_val1,cmap=plt.cm.hot,origin='lower')
ax4 = axs[1,0].imshow(np.log10(errors[0,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')
ax5 = axs[1,1].imshow(np.log10(errors[1,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')
ax6 = axs[1,2].imshow(np.log10(errors[2,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')
ax7 = axs[0,3]
ax8 = axs[1,3]

axs[1,0].set_xticks(Mticks)
axs[1,0].set_xticklabels(Mlabels)
axs[1,1].set_xticks(Mticks)
axs[1,1].set_xticklabels(Mlabels)
axs[1,2].set_xticks(Mticks)
axs[1,2].set_xticklabels(Mlabels)
axs[0,0].set_yticks(eticks)
axs[0,0].set_yticklabels(elabels)
axs[1,0].set_yticks(eticks)
axs[1,0].set_yticklabels(elabels)
axs[0,0].tick_params(direction='in',labelbottom=False)
axs[0,1].tick_params(direction='in',labelbottom=False,labelleft=False)
axs[0,2].tick_params(direction='in',labelbottom=False,labelleft=False)
axs[1,0].tick_params(direction='in')
axs[1,1].tick_params(direction='in',labelleft=False)
axs[1,2].tick_params(direction='in',labelleft=False)

axs[1,0].set_xlabel('$M$')
axs[1,1].set_xlabel('$M$')
axs[1,2].set_xlabel('$M$')
axs[0,0].set_ylabel('$e$')
axs[1,0].set_ylabel('$e$')

axs[0,0].set_title('Newton-Raphson')
axs[0,1].set_title('Iterative')
axs[0,2].set_title('Kepler\'s Goat Herd')
fig.colorbar(ax3,cax=ax7,label='$\\log10(\Delta t)$ , $\Delta t$ in seconds')
fig.colorbar(ax6,cax=ax8,label='\n$\\log10(n)$ , $n = $absolute error')
plt.savefig('test2_out.pdf')
plt.show()


### Print statistics ###
phrases = np.array(['Mean calculation time: ','Standard deviation: ','Min. calculation time: ','Max. calculation time: '])
phrases[1] = ' '*(len(phrases[0])-len(phrases[1]))+phrases[1]
[mean1,mean2,mean3] = np.mean(times,axis=(1,2))
[std1,std2,std3] = np.std(times,axis=(1,2))
[min1,min2,min3] = np.min(times,axis=(1,2))
[max1,max2,max3] = np.max(times,axis=(1,2))

print('\n')
print(' '*len(phrases[0]) + 'Newton-Raphson  |  Iterative  |  Kepler\'s Goat Herd')
print(phrases[0] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(mean1,mean2,mean3))
print(phrases[1] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(std1,std2,std3))
print(phrases[2] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(min1,min2,min3))
print(phrases[3] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(max1,max2,max3))

phrases = np.array(['Mean error: ','Standard deviation: ','Min. error: ','Max. error: '])
phrases[0] = ' '*(len(phrases[1])-len(phrases[0]))+phrases[0]
phrases[2] = ' '*(len(phrases[1])-len(phrases[2]))+phrases[2]
phrases[3] = ' '*(len(phrases[1])-len(phrases[3]))+phrases[3]
[mean1,mean2,mean3] = np.mean(errors,axis=(1,2))
[std1,std2,std3] = np.std(errors,axis=(1,2))
[min1,min2,min3] = np.min(errors,axis=(1,2))
[max1,max2,max3] = np.max(errors,axis=(1,2))

print('\n')
print(' '*len(phrases[1]) + 'Newton-Raphson  |  Iterative  |  Kepler\'s Goat Herd')
print(phrases[0] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(mean1,mean2,mean3))
print(phrases[1] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(std1,std2,std3))
print(phrases[2] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(min1,min2,min3))
print(phrases[3] + '   {:.3e}    |  {:.3e}  |    {:.3e}'.format(max1,max2,max3))
print('\n')


