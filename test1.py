# Comparison of Newton-Raphson vs Iterative (direct substitution) methods

# This script generates heatmaps of the logarithms of runtimes and iteration numbers...
# ...of each method for a grid of 200x200 values of eccentricity and mean anomaly.

# The root is considered to be found when the relative error between successive iterations...
# ...reaches a value of epsilon=1e-9, that is |E(i)-E(i+1)|/E(i+1)<epsilon

# The results of this test are printed to the console and can be seen to be best for...
# ...the Newton-Raphson method, which converges faster for most values


### imports ###
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from iterative_solver import iterative_solver
from newton_solver import newton_solver


### Grid values ###
N = 200                    # number of grid points for e and M
M = np.linspace(0,np.pi,N) # mean anomaly values in [0,pi]
e = np.arange(0,1,1/N)     # eccentricity values in [0,1)


### Calculation ###

Nt = 10 # number of calculation runtimes to average over
iterations = np.zeros([2,len(e),len(M)],float) # array to store the number of iterations
times = np.zeros([2,len(e),len(M)],float)      # array to store the run time
methods = [newton_solver,iterative_solver] # methods to compare

for i in range(2):
	method = methods[i]
	name = method.__name__                             # method name to feed timeit function
	set_up = '''from {} import {}'''.format(name,name) # setup to be subtracted from runtime estimate
	for j in range(N):
		ej = e[j]
		for k in range(N):
			Mk = M[k]
			statement = '''{}({},{},epsilon=1e-9,iter_counter=True)'''.format(name,ej,Mk) # statement to be timed
			times[i,j,k] = timeit(stmt=statement,setup=set_up,number=Nt)                  # runtime for 10 calculations
			iterations[i,j,k] = method(ej,Mk,epsilon=1e-9,iter_counter=True)[1] # values of iteration number
times /= Nt # average


### Plot ###

# figure
fig,axs = plt.subplots(2,3,
	gridspec_kw={'width_ratios':[1,1,0.05],
	             'hspace':0.05,
	             'wspace':0.05})

# relative values for colorbar
min_val1 = np.min(np.log10(times))
max_val1 = np.max(np.log10(times))
min_val2 = np.min(np.log10(iterations))
max_val2 = np.max(np.log10(iterations))

# heatmaps
ax1 = axs[0,0].imshow(np.log10(times[0,:,:]),vmin=min_val1,vmax=max_val1,cmap=plt.cm.hot,origin='lower')
ax2 = axs[0,1].imshow(np.log10(times[1,:,:]),vmin=min_val1,vmax=max_val1,cmap=plt.cm.hot,origin='lower')
ax3 = axs[1,0].imshow(np.log10(iterations[0,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')
ax4 = axs[1,1].imshow(np.log10(iterations[1,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')

# axes for colorbars
ax5 = axs[0,2]
ax6 = axs[1,2]
fig.colorbar(ax2,cax=ax5,label='$\\log(\Delta t)$ , $\Delta t$ in seconds')
fig.colorbar(ax4,cax=ax6,label='\n$\\log(n)$ , $n = $number of iterations')

# plot labeling
eticks = (e[::N//10]*N)[1:]
elabels= np.round(e[::N//10],1)[1:]
Mticks = np.linspace(0,N,5)
Mlabels= ['0','$\\frac{\pi}{4}$','$\\frac{\pi}{2}$','$\\frac{3\pi}{4}$','$\\pi$']
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

# save and show
plt.savefig('test1_out.pdf')
plt.show()


### Print statistics ###
phrases = np.array(['Mean calculation time: ','Standard deviation: ','Min. calculation time: ','Max. calculation time: '])
phrases[1] = ' '*(len(phrases[0])-len(phrases[1]))+phrases[1]
[mean1,mean2] = np.mean(times,axis=(1,2))
[std1,std2] = np.std(times,axis=(1,2))
[min1,min2] = np.min(times,axis=(1,2))
[max1,max2] = np.max(times,axis=(1,2))

print('\n')
print(' '*len(phrases[0]) + 'My implementation  |  Original')
print(phrases[0] + '   {:.3e}    |  {:.3e}'.format(mean1,mean2))
print(phrases[1] + '   {:.3e}    |  {:.3e}'.format(std1,std2))
print(phrases[2] + '   {:.3e}    |  {:.3e}'.format(min1,min2))
print(phrases[3] + '   {:.3e}    |  {:.3e}'.format(max1,max2))
print('\n')