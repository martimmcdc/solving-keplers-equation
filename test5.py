"""
Comparison of Newton-Raphson vs Iterative vs Kepler's Goat Herd (my implementation) vs Nijenhuis vs Danby methods

This script generates heatmaps of the logarithms of both runtimes and errors of each method
for a grid of 200x200 values of eccentricity and mean anomaly.

The error considered is relative to the values previously calculated through the Sympy library
and saved in a .txt file, which is imported.

The Goat Herd method is NOT an iterative method, rather it involves a complex contour integration
so its accuracy depends on the number of intervals the integral is divided into.
This number is taken to be N=10, as used by the author of the original implementation.
Due to the nature of this method values of e=0 and e=1 are not allowed.

The results of this test are printed to the console and exported as a .txt file.
They are best for the Newton-Raphson method, which converges faster for most values.
"""

### imports ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Sympy calculated values to compare with ###
sympy_solutions = np.loadtxt('sympy_200x200grid.txt',dtype=float)


### Grid values ### - used for plot axes below
N = 200                    # number of grid points for e and M
M = np.linspace(0,np.pi,N) # mean anomaly values in [0,pi]
e = np.arange(0,1,1/N)     # eccentricity values in [0,1)

### Calculation ###
names = ['Newton-Raphson','Iterative','Kepler\'s Goat Herd','Nijenhuis','Danby','CORDIC','Murison']
methods = ['newton','iterative','goat_herd','nijenhuis','danby','cordic','murison'] # methods

zeros = np.zeros([len(methods),len(e),len(M)],float) # array to store the roots
times = np.zeros(zeros.shape,float) # array to store the run time
for i in range(len(methods)):
	name = methods[i]
	fname_times = 'cpp_solver_grids/'+name+'_200x200grid_runtime.txt'
	fname_zeros = 'cpp_solver_grids/'+name+'_200x200grid_zeros.txt'
	times[i,:,:] = np.loadtxt(fname_times)
	zeros[i,:,:] = np.loadtxt(fname_zeros)
times = times[:,1:-1,1:-1]
errors = np.abs(zeros[:,1:-1,1:-1].copy()- sympy_solutions[1:-1,1:-1]) # error values
log10times = np.log10(times+1e-20)  # avoid errors when taking log(times) of times=0
log10errors = np.log10(errors+1e-20) # avoid errors when taking log(errors) of errors=0

### Plot ###

# same relative values for all plots
min_val1 = np.min(log10times)
max_val1 = np.max(log10times)
min_val2 = np.min(log10errors)
max_val2 = np.max(log10errors)

fig,axs = plt.subplots(2,len(methods)+1,
	figsize=(2.5*6.4,4.8),
	gridspec_kw={'width_ratios':[1,1,1,1,1,1,1,0.05],
				 'height_ratios':[1,1],
	             'hspace':0.05,
	             'wspace':0.05})
for i in range(len(methods)):
	ax0 = axs[0,i].imshow(log10times[i,:,:],vmin=min_val1,vmax=max_val1,cmap=plt.cm.hot,origin='lower')
	ax1 = axs[1,i].imshow(log10errors[i,:,:],vmin=min_val2,vmax=max_val2,origin='lower')
	axs[0,i].set_title(names[i])

# colorbars
fig.colorbar(ax0,cax=axs[0,-1],shrink=0.6,label='$\\log10(\Delta t)$ , $\Delta t$ in seconds')
fig.colorbar(ax1,cax=axs[1,-1],label='$\\log10(n)$ , $n = $absolute error')

# plot labeling and ticks
eticks = (e[::N//10]*N)[1:]
elabels= np.round(e[::N//10],1)[1:]
Mticks = np.linspace(0,N,5)
Mlabels= ['0','$\\frac{\pi}{4}$','$\\frac{\pi}{2}$','$\\frac{3\pi}{4}$','$\\pi$']

for i in range(len(methods)):
	axs[0,i].tick_params(direction='in',labelbottom=False)
	axs[1,i].set_xticks(Mticks)
	axs[1,i].set_xticklabels(Mlabels)
	axs[1,i].set_xlabel('$M$')
for i in range(2):
	axs[i,0].set_yticks(eticks)
	axs[i,0].set_yticklabels(elabels)
	axs[i,0].set_ylabel('$e$')
	for j in range(1,len(methods)):
		axs[i,j].tick_params(direction='in',labelleft=False)
axs[1,0].tick_params(direction='in')

# save and show
plt.savefig('test5_out.pdf',bbox_inches='tight')
plt.show()


### Statistics ###

phrases = ['Mean runtime','Std. dev.','Min. runtime','Max. runtime','Mean error','Std. dev.','Min. error','Max. error']

stats = np.empty([8,len(methods)])
stats[0,:] = np.mean(times,axis=(1,2))
stats[1,:] = np.std(times,axis=(1,2))
stats[2,:] = np.min(times,axis=(1,2))
stats[3,:] = np.max(times,axis=(1,2))
stats[4,:] = np.mean(errors,axis=(1,2))
stats[5,:] = np.std(errors,axis=(1,2))
stats[6,:] = np.min(errors,axis=(1,2))
stats[7,:] = np.max(errors,axis=(1,2))

pd.options.display.float_format = '{:.3e}'.format
df = pd.DataFrame(data=stats,index=phrases,columns=names)
df.to_csv('test5_stats.txt',sep='\t')
print('\n',df,'\n')