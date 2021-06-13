# Comparison of my implementation of Kepler's Goat Herd vs the original

# This script generates heatmaps of the logarithms of runtimes and errors...
# ...of each method for a grid of 200x200 values of eccentricity and mean anomaly.

# The error considered is relative to the values previously calculated through the Sympy library...
# ...and saved in a .txt file, which is imported.

# The Goat Herd method is NOT an iterative method, rather it involves a complex contour integration...
# ...so its accuracy depends on the number of intervals the integral is divided into.
# This number is taken to be N=10, as used by the author of the original implementation.

# My implementation has an extra step which might account for the slightly higher runtime, which is...
# ...allowing the mean anomaly input both to be a single value (scalar) or a whole range (array) of values...
# ...whereas the original implementation only takes arrays as input.

# Due to the nature of this method values of e=0 and e=1 are not allowed, but the original implementation...
# ...also fails at some extreme values of M=0 and M=pi, so these will not be considered.

# The results of this test are printed to the console and can be seen to be very similar!


### imports ###
import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit
from philcox import philcox
from goat_herd_solver import goat_herd_solver


### Sympy calculated values to compare with ###
sympy_solutions = np.loadtxt('sympy_200x200grid.txt',dtype=float)


### Grid values ###
N = 200                    # number of grid points for e and M
M = np.linspace(0,np.pi,N) # mean anomaly values in [0,pi]
e = np.arange(0,1,1/N)     # eccentricity values in [0,1)


### Calculation ###

Nt = 10 # number of calculation runtimes to average over
zeros = np.zeros([2,len(e),len(M)],float)  # array to store the roots
times = np.zeros(zeros.shape,float)        # array to store the run time
methods = [goat_herd_solver,philcox] # methods to compare

for i in range(2):
	method = methods[i]
	name = method.__name__ # method name to feed timeit function
	set_up = '''import numpy as np;from {} import {}'''.format(name,name) # setup to be subtracted from runtime estimate
	M_array = M[1:-1]
	for k in range(1,len(e)-1):
		ek = e[k]
		statement = '''{}({},np.{})'''.format(name,ek,repr(M_array)) # statement to be timed
		times[i,1:-1,k] = timeit(stmt=statement,setup=set_up,number=Nt) # runtime for 10 calculations
		zeros[i,1:-1,k] = method(ek,M_array) # values of iteration number
times /= Nt # average
times = times[:,1:-1,1:-1]
errors = np.abs(zeros[:,1:-1,1:-1].copy()- sympy_solutions[1:-1,1:-1]) # error values
times += 1e-16  # avoid errors when taking log(times) of times=0
errors += 1e-16 # avoid errors when taking log(errors) of errors=0


### Plot ###

# figure
fig,axs = plt.subplots(2,3,
	gridspec_kw={'width_ratios':[1,1,0.05],
				 'height_ratios':[1,1],
	             'hspace':0.05,
	             'wspace':0.05})

# relative values for colorbar
min_val1 = np.min(np.log10(times))
max_val1 = np.max(np.log10(times))
min_val2 = np.min(np.log10(errors))
max_val2 = np.max(np.log10(errors))

# heatmaps
ax1 = axs[0,0].imshow(np.log10(times[0,:,:]),vmin=min_val1,vmax=max_val1,origin='lower')
ax2 = axs[0,1].imshow(np.log10(times[1,:,:]),vmin=min_val1,vmax=max_val1,origin='lower')
ax3 = axs[1,0].imshow(np.log10(errors[0,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')
ax4 = axs[1,1].imshow(np.log10(errors[1,:,:]),vmin=min_val2,vmax=max_val2,origin='lower')

# axes for colorbars
ax5 = axs[0,2]
ax6 = axs[1,2]
fig.colorbar(ax2,cax=ax5,label='$\\log10(\Delta t)$ , $\Delta t$ in seconds')
fig.colorbar(ax4,cax=ax6,label='$\\log10(n)$ , $n = $absolute error')

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
axs[0,0].set_title('My implementation')
axs[0,1].set_title('Original')

# save and show
plt.savefig('test3_out.pdf',bbox_inches='tight')
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
print(phrases[0] + '     {:.3e}     |  {:.3e}'.format(mean1,mean2))
print(phrases[1] + '     {:.3e}     |  {:.3e}'.format(std1,std2))
print(phrases[2] + '     {:.3e}     |  {:.3e}'.format(min1,min2))
print(phrases[3] + '     {:.3e}     |  {:.3e}'.format(max1,max2))

phrases = np.array(['Mean error: ','Standard deviation: ','Min. error: ','Max. error: '])
phrases[0] = ' '*(len(phrases[1])-len(phrases[0]))+phrases[0]
phrases[2] = ' '*(len(phrases[1])-len(phrases[2]))+phrases[2]
phrases[3] = ' '*(len(phrases[1])-len(phrases[3]))+phrases[3]
[mean1,mean2] = np.mean(errors,axis=(1,2))
[std1,std2] = np.std(errors,axis=(1,2))
[min1,min2] = np.min(errors,axis=(1,2))
[max1,max2] = np.max(errors,axis=(1,2))

print('\n')
print(' '*len(phrases[1]) + 'My implementation  |  Original')
print(phrases[0] + '     {:.3e}     |  {:.3e}'.format(mean1,mean2))
print(phrases[1] + '     {:.3e}     |  {:.3e}'.format(std1,std2))
print(phrases[2] + '     {:.3e}     |  {:.3e}'.format(min1,min2))
print(phrases[3] + '     {:.3e}     |  {:.3e}'.format(max1,max2))
print('\n')
