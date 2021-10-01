"""
This script is an implementation of Danby's method for solving Kepler's equation.
The algorithm is basically a 3rd order version of the Newton-Raphson method, which is 1st order in the expansion terms.
The initial guesses are taken to be x0 = M + 0.85*e as suggested by Danby's 3rd paper on this matter.
The iterations stop after a precision epsilon on the functions value (which we wish to be f(x)=0) is reached.

This measure of error is discussed on a pdf file available on:
https://github.com/martimmcdc/solving-keplers-equation

In order to minimize calculations but still properly evaluate the precision of each iteration,
the error is considered to be that of the previous iteration, so that,
when the desired precision is reached, the algorithm runs one additional time.
"""

### imports
import numpy as np

### Danby solver
def danby_solver(e,M,epsilon=1e-9,iter_counter=False):
	x = M + 0.85*e # initial guess suggested by Danby III
	filt = e*np.abs(0.85 - np.sin(x))>epsilon 
	counter = 0
	while filt.any():
		array = x[filt].copy()
		h2 = e*np.sin(array)       # 2nd derivative
		h3 = e*np.cos(array)       # 3rd derivative
		h0 = array - h2 - M[filt]  # 0th derivative
		h1 = 1-h3                  # 1st derivative
		d1 = -h0/h1                # 1st order correction
		o2 = h1 + 0.5*d1*h2        # denominator for 2nd order
		d2 = -h0/o2                # second order
		d3 = -h0/(o2 + d2*d2*h3/6) # 3rd order correction
		x[filt] += d3              # update 
		filt[filt] = np.abs(h0)>epsilon
		counter += 1
	if iter_counter:
		return x,counter
	return x


if __name__=="__main__":
    """Test the Python function above with a simple example"""

    import matplotlib.pyplot as plt,time
    # Parameters
    N = 1000000
    e = 0.5

    print("\n##### PARAMETERS #####")
    print("# N_ell = %d"%N)
    print("# Eccentricity = %.2f"%e)
    print("######################")

    # Create M array from E
    E_true = (2.0*np.pi*(np.arange(N)+0.5))/N
    M_input = E_true - e*np.sin(E_true)

    # Time the function
    init = time.time()
    E_out,iters = danby_solver(e,M_input,iter_counter=True)
    runtime = time.time()-init

    print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e."%(runtime*1000.,np.mean(np.abs(E_out-E_true))))
    print(iters,'iterations\n')

    # Plot values calculated and compare with pre-calculated grid
    sympy_grid = np.loadtxt('sympy_200x200grid.txt')[::30,:]
    M = np.linspace(0,np.pi,200)
    e = np.arange(0,1,1/200)[::30]
    for power in range(6,16,3):
        for i in range(len(e)):
            vals = danby_solver(e[i],M,epsilon=10**(-power))
            sympy_vals = sympy_grid[i,:]
            plt.plot(M,vals-sympy_vals,label='e = {}'.format(e[i]))
        plt.xlabel('$M$')
        plt.ylabel('$\Delta E$')
        plt.title('$\\epsilon = 10^{}$'.format('{'+str(-power)+'}'))
        plt.legend()
        plt.grid()
        plt.show()