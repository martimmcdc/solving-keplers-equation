"""
This script is an implementation of a direct substitution method for solving Kepler's equation.
The initial guesses are taken to be x0 = M + 0.85*e due to simplicity.
The iterations stop after a precision epsilon on the functions value (which we wish to be f(x)=0) is reached.

This measure of error is discussed on a pdf file available on:
https://github.com/martimmcdc/solving-keplers-equation

In order to minimize calculations but still properly evaluate the precision of each iteration,
the error is considered to be that of the previous iteration, so that,
when the desired precision is reached, the algorithm runs one additional time.
"""

### imports
import numpy as np


def iterative_solver(e,M,epsilon=1e-9,iter_counter=False):
    # initial guess x0 = M
    x = M + 0.85*e # first iteration
    filt = np.ones(len(M),bool)
    counter = 0
    while filt.any(): 
        array1 = x[filt].copy()            # values to be changed
        array2 = M[filt] + e*np.sin(array1) # substitution
        x[filt] = array2
        filt[filt] = np.abs(array1 - array2)>epsilon # filter values which have not reached precision of epsilon 
        counter += 1
    if iter_counter:
        return x,counter
    return x

# Test function
if __name__=="__main__":
    
    import matplotlib.pyplot as plt,time

    # Parameters
    N = 1000000
    e = 0.5

    print("\n##### PARAMETERS #####")
    print("# len(M) = %d"%N)
    print("# e = %.2f"%e)
    #print("# Iterations: %d"%N_it)
    print("######################")

    # Create ell array from E
    E_true = (np.pi*(np.arange(N)+0.5))/N
    M_input = E_true - e*np.sin(E_true)

    # Time the function
    init = time.time()
    E_out,iters = iterative_solver(e,M_input,iter_counter=True)
    runtime = time.time()-init
    print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e."%(runtime*1000.,np.mean(np.abs(E_out-E_true))))
    print(iters,'iterations\n')


    # Plot values calculated and compare with pre-calculated grid
    sympy_grid = np.loadtxt('sympy_200x200grid.txt')[::30,:]
    M = np.linspace(0,np.pi,200)
    e = np.arange(0,1,1/200)[::30]
    for power in range(6,13,3):
        for i in range(len(e)):
            vals = iterative_solver(e[i],M,epsilon=10**(-power))
            sympy_vals = sympy_grid[i,:]
            plt.plot(M,vals-sympy_vals,label='e = {}'.format(e[i]))
        plt.xlabel('$M$')
        plt.ylabel('$\Delta E$')
        plt.title('$\\epsilon = 10^{}$'.format('{'+str(-power)+'}'))
        plt.legend()
        plt.grid()
        plt.show()