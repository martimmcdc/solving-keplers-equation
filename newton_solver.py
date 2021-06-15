### imports
import numpy as np,time
import matplotlib.pyplot as plt
from sympy_solver import sympy_solver

### Newton-Raphson method
def newton_solver(e, M, epsilon=1e-9, iter_counter=False):
    # initial guess x0 = M
    error = e*np.sin(M)
    x = M + error/(1-e*np.cos(M)) # first iteration
    counter = 1
    while error>epsilon: # relative error between successive iterations
        h0 = x - e*np.sin(x) - M # function
        h1 = 1 - e*np.cos(x)     # first derivative
        x -= h0/h1               # newton-raphson correction
        error = np.abs(h0)       # error in function value
        counter += 1
    if iter_counter:
        return x, counter
    return x

def newton_solver2(e,M,epsilon=1e-9,iter_counter=False):
    # initial guess x0 = M
    esinM = e*np.sin(M)
    x = M + esinM/(1 - e*np.cos(M)) # first iteration
    filt = esinM>epsilon  # filter values which have not reached precision of epsilon
    counter = 1
    while filt.any():
        array = x[filt].copy()                 # values to be changed
        h0 = array - e*np.sin(array) - M[filt] # function
        h1 = 1-e*np.cos(array)                 # first derivative
        x[filt] -= h0/h1                       # newton-raphson correction
        filt[filt] = np.abs(h0)>epsilon        # filter values which have not reached precision of epsilon
        counter += 1
    if iter_counter:
        return x,counter
    return x



if __name__=="__main__":
    """Test the Python function above with a simple example"""

    # Parameters
    N_ell = 1000000
    eccentricity = 0.5

    print("\n##### PARAMETERS #####")
    print("# N_ell = %d"%N_ell)
    print("# Eccentricity = %.2f"%eccentricity)
    #print("# Iterations: %d"%N_it)
    print("######################")

    # Create ell array from E
    E_true = (np.pi*(np.arange(N_ell)+0.5))/N_ell
    ell_input = E_true - eccentricity*np.sin(E_true)

    # Time the function
    init = time.time()
    E_out,iters = newton_solver2(eccentricity,ell_input,iter_counter=True)#,N_it)
    runtime = time.time()-init
    print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e.\n"%(runtime*1000.,np.mean(np.abs(E_out-E_true))))
    print(iters,'iterations')


    # Plot values calculated and compare with pre-calculated grid
    sympy_grid = np.loadtxt('sympy_200x200grid.txt')[::30,:]
    M = np.linspace(0,np.pi,200)
    e = np.arange(0,1,1/200)[::30]
    for power in range(6,16,3):
        for i in range(len(e)):
            vals = newton_solver2(e[i],M,epsilon=10**(-power))
            sympy_vals = sympy_grid[i,:]
            plt.plot(M,vals-sympy_vals,label='e = {}'.format(e[i]))
        plt.xlabel('$M$')
        plt.ylabel('$\Delta E$')
        plt.title('$\\epsilon = 10^{}$'.format('{'+str(-power)+'}'))
        plt.legend()
        plt.grid()
        plt.show()















