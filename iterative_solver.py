### imports
import numpy as np

### Iterative method
def iterative_solver(e,M,epsilon=1e-9,iter_counter=False):
    # initial guess x0 = M
    x = M + e*np.sin(M) # first iteration
    error = x*epsilon + 1
    counter = 1
    while error>abs(x)*epsilon: # relative error between successive iterations
        x0 = x                 # update to last calculated value
        x = M + e*np.sin(x0)      # calculate next value
        error = abs(x0-x)
        counter += 1

    if iter_counter:
        return x,counter

    return x

def iterative_solver2(e,M,epsilon=1e-6,iter_counter=False):
    # initial guess x0 = M
    x = M + e*np.sin(M) # first iteration
    filt = np.abs(M-x)>np.abs(x)*epsilon  # filter values which have not reached precision of epsilon
    counter = 1

    while filt.any(): 
        x0 = x.copy()                         # update current guess
        array = x0[filt].copy()               # values to be changed
        x[filt] = M[filt] + e*np.sin(array)   # newton-raphson correction
        filt = np.abs(x0-x)>np.abs(x)*epsilon # filter values which have not reached precision of epsilon 
        counter += 1

    if iter_counter:
        return x,counter
    return x