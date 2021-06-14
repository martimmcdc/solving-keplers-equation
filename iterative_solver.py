### imports
import numpy as np

### Iterative method
def iterative_solver(e,M,epsilon=1e-9,iter_counter=False):
    # initial guess x0 = M
    x = e*np.sin(M) + M # first iteration
    error = np.abs(x-e*np.sin(x)-M)
    counter = 1
    while error>epsilon: # relative error between successive iterations
        x = e*np.sin(x) + M    # calculate next value
        esinx = e*np.sin(x)
        error = np.abs(x-esinx-M)
        counter += 1

    if iter_counter:
        return x,counter

    return x

def iterative_solver2(e,M,epsilon=1e-9,iter_counter=False):
    # initial guess x0 = M
    x = M + e*np.sin(M) # first iteration
    filt = np.abs(x-np.sin(x)-M)>epsilon # filter values which have not reached precision of epsilon
    counter = 1
    while filt.any(): 
        array = x[filt]#.copy()           # values to be changed
        Marray = M[filt]
        array = Marray + e*np.sin(array) # newton-raphson correction
        x[filt] = array.copy()
        filt[filt] = np.abs(array-e*np.sin(array)-Marray)>epsilon # filter values which have not reached precision of epsilon 
        counter += 1

    if iter_counter:
        return x,counter
    return x