### imports
from numpy import sin

### Iterative method
def iterative_solver(e,M,epsilon=1e-9,iter_counter=False):
    x0 = M                  # initial guess
    x = M + e*sin(x0)       # first iteration
    counter = 1
    while abs(x0-x)>x*epsilon: # relative error between successive iterations
        x0 = x                 # update to last calculated value
        x = M + e*sin(x0)      # calculate next value
        counter += 1

    if iter_counter:
        return x,counter

    return x
