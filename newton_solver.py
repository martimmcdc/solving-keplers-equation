### imports
from numpy import sin,cos

### Newton-Raphson method
def newton_solver(e, M, epsilon=1e-9, iter_counter=False):
    x0 = M                                        # initial guess
    x = x0 - (x0 - e*sin(x0) - M)/(1 - e*cos(x0)) # first iteration
    counter = 1
    while abs(x0-x)>x*epsilon:                        # relative error between successive iterations 
        x0 = x                                        # update last to calculated value
        x = x0 - (x0 - e*sin(x0) - M)/(1 - e*cos(x0)) # calculate next value
        counter += 1

    if iter_counter:
        return x, counter

    return x
