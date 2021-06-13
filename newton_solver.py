### imports
import numpy as np,time

### Newton-Raphson method
def newton_solver(e, M, epsilon=1e-9, iter_counter=False):
    # initial guess x0 = M
    x = M + e*np.sin(M)/(1-e*np.cos(M)) # first iteration
    error = x*epsilon + 1
    counter = 1
    while error>abs(x)*epsilon: # relative error between successive iterations 
        x0 = x                                          # update last to calculated value
        x -= (x0 - e*np.sin(x0) - M)/(1 - e*np.cos(x0)) # calculate next value
        error = abs(x0-x)
        counter += 1

    if iter_counter:
        return x, counter

    return x

def newton_solver2(e,M,epsilon=1e-6,iter_counter=False):
    # initial guess x0 = M
    x = M + e*np.sin(M)/(1 - e*np.cos(M)) # first iteration
    filt = np.abs(M-x)>np.abs(x)*epsilon  # filter values which have not reached precision of epsilon
    counter = 1

    while filt.any(): 
        x0 = x.copy()                          # update current guess
        array = x0[filt].copy()                # values to be changed
        h0 = array - e*np.sin(array) - M[filt] # function
        h1 = 1-e*np.cos(array)                 # first derivative
        x[filt] -= h0/h1                       # newton-raphson correction
        filt = np.abs(x0-x)>np.abs(x)*epsilon  # filter values which have not reached precision of epsilon 
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