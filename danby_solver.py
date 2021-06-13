### imports
import numpy as np,time

def danby_solver(e,M,epsilon=1e-9,iter_counter=False):
	x0 = M + 0.85*e # initial guess suggested by Danby
	x = x0.copy()
	filt = np.ones(len(M),bool)
	counter = 0
	while filt.any():
		array = x0[filt].copy()
		h1 = 1-e*np.cos(array)
		h2 = e*np.sin(array)
		h3 = 1 - h1
		h0 = array - h2 - M[filt]
		d1 = -h0/h1
		o2 = h1 + 0.5*d1*h2
		d2 = -h0/o2
		d3 = -h0/(o2 + d2*d2*h3/6)
		x[filt] += d3
		filt = np.abs(x0-x)>np.abs(x)*epsilon
		x0 = x.copy()
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
    E_true = (2.0*np.pi*(np.arange(N_ell)+0.5))/N_ell
    ell_input = E_true - eccentricity*np.sin(E_true)

    # Time the function
    init = time.time()
    E_out,iters = danby_solver(eccentricity,ell_input,iter_counter=True)#,N_it)
    runtime = time.time()-init

    print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e.\n"%(runtime*1000.,np.mean(np.abs(E_out-E_true))))
    print(iters,'iterations')