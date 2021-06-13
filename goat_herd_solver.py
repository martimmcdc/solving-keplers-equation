### imports
import numpy as np,time

def goat_herd_solver(e,M,N=10):
	x0 = M + 0.5*e # contour center
	dx = 0.5*e     # contour radius
	x = np.linspace(0,0.5,N) # integration variable array

	### Uncomment this block to allow for both scalar and array values of M and comment the 2 lines below it
	# Determine whether given M is array or scalar
	# if hasattr(M,"__len__"):
	# 	x0[M>np.pi] -= e    # (x0=M-e/2 , M < pi) and (x0=M+e/2 , M >= pi)
	# 	x = x[:,np.newaxis] # (axis = 0 for integration variable) and (axis = 1 for M dependence)
	###
	
	x0[M>np.pi] -= e    # (x0=M-e/2 , M < pi) and (x0=M+e/2 , M >= pi)
	x = x[:,np.newaxis] # (axis = 0 for integration variable) and (axis = 1 for M dependence)

	# Pre calculation of trigonometric and hyperbolic functions
	arg = 2*np.pi*x
	cosx0 = np.cos(x0)
	sinx0 = np.sin(x0)
	cos2pix = np.cos(arg) # real part of exp(-2i.pi.k.x) , k = -1
	sin2pix = np.sin(arg) # imag part of exp(-2i.pi.k.x) , k = -1
	cos4pix = cos2pix*cos2pix - sin2pix*sin2pix # real part of exp(-2i.pi.k.x) , k = -2
	sin4pix = 2*sin2pix*cos2pix                 # imag part of exp(-2i.pi.k.x) , k = -2
	dxcos = dx*cos2pix # argument of trigonometric functions that appear in expansion of e*sin(x) below 
	dxsin = dx*sin2pix # argument of hyperbolic functions that appear in expansion of e*sin(x) below
	ecosdx = e*np.cos(dxcos)
	esindx = e*np.sin(dxcos)
	coshdx = np.cosh(dxsin)
	sinhdx = np.sinh(dxsin)

	# f(x) = x - e*sin(x) - M
	fR = x0 + dxcos - (sinx0*ecosdx + cosx0*esindx)*coshdx - M # real part of f(x)
	fI = dxsin - (cosx0*ecosdx - sinx0*esindx)*sinhdx     # imag part of f(x)
	
	# Integrand functions up to constant which disappears in division
	# Re[d(ak)/dx] = Re[a(x)*exp(-2i.pi.k.x)], where a(x) = 1/f(x) = 1/(fR + ifI)
	# such that: aR = fR/(fR^2 + fI^2) , aI = -fI/(fR^2 + fI^2)
	a_1sub = fR*cos2pix + fI*sin2pix
	a_2sub = fR*cos4pix + fI*sin4pix
	abs_f = (fR*fR + fI*fI)
	a_1sub /= abs_f
	a_2sub /= abs_f

	# Trapezoidal integral up to a constant which disappears in division
	a_1 = 0.5*(a_1sub[0] + a_1sub[-1]) + np.sum(a_1sub[1:-1],axis=0)
	a_2 = 0.5*(a_2sub[0] + a_2sub[-1]) + np.sum(a_2sub[1:-1],axis=0)
	return x0 + dx*a_2/a_1


if __name__=="__main__":
    """Test the Python function above with a simple example"""

    # Parameters
    N_ell = 1000000
    eccentricity = 0.5
    N_it = 10

    print("\n##### PARAMETERS #####")
    print("# N_ell = %d"%N_ell)
    print("# Eccentricity = %.2f"%eccentricity)
    print("# Iterations: %d"%N_it)
    print("######################")

    # Create ell array from E
    E_true = (2.0*np.pi*(np.arange(N_ell)+0.5))/N_ell
    ell_input = E_true - eccentricity*np.sin(E_true)

    # Time the function
    init = time.time()
    E_out = goat_herd_solver(eccentricity,ell_input,N_it)
    runtime = time.time()-init

    print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e.\n"%(runtime*1000.,np.mean(np.abs(E_out-E_true))))











