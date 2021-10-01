"""
This script is an implementation of M. Zechmeister's CORDIC algorithm for solving Kepler's equation.
It is slightly different than that available on the author's GitHub, since it works for mean anomaly arrays,
rather than scalar values. I used the NumPy library to improve performance, given this difference.
"""

### imports
import numpy as np

### CORDIC solver
def cordic_solver(e,M,n=29):
	pi2 = 2*np.pi # constant
	a = pi2/2**(1+np.linspace(1,n,n)) # angle: a
	acs = np.array([a,np.cos(a),np.sin(a)],float) # a,cos(a),sin(a)
	E = pi2*np.floor(M/pi2+0.5) # initial values for E
	cosE,sinE = 1.,0. # initial values for cos(E) and sin(E)
	for i in range(n):
		sigma = np.ones(len(M),float)
		sigma[E>M+e*sinE] = -1 # sigma (see algorithm)
		a,c,s = acs[:,i]
		s *= sigma
		E += a*sigma
		cosE,sinE = cosE*c-sinE*s,cosE*s+sinE*c
	return E


if __name__=="__main__":
    """Test the Python function above with a simple example"""

    import matplotlib.pyplot as plt,time
    # Parameters
    N = 1000000
    e = 0.5

    print("\n##### PARAMETERS #####")
    print("# N_M = %d"%N)
    print("# Eccentricity = %.2f"%e)
    print("######################")

    # Create M array from E
    E_true = (2.0*np.pi*(np.arange(N)+0.5))/N
    M_input = E_true - e*np.sin(E_true)

    # Time the function
    init = time.time()
    E_out = cordic_solver(e,M_input)
    runtime = time.time()-init

    print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e."%(runtime*1000.,np.mean(np.abs(E_out-E_true))))

    # Plot values calculated and compare with pre-calculated grid
    sympy_grid = np.loadtxt('sympy_200x200grid.txt')[::30,:]
    M = np.linspace(0,np.pi,200)
    e = np.arange(0,1,1/200)[::30]
    for ni in range(5,30,5):
        for i in range(len(e)):
            vals = cordic_solver(e[i],M,n=ni)
            sympy_vals = sympy_grid[i,:]
            plt.plot(M,vals-sympy_vals,label='e = {}'.format(e[i]))
        plt.xlabel('$M$')
        plt.ylabel('$\Delta E$')
        plt.title('$n = {}$'.format(ni))
        plt.legend()
        plt.grid()
        plt.show()