"""
This script is an implementation of Nijenhuis' method for solving Kepler's equation.
An algorithm is provided, suggesting the separation of (M,e)-space into 4 regions,
where different initial guesses are to be used.
The initial guesses are refined through one application of Halley's method in regions A, B and C
and one application of Newton's method in region D.
Thereafter, a final step corrects the refined guess through one application of an m-th order approximation,
where m is given (I set the default to m=1).

The main difference between my implementation and Nijenhuis' is the approximation on sin(x).
For this function Nijenhuis uses Carlson and Goldstein's approximation, which I have also tested out.
It gave much worse results for the same order with essentially the same runtime, so I used NumPy's sin() and cos().
"""


### imports
import numpy as np

### Factorial function
def factorial(n):
	fact = 1
	for i in range(2,n+1):
		fact *= i
	return fact

### Nijenhuis' method
def nijenhuis_method(e,M,order=1):

	# regions
	lim1 = np.pi - 1 - e
	if e<0.5:
		lim2 = 1-e
	else:
		lim2 = 0.5
	rA = M>=lim1
	rB = (M<lim1)*(M>=lim2)
	rC = M<lim2 # if e > 0.5 region C is region D

	# rough starters and refinement through Halley's method in A, B and C and Newton-Raphson in D
	E_rgh = np.zeros(len(M),float)
	E_rgh[rA] = (M[rA]+np.pi*e)/(1+e)   # rough starter in region A
	E_rgh[rB] = M[rB] + e               # rough starter in region B
	E_ref = np.zeros(E_rgh.shape,float) # array to store refined values

	if e<0.5:
		# treat whole interval
		E_rgh[rC] = M[rC]/(1-e) # rough starter in region C
		esinx = e*np.sin(E_rgh)
		ecosx = e*np.cos(E_rgh)
		h0 = E_rgh - esinx - M
		h1 = 1 - ecosx
		h2 = esinx
		E_ref = E_rgh - h0*h1/(h1*h1 - 0.5*h0*h2)
	else:
		# treat regions A and B separately
		E_rghAB = E_rgh[rA|rB]
		esinx = e*np.sin(E_rghAB)
		ecosx = e*np.cos(E_rghAB)
		h0 = E_rghAB - esinx - M[rA|rB]
		h1 = 1 - ecosx
		h2 = esinx
		E_ref[rA|rB] = E_rghAB - h0*h1/(h1*h1 - 0.5*h0*h2)

		# treat region D
		denom = 4*e + 0.5 
		q = M[rC]/(2*denom)
		p = (1-e)/denom
		q2 = q*q
		p2 = p*p
		z = (np.sqrt(p*p2 + q2) + q)**(1/3)
		z2 = z*z
		s = 2*q/(z2 + p + p2/z2)
		s2 = s*s
		s3 = s*s2
		s5 = s2*s3
		# h0 = 0.075*s5 + (denom*s3-M[rC])/3 + (1-e)*s
		# h1 = 0.375*s2*s2 + denom*s2 + 1 - e
		h0 = (denom*s3-M[rC])/3 + (1-e)*s
		h1 = denom*s2 + 1 - e
		s -= h0/h1
		E_rgh[rC] = s
		E_ref[rC] = M[rC] + e*(3*s - 4*s3)

	# Final Step

	# Pre-compute e*sin(E) and e*cos(E)
	esinE = e*np.sin(E_ref)
	ecosE = e*np.cos(E_ref)

	# Arrays to store values in terms of order (1st index) and different M values (2nd index)
	func = np.empty([order+1,len(M)],float) # f and its m = order derivatives
	h = np.zeros(func.shape,float)          # h constants (0th order has no meaning)
	delta = np.zeros(func.shape,float)      # delta constants (0th order has no meaning)

	# Substitute function's and derivatives' values into array
	func[::4,:] = -esinE   # 0th, 4th, ... order derivatives
	func[1::4,:]= -ecosE   # 1st, 5th, ... order derivatives
	func[2::4,:]= esinE    # 2nd, 6th, ... order derivatives
	func[3::4,:]= ecosE    # 3rd, 7th, ... order derivatives
	func[0,:] += E_ref - M # add linear and constant terms to function
	func[1,:] += 1         # add linear term's 1st derivative

	delta[1,:] = func[1,:]         # 1st delta value
	h[1,:] = -func[0,:]/delta[1,:] # 1st h value

	# Compute h[i] from i=2 to i=order, using h[j] values from j=1 to j=i-1
	for i in range(2,order+1):
		for j in range(1,i):
			delta[i,:] = delta[i,:] + func[i-j+1,:]/factorial(i-j+1)
			delta[i,:] *= h[j,:]
		delta[i,:] += func[1,:]
		h[i,:] = -func[0,:]/delta[i,:].copy()
	return E_ref + h[-1,:]

### Nijenhuis' solver
def nijenhuis_solver(e,M,epsilon=1e-9,iter_order=False):
	x = M
	filt = e*np.abs(np.sin(x))>epsilon
	#filt = np.ones(len(M),bool)
	solutions = np.empty(len(M),float)
	m = 0
	while filt.any():
		m += 1 # raise order for values which have not reached epsilon
		array = nijenhuis_method(e,M[filt],order=m)
		solutions[filt] = array
		filt[filt] = np.abs(array - e*np.sin(array)-M[filt])>epsilon
	if iter_order:
		return solutions,m
	else:
		return solutions

### Test function
if __name__ == '__main__':

	import matplotlib.pyplot as plt,time
	# Parameters
	N = 1000000
	e = 0.5

	print("\n##### PARAMETERS #####")
	print("# len(M) = %d"%N)
	print("# e = %.2f"%e)
	print("######################")

	# Create ell array from E
	E_true = (np.pi*(np.arange(N)+0.5))/N
	M_input = E_true - e*np.sin(E_true)

	# Time the function
	init = time.time()
	E_out = nijenhuis_method(e,M_input)
	runtime = time.time()-init
	print("\nEstimation complete after %.1f millseconds, achieving mean error %.2e.\n"%(runtime*1000.,np.mean(np.abs(E_out-E_true))))

	# Map the region of different initial guess
	def mapper(e,M):
		# regions
		lim1 = np.pi - 1 - e
		if e>0.5:
			lim2 = 0.5
		else:
			lim2 = 1-e
		rA = M>lim1
		rB = (M<=lim1)*(M>lim2)
		rC = M<=lim2 # if e > 0.5 region C is region D
		a = np.zeros(len(M),float)
		a[rA] += 2
		a[rB] += 4
		a[rC] += 6
		if e>0.5:
			a[rC] += 2
		return a

	# Grid values for map
	N = 1000
	M = np.linspace(0,np.pi,N)
	e = np.arange(0,1,1/N)
	vals = np.zeros([len(e),len(M)])
	for i in range(N):
		vals[i,:] = mapper(e[i],M)

	# Plot map
	plt.imshow(vals,origin='lower',aspect=1/np.pi)
	plt.xticks(np.arange(0,N)[::200],labels=np.round(M[::200],decimals=2))
	plt.yticks(np.arange(0,N)[::200],labels=e[::200])
	plt.xlabel('$M$')
	plt.ylabel('$e$')
	plt.title('Regions with different initial guess \nand refinement method')
	plt.show()

	# Plot values calculated and compare with pre-calculated grid
	sympy_grid = np.loadtxt('sympy_200x200grid.txt')[1::30,1:-1]
	M = np.linspace(0,np.pi,200)[1:-1]
	e = np.arange(0,1,1/200)[1::30]
	for order in range(1,6):
		for i in range(len(e)):
			vals = nijenhuis_method(e[i],M,order)
			sympy_vals = sympy_grid[i,:]
			plt.plot(M,vals-sympy_vals,label='e = {}'.format(e[i]))
		plt.xlabel('$M$')
		plt.ylabel('$\Delta E$')
		plt.title('order: $m = {}$'.format(order))
		plt.legend()
		plt.grid()
		plt.show()

	import matplotlib
	matplotlib.rcParams["text.usetex"] = True
	for i in range(len(e)):
		start = time.time()
		vals,order = nijenhuis_solver(e[i],M,epsilon=1e-12,iter_order=True)
		stop = time.time()-start
		sympy_vals = sympy_grid[i,:]
		plt.plot(M,vals-sympy_vals,label='e = {} , t = {:.3e}s , order = {}'.format(e[i],stop,order))
	Mticks = np.linspace(0,np.pi,5)
	Mlabels= ['0','$\\frac{\pi}{4}$','$\\frac{\pi}{2}$','$\\frac{3\pi}{4}$','$\\pi$']
	plt.xticks(ticks=Mticks,labels=Mlabels)
	plt.xlabel('$M$')
	plt.ylabel('$\Delta E$')
	plt.legend()
	plt.show()