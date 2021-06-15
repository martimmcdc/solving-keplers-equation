### imports
import numpy as np
import matplotlib.pyplot as plt
from sympy_solver import sympy_solver


def factorial(n):
	fact = 1
	for i in range(2,n+1):
		fact *= i
	return fact

### Nijenhuis' method
def nijenhuis_solver(e,M,order=4):

	# regions
	lim1 = np.pi - 1 - e
	if e>0.5:
		lim2 = 0.5
	else:
		lim2 = 1-e
	rA = M>=lim1
	rB = (M<lim1)*(M>=lim2)
	rC = M<lim2 # if e > 0.5 region C is region D

	# rough starters and refinement through Halley's method in A, B and C and Newton-Raphson in D
	E_rough = np.zeros(len(M),float)
	E_rough[rA] = (M[rA]+np.pi*e)/(1+e)   # rough starter in region A
	E_rough[rB] = M[rB] + e               # rough starter in region B
	E_ref = np.zeros(E_rough.shape,float) # array to store refined values

	# Halley's method in regions A, B and C: sin(x) -> sn(x) = x - a1*x^3 + a2*x^5
	a1,a2 = 0.16605,0.00761

	if e<0.5:
		# treat whole interval
		E_rough[rC] = M[rC]/(1-e) # rough starter in region C
		E_rough2 = E_rough*E_rough
		E_rough3 = E_rough*E_rough2
		E_rough5 = E_rough2*E_rough3
		h0 = E_rough - e*(E_rough - a1*E_rough3 + a2*E_rough5) - M
		h1 = 1 - e*(1 - a1*3*E_rough2 - a2*5*E_rough2*E_rough2)
		h2 = e*(a1*6*E_rough - a2*20*E_rough3)
		E_ref = E_rough - h0*h1/(h1*h1 - 0.5*h0*h2)
	else:
		# treat regions A and B separately
		E_roughAB = E_rough[rA|rB]
		E_roughAB2 = E_roughAB*E_roughAB
		E_roughAB3 = E_roughAB*E_roughAB2
		E_roughAB5 = E_roughAB2*E_roughAB3
		h0 = E_roughAB - e*(E_roughAB - a1*E_roughAB3 + a2*E_roughAB5) - M[rA|rB]
		h1 = 1 - e*(1 - a1*3*E_roughAB2 + a2*5*E_roughAB2*E_roughAB2)
		h2 = e*(a1*6*E_roughAB - a2*20*E_roughAB3)
		E_ref[rA|rB] = E_roughAB - h0*h1/(h1*h1 - 0.5*h0*h2)

		# treat region D
		denom = 4*e + 0.5 
		q = 0.5*M[rC]/denom
		p = (1-e)/denom
		q2 = q*q
		p2 = p*p
		p3 = p*p2
		z2 = (np.sqrt(p3 + q2) + q)**(1/3)
		z2 *= z2
		s = 2*q/(z2 + p + p2/z2)
		s2 = s*s
		s3 = s*s2
		s5 = s2*s3
		h0 = 0.075*s5 + (denom*s3-M[rC])/3 + (1-e)*s
		h1 = 0.375*s2*s2 + denom*s2 + 1 - e
		s -= h0/h1
		E_rough[rC] = s
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
			delta[i,:] += func[i-j+1,:]/factorial(i-j+1)
			delta[i,:] *= h[j,:]
		delta[i,:] += func[1,:]
		h[i,:] = -func[0,:]/delta[i,:].copy()
	return E_ref + h[-1,:]



### Test function
if __name__ == '__main__':

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
	sympy_grid = np.loadtxt('sympy_200x200grid.txt')[::30]
	M = np.linspace(0,np.pi,200)
	e = np.arange(0,1,1/200)[::30]
	for order in range(1,6):
		for i in range(len(e)):
			vals = nijenhuis_solver(e[i],M,order)
			sympy_vals = sympy_grid[i,:]
			plt.plot(M,vals-sympy_vals,label='e = {}'.format(e[i]))
		plt.xlabel('$M$')
		plt.ylabel('$\Delta E$')
		plt.title('order: $m = {}$'.format(order))
		plt.legend()
		plt.grid()
		plt.show()




