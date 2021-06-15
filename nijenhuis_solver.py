### imports
import numpy as np
import matplotlib.pyplot as plt

### Nijenhuis' method
def nijenhuis_solver(e,M,order=4):

	logpi = np.log10(np.pi)

	# regions
	lim1 = np.pi - 1 - e
	if e>logpi:
		lim2 = logpi
	else:
		lim2 = 1-e
	rA = M>=lim1
	rB = (M<lim1)*(M>=lim2)
	rC = M<lim2 # if e > 0.5 region C is region D

	# rough starters and refinement through Halley's method in A, B and C and Newton-Raphson in D
	E_rough = np.zeros(len(M),float)
	E_rough[rA] = (M[rA]+np.pi*e)/(1+e)
	E_rough[rB] = M[rB] + e
	E_ref = np.zeros(E_rough.shape,float)

	# Halley's method in regions A, B and C
	# sn(x) = x - a1*x^3 + a2*x^5
	a1 = 0.16605
	a2 = 0.00761

	if e<logpi:
		# treat whole interval
		E_rough[rC] = M[rC]/(1-e)
		E_rough2 = E_rough*E_rough
		E_rough3 = E_rough*E_rough2
		E_rough5 = E_rough2*E_rough3
		h0 = E_rough - e*(E_rough - a1*E_rough3 + a2*E_rough5) - M
		h1 = 1 - e*(1 - a1*3*E_rough2 - a2*5*E_rough2*E_rough2)
		h2 = e*(a1*6*E_rough - a2*20*E_rough3)
		E_ref = E_rough - h0/(h1 - 0.5*h0*h2/h1)
	else:
		# treat regions A and B
		E_roughAB = E_rough[rA|rB]
		E_roughAB2 = E_roughAB*E_roughAB
		E_roughAB3 = E_roughAB*E_roughAB2
		E_roughAB5 = E_roughAB2*E_roughAB3
		h0 = (1-e)*E_roughAB + e*(a1*E_roughAB3 - a2*E_roughAB5) - M[rA|rB]
		h1 = 1 - e + e*(a1*3*E_roughAB2 - a2*5*E_roughAB2*E_roughAB2)
		h2 = e*(a1*6*E_roughAB - a2*20*E_roughAB3)
		E_ref[rA|rB] = E_roughAB - h0/(h1 - 0.5*h0*h2/h1)

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
		E_ref[rC] = M[rC] + e*s*(3 - 4*s*s)


	# Final Step
	esinE = e*np.sin(E_ref)
	ecosE = e*np.cos(E_ref)

	func = np.empty([order+1,len(M)],float) # f and its derivatives
	h = np.zeros(func.shape,float)
	delta = np.zeros(func.shape,float)
	n = np.arange(0,order+1,1) # array: [1,2,...,n-1,n]

	func[::4,:] = -esinE 
	func[1::4,:]= -ecosE
	func[2::4,:]= esinE
	func[3::4,:]= ecosE

	func[0,:] += E_ref - M
	func[1,:] += 1

	delta[1,:] = func[1,:]
	h[1,:] = -func[0,:]/delta[1,:]

	for i in range(2,order):
		for j in range(i-1):
			delta[i,:] += func[i-j,:]/np.prod(n[:i-j])
			delta[i,:] *= h[j+1,:]
		h[i,:] = -func[0,:]/delta[i,:]
	E_ref += h[-1,:]

	return E_ref



if __name__ == '__main__':

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

	N = 1000
	M = np.linspace(0,np.pi,N)
	e = np.arange(0,1,1/N)



	vals = np.zeros([len(e),len(M)])

	for i in range(N):
		vals[i,:] = mapper(e[i],M)


	plt.imshow(vals,origin='lower',aspect=1/np.pi)
	plt.axvline(2*N/np.pi)
	plt.xticks(np.arange(0,N)[::200],labels=np.round(M[::200],decimals=2))
	plt.yticks(np.arange(0,N)[::200],labels=e[::200])
	plt.colorbar(shrink=1/np.pi)
	plt.show()

	sympy_grid = np.loadtxt('sympy_200x200grid.txt')
	M = np.linspace(0,np.pi,200)
	
	for e in np.arange(0,1,0.2):
		vals = nijenhuis_solver(e,M,10)
		sympy_vals = sympy_grid[int(e*200),:]
		ax = plt.plot(M,vals,label='e = {:.3}'.format(e))
		plt.plot(M,sympy_vals,'--',color=ax[0].get_color())
	plt.legend()
	plt.grid()
	plt.show()




