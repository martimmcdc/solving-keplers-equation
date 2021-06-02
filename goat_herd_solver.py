### imports
import numpy as np

def goat_herd_solver(e,M,N=10):
	x0 = M + e/2 # contour center
	dx = e/2     # contour radius
	x = np.linspace(0,0.5,N) # integration variable array

	# Pre calculation of trigonometric functions
	cosx0 = np.cos(x0)
	sinx0 = np.sin(x0)
	cos2pix = np.cos(2*np.pi*x) # real part of exp(-2i.pi.k.x) , k = -1
	sin2pix = np.sin(2*np.pi*x) # imag part of exp(-2i.pi.k.x) , k = -1
	cos4pix = cos2pix*cos2pix - sin2pix*sin2pix # real part of exp(-2i.pi.k.x) , k = -2
	sin4pix = 2*sin2pix*cos2pix                 # imag part of exp(-2i.pi.k.x) , k = -2

	# gamma(x) = x0 + dx * exp(2i.pi.x)
	gammaR = x0 + dx*cos2pix # real part of gamma(x)
	gammaI = dx*sin2pix      # imag part of gamma(x)

	# f(x) = x - e*sin(x) - M
	fR = gammaR - e*(sinx0*np.cos(gammaR - x0) + cosx0*np.sin(gammaR - x0))*np.cosh(gammaI) - M # real part of f(x)
	fI = gammaI - e*(cosx0*np.cos(gammaR - x0) - sinx0*np.sin(gammaR - x0))*np.sinh(gammaI)     # imag part of f(x)
	
	# Integrand functions up to constant which disappears in division
	# Re(d(ak)/dx) = a(x)*exp(-2i.pi.k.x), where a(x) = 1/f(x)
	# such that: aR = fR/(fR^2 + fI^2) , aI = -fI/(fR^2 + fI^2)
	aR = fR/(fR*fR + fI*fI)
	aI = -fI/(fR*fR + fI*fI)
	a_1sub = aR*cos2pix + aI*sin2pix
	a_2sub = aR*cos4pix + aI*sin4pix

	# Trapezoidal integral up to a constant which disappears in division
	a_1 = a_1sub[0] + a_1sub[-1] + 2*np.sum(a_1sub[1:-1])
	a_2 = a_2sub[0] + a_2sub[-1] + 2*np.sum(a_2sub[1:-1])
	return x0 + dx*a_2/a_1