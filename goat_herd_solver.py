### imports
import numpy as np

def goat_herd_solver(e,M,N=5):
	x0 = M + e/2
	dx = e/2
	x = np.linspace(0,0.5,N)[1:-1]
	cosx0 = np.cos(x0)
	sinx0 = np.sin(x0)
	cos2pix = np.cos(2*np.pi*x)
	sin2pix = np.sin(2*np.pi*x)
	cos4pix = cos2pix*cos2pix - sin2pix*sin2pix
	sin4pix = 2*sin2pix*cos2pix
	gammaR = x0 + dx*cos2pix
	gammaI = dx*sin2pix
	fR = gammaR - e*(sinx0*np.cos(dx*cos2pix) + cosx0*np.sin(dx*cos2pix))*np.cosh(dx*sin2pix) - M
	fI = gammaI - e*(cosx0*np.cos(dx*cos2pix) - sinx0*np.sin(dx*cos2pix))*np.sinh(dx*sin2pix)
	aR = fR#/(fR**2 + fI**2)
	aI = -fI#/(fR**2 + fI**2)
	a_1sub = aR*cos2pix + aI*sin2pix
	a_2sub = aR*cos4pix + aI*sin4pix
	a_1 = a_1sub[0] + a_1sub[-1] + 2*np.sum(a_1sub[1:-1])
	a_2 = a_2sub[0] + a_2sub[-1] + 2*np.sum(a_2sub[1:-1])
	return x0 + dx*a_2/a_1