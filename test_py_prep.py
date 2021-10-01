"""
This script generates 200x200 grids in (e,M)-space containing for each method the calculated root and runtime.
The values are saved as .txt files containing the arrays in the solver_grids/ folder.
They are accessed and used by test4.py .
"""

### imports ###
import numpy as np
from timeit import timeit
from py_implementations.newton_solver import newton_solver
from py_implementations.iterative_solver import iterative_solver
from py_implementations.goat_herd_solver import goat_herd_solver
from py_implementations.nijenhuis_solver import nijenhuis_solver
from py_implementations.danby_solver import danby_solver
from py_implementations.cordic_solver import cordic_solver
from py_implementations.murison_solver import murison_solver

directory = 'py_implementations'
methods = [newton_solver,iterative_solver,goat_herd_solver,nijenhuis_solver,danby_solver,cordic_solver,murison_solver]

### Grid values ###
N = 200                    # number of grid points for e and M
M = np.linspace(0,np.pi,N) # mean anomaly values in [0,pi]
e = np.arange(0,1,1/N)     # eccentricity values in [0,1)

### Calculation ###
Nt = 5
zeros = np.zeros([len(methods),len(e),len(M)],float)  # array to store the roots
times = np.zeros(zeros.shape,float)        # array to store the run time
M_array = M # M input as array without 0 and pi values
for i in range(len(methods)):
	method = methods[i]
	name = method.__name__ # method name to feed timeit function
	set_up = "import numpy as np;from {}.{} import {}".format(directory,name,name) # setup to be subtracted from runtime estimate
	for k in range(1,len(M_array)-1):
		Mk = np.array([M_array[k]])
		for j in range(1,len(e)-1):
			ej = e[j]
			statement = "{}({},np.{})".format(name,ej,repr(Mk)) # statement to be timed
			times[i,j,k] = timeit(stmt=statement,setup=set_up,number=Nt) # runtime for 10 calculations
			zeros[i,j,k] = method(ej,Mk) # values of iteration number
	print(name+' done')
	fname = 'py_solver_grids/'+name.strip('solver')+'200x200grid_'
	np.savetxt(fname+'runtime.txt',times[i,:,:])
	np.savetxt(fname+'zeros.txt',zeros[i,:,:])