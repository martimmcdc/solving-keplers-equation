if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt
	from build.cpp_solvers import *
	from time import time

	methods = [newton_solver,newton_solver,iterative_solver,goat_herd_solver,nijenhuis_solver,danby_solver,cordic_solver,murison_solver]

	e = np.arange(0,1,0.1)
	M = np.linspace(0,np.pi,1000)

	for method in methods:
		start = time()
		for e_val in e:
			solution = method(e_val,M)
			# print(solution)
			plt.plot(M,solution,label='e = '+str(e_val))
		t = time() - start
		plt.title(method.__name__+f" (in {round(t,4)}s)")
		plt.xlabel('M')
		plt.ylabel('E')
		plt.legend()
		plt.show()