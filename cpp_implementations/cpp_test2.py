if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt

	values = np.loadtxt('val_list.txt')
	esinE = values[::2]
	ecosE = values[1::2]
	plt.plot(esinE)
	plt.plot(ecosE)
	plt.show()

	values = np.loadtxt('val_list2.txt')
	esinE = values[::2]
	ecosE = values[1::2]
	plt.plot(esinE)
	plt.plot(ecosE)
	plt.show()