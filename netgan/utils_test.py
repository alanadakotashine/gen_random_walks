import scipy.sparse as sp
import numpy as np

def con_coo():
	row  = np.array([0, 0, 1, 3, 1, 0, 0])
	col  = np.array([0, 2, 1, 3, 1, 0, 0])
	data = np.array([1, 1, 1, 1, 1, 1, 1])
	print(type(row))
	print(type(col))
	print(type(data))
	coo = sp.coo_matrix((data, (row, col)), shape=(4, 4))