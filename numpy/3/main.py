import numpy as np

matrix = np.zeros((5, 5))

vals = np.arange(1, 6)

np.fill_diagonal(matrix, vals)

matrix[0, :] = vals

matrix[:, 0] = vals

print(matrix)