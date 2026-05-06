import numpy as np

m = np.zeros((7, 7))

np.fill_diagonal(m, 1)

m[np.arange(7), np.arange(6, -1, -1)] = 1

m[0, :] = 1
m[-1, :] = 1

m[:, 0] = 1
m[:, -1] = 1

print(m)