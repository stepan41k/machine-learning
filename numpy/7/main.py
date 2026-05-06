import numpy as np

m = np.random.randn(5, 5)

print("Матрица до перестановки:")
print(m)

m[:, [0, -1]] = m[:, [-1, 0]]

print("\nМатрица после перестановки:")
print(m)