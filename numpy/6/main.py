import numpy as np

matrix = np.array([
    [1, 3],
    [4, 5],
    [6, 1],
    [9, 7]
])

means = matrix.mean(axis=0)

centered_matrix = matrix - means

col1 = centered_matrix[:, 0]
col2 = centered_matrix[:, 1]

result = np.dot(col1, col2)

print("Исходная матрица:\n", matrix)
print("\nСредние значения столбцов:", means)
print("\nЦентрированная матрица:\n", centered_matrix)
print("\nСкалярное произведение:", result)