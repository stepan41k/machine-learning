import numpy as np

a = np.zeros((3, 3))

b = np.ones((2, 3))

result = np.vstack((a, b))

print("Результирующий массив:")
print(result)
print("\nОбщее количество элементов:", result.size)