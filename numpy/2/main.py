import numpy as np

v = np.arange(12)
print("Исходный вектор:", v)
print()

shapes_2d = [(1, 12), (12, 1), (2, 6), (6, 2), (3, 4), (4, 3)]

print("Двумерные массивы:")
for i, shape in enumerate(shapes_2d, 1):
    arr_2d = v.reshape(shape)
    print(f"{i}) Форма {shape}:\n{arr_2d}\n")

shapes_3d = [(2, 2, 3), (3, 2, 2), (1, 3, 4)]

print("-" * 30)
print("Трехмерные массивы:")
for i, shape in enumerate(shapes_3d, 1):
    arr_3d = v.reshape(shape)
    print(f"{i}) Форма {shape}:\n{arr_3d}\n")