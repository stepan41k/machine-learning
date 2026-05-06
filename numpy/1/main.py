import numpy as np

a = np.arange(5, 51)

b = a[a % 5 == 0]

print(b)