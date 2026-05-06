import numpy as np
import pandas as pd

np.random.seed(42)

temps = np.random.uniform(-15, -5, size=31)

dates = pd.date_range('2026-01-01', periods=31, freq='D')

january_temps = pd.Series(temps, index=dates)

print("Температурный ряд за январь 2026:")
print(january_temps)

average_temp = january_temps.mean()

print("\nСредняя температура за период:")
print(average_temp)