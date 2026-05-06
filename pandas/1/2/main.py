import numpy as np
import pandas as pd

np.random.seed(42)
temps = np.random.uniform(-15, -5, 31)
dates = pd.date_range('2026-01-01', periods=31, freq='D')
january_temps = pd.Series(temps, index=dates)

df = pd.DataFrame(january_temps, columns=['temp_day'])

df['temp_night'] = df['temp_day'] * 1.5

print("Первые 5 строк датафрейма:")
print(df.head())

print("\nВесь датафрейм:")
print(df.info())