import numpy as np
import pandas as pd

np.random.seed(42)
temps = np.random.uniform(-15, -5, 31)
dates = pd.date_range('2026-01-01', periods=31, freq='D')
january_temps = pd.Series(temps, index=dates)
df = pd.DataFrame(january_temps, columns=['temp_day'])
df['temp_night'] = df['temp_day'] * 1.5

df['temp_diff'] = (df['temp_day'] - df['temp_night']).abs()

print("Датафрейм с новым столбцом temp_diff:")
print(df.head())

diff_jan_7 = df.loc['2026-01-07', 'temp_diff']

print(f"\nРазница температур 7 января: {diff_jan_7:.4f}")