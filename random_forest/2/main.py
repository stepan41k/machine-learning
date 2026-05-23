import pandas as pd# Визуализация важности признаков (опционально)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('../../content/prepared.csv')

X = df.drop(columns=['Id', 'Price', 'PriceOneRoom'])
y = df['Price']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

final_model = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_valid)
r2_valid = r2_score(y_valid, y_pred)
print(f"Коэффициент детерминации R2 на валидации: {r2_valid:.4f}")

if r2_valid > 0.73:
    print("Условие R2 > 0.73 выполнено")
else:
    print("Нужно еще подкорректировать параметры")

importances = pd.Series(final_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nСамые важные признаки:")
print(importances.head(10))

importances.head(10).plot(kind='barh', figsize=(8, 5))
plt.title('Топ-10 важных признаков')
plt.show()