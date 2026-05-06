import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Загрузка датасета (предполагается, что файл в той же папке)
df = pd.read_csv('../../content/prepared.csv') # Если нужно загрузить заново

# 2.1 Выделение матрицы признаков X и целевой переменной y
# Удаляем 'Id', 'Price' и 'PriceOneRoom', как указано в задании
X = df.drop(columns=['Id', 'Price', 'PriceOneRoom'])
y = df['Price']

# 2.2 Разбиение на train и valid (возьмем стандартное соотношение 70/30)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

# 2.3 - 2.6 Обучение модели с подбором гиперпараметров
# Опытным путем для данного датасета: n_estimators=100 и max_depth=12-15 дают отличный результат
final_model = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

# Проверка точности на валидации
y_pred = final_model.predict(X_valid)
r2_valid = r2_score(y_valid, y_pred)
print(f"Коэффициент детерминации R2 на валидации: {r2_valid:.4f}")

# Проверка условия задания
if r2_valid > 0.73:
    print("Условие R2 > 0.73 выполнено!")
else:
    print("Нужно еще подкорректировать параметры.")

# 2.5 Поиск самых важных признаков
importances = pd.Series(final_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nСамые важные признаки:")
print(importances.head(10))

# Визуализация важности признаков (опционально)
importances.head(10).plot(kind='barh', figsize=(8, 5))
plt.title('Топ-10 важных признаков')
plt.show()