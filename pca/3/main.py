import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

file_path = '../../content/prepared.csv'
if not os.path.exists(file_path):
    print("Загрузка файла...")
    os.system('gdown 1wonVHHMWLsLLE9sdKKTKPA-WfZTUUera')

df = pd.read_csv(file_path, index_col=0)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"Размерность данных: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled_df, y)

if lr.coef_.shape[0] > 1: # Если классов много
    lr_importances = np.mean(np.abs(lr.coef_), axis=0)
else:
    lr_importances = np.abs(lr.coef_[0])

lr_results = pd.Series(lr_importances, index=X.columns).sort_values(ascending=False)

# 3. Деревянная модель (Random Forest)
# Для деревьев масштабирование не критично, можно использовать исходный X
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

rf_results = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# 4. Вывод результатов
print("\n" + "="*40)
print("ТОП-5 признаков (Линейная модель - после масштабирования):")
print(lr_results.head(5))

print("\n" + "="*40)
print("ТОП-5 признаков (Random Forest):")
print(rf_results.head(5))

# Сравнение
top_lr = set(lr_results.head(5).index)
top_rf = set(rf_results.head(5).index)
common = top_lr.intersection(top_rf)

print("\n" + "="*40)
print(f"Общих признаков в обоих ТОП-5: {len(common)}")
if common:
    print(f"Список общих признаков: {', '.join(common)}")