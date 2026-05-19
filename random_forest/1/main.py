import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('../../content/prepared.csv'

X = df.drop(columns=['Id', 'Price', 'PriceOneRoom'])
y = df['Price']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

n_trees = [1, 5, 10, 20, 50, 100, 250, 500, 1000]
train_scores = []
valid_scores = []

for n in n_trees:
    rf = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    train_scores.append(r2_score(y_train, rf.predict(X_train)))
    valid_scores.append(r2_score(y_valid, rf.predict(X_valid)))

    print(f"Готово для n_estimators = {n}")

plt.figure(figsize=(10, 6))
plt.plot(n_trees, train_scores, label='Обучающая выборка', marker='o')
plt.plot(n_trees, valid_scores, label='Отложенная выборка', marker='o')
plt.xlabel('Количество деревьев (n_estimators)')
plt.ylabel('Точность (R2 score)')
plt.title('Зависимость точности RF от количества деревьев')
plt.legend()
plt.grid(True)
plt.show()