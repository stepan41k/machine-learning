import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('../../content/salaries.csv')
x = df['Level'].values
y = df['Salary'].values

X = x.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Level'], y=df['Salary'], label='Данные', color='blue')
plt.plot(X, lin_reg.predict(X), color='red', label='Линейная регрессия')
plt.title('Линейная регрессия: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

y_pred_lin = lin_reg.predict(X)
r2_lin = r2_score(y, y_pred_lin)
print(f"R2 score для линейной регрессии: {r2_lin:.4f}")

degrees = np.arange(1, 10)
r2_scores = []

for d in degrees:
    poly_features = PolynomialFeatures(degree=d)
    X_poly = poly_features.fit_transform(X)

    poly_reg = LinearRegression()
    poly_reg.fit(X_poly, y)

    y_poly_pred = poly_reg.predict(X_poly)
    score = r2_score(y, y_poly_pred)
    r2_scores.append(score)

plt.figure(figsize=(10, 5))
plt.plot(degrees, r2_scores, marker='o', linestyle='-', color='green')
plt.title('Зависимость R2 score от степени полинома')
plt.xlabel('Степень полинома')
plt.ylabel('R2 Score')
plt.xticks(degrees)
plt.grid(True)
plt.show()

for d, s in zip(degrees, r2_scores):
    print(f"Степень {d}: R2 = {s:.4f}")

best_degree = 4
print(f"\nРекомендуемая максимальная степень: {best_degree}")

poly_features = PolynomialFeatures(degree=best_degree)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression().fit(X_poly, y)

X_grid = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Level'], y=df['Salary'], label='Данные')
plt.plot(X_grid, poly_reg.predict(poly_features.fit_transform(X_grid)), color='orange', label=f'Полином {best_degree} степени')
plt.title(f'Сравнение: Степень {best_degree} vs Линейная')
plt.plot(X, lin_reg.predict(X), color='red', linestyle='--', label='Линейная')
plt.legend()
plt.show()