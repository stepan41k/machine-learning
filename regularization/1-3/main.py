import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme(style="whitegrid")

X, y, coef_real = make_regression(n_samples=1000,
                                  n_features=2,
                                  n_informative=2,
                                  n_targets=1,
                                  noise=10,
                                  bias=0,
                                  coef=True,
                                  random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

degree = 10
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"Количество признаков после полиномизации (степень {degree}): {X_train_poly.shape[1]}")

lr = LinearRegression()
lr.fit(X_train_poly, y_train)
print(f"Linear Regression R2 (Train): {lr.score(X_train_poly, y_train):.4f}")
print(f"Linear Regression R2 (Test): {lr.score(X_test_poly, y_test):.4f}")

alphas = np.logspace(-2, 3, 20)
ridge_coefs = []
lasso_coefs = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_poly, y_train)
    ridge_coefs.append(ridge.coef_)

    lasso = Lasso(alpha=a, max_iter=100000, tol=0.1)
    lasso.fit(X_train_poly, y_train)
    lasso_coefs.append(lasso.coef_)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].plot(alphas, ridge_coefs)
ax[0].set_xscale('log')
ax[0].set_title('Влияние L2 (Ridge) на величину весов')
ax[0].set_xlabel('Alpha (сила регуляризации)')
ax[0].set_ylabel('Значения коэффициентов')

ax[1].plot(alphas, lasso_coefs)
ax[1].set_xscale('log')
ax[1].set_title('Влияние L1 (Lasso) на величину весов')
ax[1].set_xlabel('Alpha (сила регуляризации)')
ax[1].set_ylabel('Значения коэффициентов')

plt.tight_layout()
plt.show()

param_grid = {'alpha': np.logspace(-2, 2, 50)}

ridge_grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
ridge_grid.fit(X_train_poly, y_train)

lasso_grid = GridSearchCV(
    Lasso(max_iter=100000, tol=0.1),
    param_grid,
    cv=5,
    scoring='r2'
)
lasso_grid.fit(X_train_poly, y_train)

best_ridge_alpha = ridge_grid.best_params_['alpha']
best_lasso_alpha = lasso_grid.best_params_['alpha']

print(f"\nОптимальный alpha для Ridge: {best_ridge_alpha:.4f}")
print(f"Accuracy (R2) Ridge на тесте: {ridge_grid.best_estimator_.score(X_test_poly, y_test):.4f}")

print(f"Оптимальный alpha для Lasso: {best_lasso_alpha:.4f}")
print(f"Accuracy (R2) Lasso на тесте: {lasso_grid.best_estimator_.score(X_test_poly, y_test):.4f}")