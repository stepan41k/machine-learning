import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [10, 5, 3, 2, 0, 9, 1, 0, 8, 2]]).T
y = np.array([45, 55, 50, 55, 65, 35, 75, 80, 50, 60]).reshape(-1, 1)

def gradient_descent(X, y, eta=0.01, n_iterations=1000, tol=None):
    n = len(y)
    w = np.zeros((X.shape[1], 1))
    mse_history = []

    for i in range(n_iterations):
        gradients = 2/n * X.T.dot(X.dot(w) - y)
        w_new = w - eta * gradients
        current_mse = mean_squared_error(y, X.dot(w_new))
        mse_history.append(current_mse)

        if tol is not None and i > 0:
            if abs(mse_history[-2] - mse_history[-1]) < tol:
                return w_new, i + 1, mse_history
        w = w_new
    return w, n_iterations, mse_history

# task 1
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X, y)
w_ols = lin_reg.coef_.flatten()

w_gd, _, _ = gradient_descent(X, y, eta=0.01, n_iterations=2000)

print("Задание 1:")
print(f"Веса МНК: {w_ols}")
print(f"Веса GD (eta=0.01): {w_gd.flatten()}")

# task 2
eta_2 = 1e-2
tol_2 = 1e-4
w_2, iters_needed, _ = gradient_descent(X, y, eta=eta_2, n_iterations=10000, tol=tol_2)

print("\nЗадание 2:")
print(f"При eta={eta_2} и tol={tol_2} алгоритм остановился на итерации: {iters_needed}")

# task 3
plt.figure(figsize=(10, 6))
etas = [1e-4, 1e-3, 1e-2, 3e-2]
for e in etas:
    _, _, history = gradient_descent(X, y, eta=e, n_iterations=500)
    plt.plot(range(len(history)), history, label=f'eta = {e}')

plt.xlabel('Число итераций')
plt.ylabel('MSE')
plt.title('Зависимость ошибки от числа итераций при разных eta')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()