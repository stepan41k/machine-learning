import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.utils.extmath import safe_sparse_dot

X, y, coef_real = make_regression(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_targets=1,
    noise=5,
    bias=0,
    coef=True,
    random_state=42
)

y = y.reshape(-1, 1)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def GD(X, y, alpha=0.01, n_iter=100):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    errors = []
    weights_history = [weights.flatten().copy()]

    for _ in range(n_iter):
        y_pred = safe_sparse_dot(X, weights)
        gradient = (2 / n_samples) * safe_sparse_dot(X.T, (y_pred - y))
        weights -= alpha * gradient

        errors.append(mse_loss(y, y_pred))
        weights_history.append(weights.flatten().copy())

    return weights, errors, np.array(weights_history)

def SGD(X, y, alpha=0.01, n_iter=100):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    errors = []
    weights_history = [weights.flatten().copy()]

    for i in range(n_iter):
        idx = np.random.randint(n_samples)
        x_i = X[idx:idx+1]
        y_i = y[idx:idx+1]

        y_pred_i = safe_sparse_dot(x_i, weights)
        gradient = 2 * safe_sparse_dot(x_i.T, (y_pred_i - y_i))
        weights -= alpha * gradient

        current_y_pred = safe_sparse_dot(X, weights)
        errors.append(mse_loss(y, current_y_pred))
        weights_history.append(weights.flatten().copy())

    return weights, errors, np.array(weights_history)

def mbGD(X, y, alpha=0.01, n_iter=100, batch_size=32):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    errors = []
    weights_history = [weights.flatten().copy()]

    for _ in range(n_iter):
        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        y_pred_batch = safe_sparse_dot(X_batch, weights)
        gradient = (2 / batch_size) * safe_sparse_dot(X_batch.T, (y_pred_batch - y_batch))
        weights -= alpha * gradient

        current_y_pred = safe_sparse_dot(X, weights)
        errors.append(mse_loss(y, current_y_pred))
        weights_history.append(weights.flatten().copy())

    return weights, errors, np.array(weights_history)

n_iter = 30
alpha = 0.05

w_gd, err_gd, hist_gd = GD(X, y, alpha, n_iter)
w_sgd, err_sgd, hist_sgd = SGD(X, y, alpha, n_iter)
w_mbgd, err_mbgd, hist_mbgd = mbGD(X, y, alpha, n_iter, batch_size=32)


plt.figure(figsize=(12, 6))
plt.plot(err_gd, label='Gradient Descent (GD)', linewidth=2)
plt.plot(err_sgd, label='Stochastic GD (SGD)', alpha=0.7)
plt.plot(err_mbgd, label='Mini-batch GD (mbGD)', alpha=0.8)
plt.title('Зависимость MSE от числа итераций')
plt.xlabel('Итерация')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(hist_gd[:, 0], label='GD w1')
ax1.plot(hist_sgd[:, 0], label='SGD w1', alpha=0.5)
ax1.plot(hist_mbgd[:, 0], label='mbGD w1', alpha=0.7)
ax1.axhline(y=coef_real[0], color='r', linestyle='--', label='True w1')
ax1.set_title('Изменение веса w1')
ax1.legend()

ax2.plot(hist_gd[:, 1], label='GD w2')
ax2.plot(hist_sgd[:, 1], label='SGD w2', alpha=0.5)
ax2.plot(hist_mbgd[:, 1], label='mbGD w2', alpha=0.7)
ax2.axhline(y=coef_real[1], color='r', linestyle='--', label='True w2')
ax2.set_title('Изменение веса w2')
ax2.legend()

plt.show()

print(f"Реальные веса: {coef_real}")
print(f"Итоговые веса GD: {w_gd.flatten()}")
print(f"Итоговые веса SGD: {w_sgd.flatten()}")
print(f"Итоговые веса mbGD: {w_mbgd.flatten()}")