import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# task 1
def logloss(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    err = np.mean(- y * np.log(p) - (1.0 - y) * np.log(1.0 - p))
    return err

# task 3
def calc_accuracy(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return np.mean(y_true == y_pred)

# task 2,4
class LogReg:
    def __init__(self, eta=0.01, iterations=1000):
        self.eta = eta
        self.iterations = iterations
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.w = np.zeros(X.shape[1])
        self.loss_history = []

        for i in range(self.iterations):
            z = np.dot(X, self.w)
            p = self.sigmoid(z)

            gradient = np.dot(X.T, (p - y)) / len(y)
            self.w -= self.eta * gradient

            err = logloss(y, p)
            self.loss_history.append(err)

    def predict_proba(self, X):
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return self.sigmoid(np.dot(X, self.w))

# task 2
X_simple, y_simple = make_classification(n_samples=500, n_features=2, n_redundant=0, random_state=42)

etas = [0.5, 0.1, 0.01]
plt.figure(figsize=(10, 6))

for eta in etas:
    model = LogReg(eta=eta, iterations=500)
    model.fit(X_simple, y_simple)
    plt.plot(range(len(model.loss_history)), model.loss_history, label=f'eta = {eta}')

plt.title('Зависимость LogLoss от числа итераций')
plt.xlabel('Итерации')
plt.ylabel('LogLoss')
plt.legend()
plt.grid(True)
plt.show()

# task 4
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

custom_model = LogReg(eta=0.1, iterations=1000)
custom_model.fit(X_train_scaled, y_train)
custom_probs = custom_model.predict_proba(X_test_scaled)
custom_acc = calc_accuracy(y_test, custom_probs)

sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
sgd_model.fit(X_train_scaled, y_train)
sgd_probs = sgd_model.predict_proba(X_test_scaled)[:, 1]
sgd_acc = calc_accuracy(y_test, sgd_probs)

print(f"Результаты сравнения:")
print(f"Accuracy (Собственная LogReg): {custom_acc:.4f}")
print(f"Accuracy (Sklearn SGDClassifier): {sgd_acc:.4f}")