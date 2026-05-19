import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = range(1, 71)

def weight_q(distances, q=0.5):
    return q ** distances

def weight_inv_a(distances, a=0.1):
    return 1 / (distances + a)

def weight_inv_ab(distances, a=0.1, b=2):
    return 1 / (distances + a)**b

results = {
    'uniform': [],
    'q^d': [],
    '1/(d+a)': [],
    '1/(d+a)^b': []
}

for k in k_values:
    knn_u = KNeighborsClassifier(n_neighbors=k, weights='uniform').fit(X_train, y_train)
    results['uniform'].append(accuracy_score(y_test, knn_u.predict(X_test)))
    
    knn_q = KNeighborsClassifier(n_neighbors=k, weights=lambda d: weight_q(d, q=0.5)).fit(X_train, y_train)
    results['q^d'].append(accuracy_score(y_test, knn_q.predict(X_test)))
    
    knn_a = KNeighborsClassifier(n_neighbors=k, weights=lambda d: weight_inv_a(d, a=0.1)).fit(X_train, y_train)
    results['1/(d+a)'].append(accuracy_score(y_test, knn_a.predict(X_test)))

    knn_ab = KNeighborsClassifier(n_neighbors=k, weights=lambda d: weight_inv_ab(d, a=0.1, b=2)).fit(X_train, y_train)
    results['1/(d+a)^b'].append(accuracy_score(y_test, knn_ab.predict(X_test)))

plt.figure(figsize=(12, 7))
for label, accs in results.items():
    plt.plot(k_values, accs, label=label)

plt.title('Задание 2: Сравнение точности с разными весовыми функциями')
plt.xlabel('Количество соседей k')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()