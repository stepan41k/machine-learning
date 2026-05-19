import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def my_pca_svd(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    X_reduced = X_centered @ Vt.T[:, :n_components]
    
    return X_reduced

iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

n_comp = 2

my_res = my_pca_svd(X_scaled, n_comp)

sk_pca = PCA(n_components=n_comp)
sk_res = sk_pca.fit_transform(X_scaled)

print("--- Сравнение результатов ---")
print(f"Первые 3 строки (Моя реализация):\n{my_res[:3]}")
print(f"\nПервые 3 строки (Sklearn):\n{sk_res[:3]}")

diff = np.abs(np.abs(my_res) - np.abs(sk_res)).max()
print(f"\nМаксимальная разница между модулями значений: {diff:.2e}")

if diff < 1e-10:
    print("Результаты идентичны!")
else:
    print("Есть существенные различия.")