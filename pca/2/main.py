import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

def my_pca_svd(X, n_components):
    """
    Собственная реализация PCA через сингулярное разложение (SVD).
    """
    # 1. Центрирование данных (вычитаем среднее по каждому признаку)
    X_centered = X - np.mean(X, axis=0)
    
    # 2. Применяем SVD: X = U * S * Vt
    # U - левые сингулярные векторы
    # s - сингулярные числа (диагональ матрицы S)
    # Vt - правые сингулярные векторы (транспонированные)
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # 3. Вычисляем проекцию данных на компоненты
    # Матрица весов (собственные векторы) - это первые n_components столбцов матрицы V (или строк Vt)
    # Формула проекции: X_reduced = X_centered @ V
    # Или через U и s: X_reduced = U[:, :n_components] * s[:n_components]
    X_reduced = X_centered @ Vt.T[:, :n_components]
    
    return X_reduced

# --- Тестирование и сравнение ---

# Загрузим данные для проверки
iris = load_iris()
X = iris.data
# Для PCA данные обязательно должны быть масштабированы
X_scaled = StandardScaler().fit_transform(X)

n_comp = 2

# Применяем нашу реализацию
my_res = my_pca_svd(X_scaled, n_comp)

# Применяем реализацию sklearn
sk_pca = PCA(n_components=n_comp)
sk_res = sk_pca.fit_transform(X_scaled)

print("--- Сравнение результатов ---")
print(f"Первые 3 строки (Моя реализация):\n{my_res[:3]}")
print(f"\nПервые 3 строки (Sklearn):\n{sk_res[:3]}")

# Проверка на идентичность (учитываем, что знаки могут быть инвертированы)
# PCA определяет направление осей, поэтому вектор [1, 2] и [-1, -2] — это одна и та же компонента
diff = np.abs(np.abs(my_res) - np.abs(sk_res)).max()
print(f"\nМаксимальная разница между модулями значений: {diff:.2e}")

if diff < 1e-10:
    print("Результаты идентичны!")
else:
    print("Есть существенные различия.")