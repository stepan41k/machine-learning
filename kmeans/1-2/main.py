import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons, make_blobs

# Реализация алгоритма K-means
def custom_kmeans(X, k, max_iter=300):
    # 1. Инициализация центроидов (случайные точки из данных)
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    
    for _ in range(max_iter):
        # 2. Определение кластеров (к какому центроиду ближе точка)
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # 3. Обновление центроидов (среднее арифметическое точек в кластере)
        new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 
                                  else centroids[i] for i in range(k)])
        
        # Если центроиды не изменились, выходим из цикла
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        
    return centroids, labels

def calculate_wcss(X, centroids, labels):
    wcss = 0
    for i in range(len(centroids)):
        # Квадрат евклидова расстояния от точек кластера до его центра
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i])**2)
    # Возвращаем среднее (делим на общее кол-во точек)
    return wcss / X.shape[0]

# Генерация данных для примера (возьмем blobs)
X_task1, _ = make_blobs(n_samples=500, centers=3, random_state=42)

# Сбор метрик для k от 1 до 10
qualities = []
k_range = range(1, 11)

for k in k_range:
    centroids, labels = custom_kmeans(X_task1, k)
    quality = calculate_wcss(X_task1, centroids, labels)
    qualities.append(quality)

# Визуализация
plt.figure(figsize=(8, 5))
plt.plot(k_range, qualities, marker='o')
plt.title('Зависимость качества кластеризации от количества кластеров')
plt.xlabel('Number of clusters')
plt.ylabel('Quality (Mean Squared Distance)')
plt.grid(True)
plt.show()


# 1. Генерация данных
X1, y1 = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)
X2, y2 = make_moons(n_samples=1000, noise=0.05, random_state=42)
X3, y3 = make_blobs(n_samples=1000, centers=3, cluster_std=[1.0, 5.0, 2.5], random_state=42)

datasets = [
    (X1, "Noisy Circles", 2),
    (X2, "Noisy Moons", 2),
    (X3, "Blobs", 3)
]

fig, axes = plt.subplots(3, 2, figsize=(12, 15))

for i, (X, title, k) in enumerate(datasets):
    # Кастомный K-means
    _, custom_labels = custom_kmeans(X, k)
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=custom_labels, s=10, cmap='viridis')
    axes[i, 0].set_title(f"Custom KMeans: {title}")
    
    # Sklearn K-means
    sk_kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    sk_labels = sk_kmeans.fit_predict(X)
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=sk_labels, s=10, cmap='viridis')
    axes[i, 1].set_title(f"Sklearn KMeans: {title}")

plt.tight_layout()
plt.show()