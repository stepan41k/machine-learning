import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# 1. Загрузка данных
iris = load_iris()
X, y = iris.data, iris.target

# Стандартизация (обязательна для PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Модель
model = LogisticRegression()

# 2. Оценка качества ДО PCA (на всех признаках)
cv_scores_before = cross_val_score(model, X_scaled, y, cv=5)
print(f"Средняя точность (до PCA): {cv_scores_before.mean():.4f}")

# 3. Применение PCA (сократим до 2 компонент для наглядности)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Оценка качества ПОСЛЕ PCA
cv_scores_after = cross_val_score(model, X_pca, y, cv=5)
print(f"Средняя точность (после PCA, 2 компоненты): {cv_scores_after.mean():.4f}")
print(f"Объясненная дисперсия (PCA): {pca.explained_variance_ratio_.sum():.4f}")