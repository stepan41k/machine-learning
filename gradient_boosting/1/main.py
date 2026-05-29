from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=2,
    flip_y=0.1,
    random_state=1,
)

params = {"n_estimators": [100, 200], "max_depth": [3, 4], "learning_rate": [0.1]}

boost = GradientBoostingClassifier(random_state=42)

grid = GridSearchCV(boost, params, scoring="accuracy", cv=3, n_jobs=-1)
grid.fit(X, y)

print(f"Лучшие параметры: {grid.best_params_}")
print(f"Точность (accuracy): {grid.best_score_:.4f}")
