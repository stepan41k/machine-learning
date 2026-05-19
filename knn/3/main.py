import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_neighbors': range(1, 50),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2, 3] 
}

grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1 
)

print("Начинаю поиск лучших параметров...")
grid_search.fit(X_train, y_train)

print("-" * 30)
print(f"Лучшие найденные параметры: {grid_search.best_params_}")
print(f"Лучшая точность (cross-val score): {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, test_predictions)

print(f"Итоговая точность на тестовых данных: {final_accuracy:.4f}")