import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# 1. Загрузка данных
df = pd.read_csv('../../content/prepared.csv', index_col=0)

# Удаляем лишнее. X - признаки, y - цель
X = df.drop(columns=['Id', 'Price', 'PriceOneRoom'], errors='ignore')
y = df['Price']

# 2. Быстрая проверка базовой модели (100 деревьев)
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
cv_score = cross_val_score(gbr, X, y, cv=3, scoring='r2', n_jobs=-1).mean()
print(f"Базовый R2: {cv_score:.4f}")

# 3. Ускоренный подбор параметров через RandomizedSearchCV
# Мы проверяем только 10 случайных комбинаций из сетки
param_distributions = {
    'n_estimators': [200, 300, 400], 
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1]
}

# n_iter=10 — это ограничение количества попыток для скорости
rand_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=5, 
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

rand_search.fit(X, y)

best_model = rand_search.best_estimator_
best_r2 = rand_search.best_score_

print(f"Лучшие параметры: {rand_search.best_params_}")
print(f"Итоговый R2: {best_r2:.4f}")

# 4. Важность признаков (считается мгновенно на лучшей модели)
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(10).plot(kind='bar', figsize=(10, 5), title="Топ-10 важных признаков")
plt.show()

if best_r2 > 0.73:
    print("Результат достигнут!")