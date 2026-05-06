import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Загрузка данных
iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target

# 2. Разбиение на обучающую и валидационную выборки
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

# 3. Обучение логистической регрессии
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Получение предсказаний
y_pred = model.predict(X_valid)

# 5. Вывод текстовых метрик
print(f"Общая точность (Accuracy): {accuracy_score(y_valid, y_pred):.4f}")
print("\nОтчет о классификации:")
print(classification_report(y_valid, y_pred, target_names=iris_dataset.target_names))

# 6. Визуализация матрицы ошибок
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris_dataset.target_names)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='viridis')
ax.set_title('Матрица ошибок для Iris')
plt.show()

# ВЫВОД:
# Модель логистической регрессии успешно обучена на датасете Iris. 
# Высокая точность (Accuracy) и показатели Precision/Recall подтверждают, 
# что признаки (длина и ширина лепестков/чашелистиков) позволяют 
# четко разделять виды ирисов между собой.