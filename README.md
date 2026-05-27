# Machine Learning Sandbox

Репозиторий содержит коллекцию примеров, упражнений и реализаций алгоритмов машинного обучения на языке Python. Проект охватывает основные концепции от обработки данных до глубокого обучения и продвинутых ансамблевых методов.

## Структура проекта

Репозиторий разделен на тематические директории, каждая из которых содержит практические примеры (обычно `main.py`):

- **[numpy](./numpy/)** — основы работы с библиотекой NumPy для научных вычислений.
- **[pandas](./pandas/)** — манипуляция и анализ данных с помощью Pandas.
- **[vizual](./vizual/)** — визуализация данных (Matplotlib, Seaborn).
- **[linear_regression](./linear_regression/)** — линейная регрессия и градиентный спуск.
- **[polynomial_regression](./polynomial_regression/)** — полиномиальная регрессия.
- **[logistic_regression](./logistic_regression/)** — логистическая регрессия.
- **[classification](./classification/)** — задачи классификации.
- **[decision_tree](./decision_tree/)** — решающие деревья.
- **[random_forest](./random_forest/)** — случайный лес.
- **[gradient_boosting](./gradient_boosting/)** — градиентный бустинг.
- **[knn](./knn/)** — метод k-ближайших соседей.
- **[kmeans](./kmeans/)** — кластеризация методом k-средних.
- **[pca](./pca/)** — метод главных компонент (снижение размерности).
- **[regularization](./regularization/)** — методы регуляризации (L1, L2).
- **[sgd](./sgd/)** — стохастический градиентный спуск.
- **[content](./content/)** — наборы данных (CSV файлы) для обучения моделей.

## Технологический стек

- **Язык:** Python 3.x
- **Библиотеки:**
  - `numpy` — работа с массивами и линейная алгебра.
  - `pandas` — обработка табличных данных.
  - `matplotlib`, `seaborn` — построение графиков и диаграмм.
  - `scikit-learn` — реализация алгоритмов машинного обучения и метрик.

## Как использовать

1. Убедитесь, что у вас установлен Python.
2. Рекомендуется использовать виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # На Windows: .venv\Scripts\activate
   ```
3. Установите необходимые зависимости:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
4. Запустите любой интересующий скрипт:
   ```bash
   python linear_regression/1-4/main.py
   ```

## Данные
Используемые наборы данных находятся в папке `content/`:
- `IMDB-Movie-Data.csv`
- `salaries.csv`
- `prepared.csv`
