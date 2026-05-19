import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score

seed = 1
X, y = datasets.make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=2,
    random_state=seed
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, s=50, edgecolor='black')
plt.xlabel('Признак x1')
plt.ylabel('Признак x2')
plt.title('Визуализация первых двух признаков')
plt.show()

print("\n")

clf_base = DecisionTreeClassifier(random_state=seed)
clf_base.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf_base, max_depth=3, filled=True, feature_names=[f'f{i}' for i in range(20)])
plt.title("Верхняя часть структуры дерева")
plt.show()

depth = clf_base.get_depth()
leaves = clf_base.get_n_leaves()

print(f"Глубина дерева (уровней): {depth}")
print(f"Количество листьев: {leaves}")

train_probs = clf_base.predict_proba(X_train)[:, 1]
test_probs = clf_base.predict_proba(X_test)[:, 1]

roc_auc_train = roc_auc_score(y_train, train_probs)
roc_auc_test = roc_auc_score(y_test, test_probs)

print(f"ROC AUC на обучающей выборке: {roc_auc_train:.3f}")
print(f"ROC AUC на тестовой выборке: {roc_auc_test:.3f}")

if roc_auc_train > 0.99 and (roc_auc_train - roc_auc_test) > 0.1:
    print("Наблюдается явное переобучение: точность на обучении идеальна, а на тесте заметно ниже.")
elif roc_auc_train < 0.7:
    print("Наблюдается недообучение.")
else:
    print("Модель работает стабильно.")

clf_tuned = DecisionTreeClassifier(
    max_depth=6,             
    min_samples_leaf=15,      
    max_features=15,         
    random_state=seed
)

clf_tuned.fit(X_train, y_train)

tuned_train_auc = roc_auc_score(y_train, clf_tuned.predict_proba(X_train)[:, 1])
tuned_test_auc = roc_auc_score(y_test, clf_tuned.predict_proba(X_test)[:, 1])

print(f"Настроенное дерево:")
print(f"ROC AUC Train: {tuned_train_auc:.3f}")
print(f"ROC AUC Test: {tuned_test_auc:.3f}")

if tuned_test_auc > 0.85:
    print("Цель достигнута! Точность > 0.85 на отложенной выборке.")