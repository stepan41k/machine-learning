import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../../content/IMDB-Movie-Data.csv')

comedy = df[df['Genre'].str.contains('Comedy', na=False)][['Rating']].copy()
comedy['Genre_Label'] = 'Комедии'

thriller = df[df['Genre'].str.contains('Thriller', na=False)][['Rating']].copy()
thriller['Genre_Label'] = 'Триллеры'

combined_df = pd.concat([comedy, thriller])

plt.figure(figsize=(10, 6))

sns.violinplot(x='Genre_Label', y='Rating', data=combined_df, palette="Pastel1")

plt.title('Сравнение распределения рейтингов: Комедии и Триллеры', fontsize=15)
plt.xlabel('Жанр')
plt.ylabel('Рейтинг (Rating)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

print(f"Средний рейтинг комедий: {comedy['Rating'].mean():.2f}")
print(f"Средний рейтинг триллеров: {thriller['Rating'].mean():.2f}")
print(f"Медианный рейтинг комедий: {comedy['Rating'].median():.2plt.xscale('log')f}")
print(f"Медианный рейтинг триллеров: {thriller['Rating'].median():.2f}")