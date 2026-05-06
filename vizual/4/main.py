import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../../content/IMDB-Movie-Data.csv')

df_filtered = df[(df['Year'] >= 2010) & (df['Year'] <= 2016)].copy()

horror_counts = df_filtered[df_filtered['Genre'].str.contains('Horror', na=False)].groupby('Year').size().reset_index(name='Count')
family_counts = df_filtered[df_filtered['Genre'].str.contains('Family', na=False)].groupby('Year').size().reset_index(name='Count')

horror_counts['Genre_Type'] = 'Horror'
family_counts['Genre_Type'] = 'Family'

plot_data = pd.concat([horror_counts, family_counts])

plt.figure(figsize=(12, 6))
sns.barplot(data=plot_data, x='Year', y='Count', hue='Genre_Type', palette={'Horror': '#4c72b0', 'Family': '#f39c12'})

plt.title('Сравнение количества фильмов ужасов и семейных фильмов (2010-2016)', fontsize=14)
plt.xlabel('Год', fontsize=12)
plt.ylabel('Количество фильмов', fontsize=12)
plt.legend(title='Жанр')
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()