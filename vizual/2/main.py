import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../../content/IMDB-Movie-Data.csv')

plt.figure(figsize=(12, 7))

sns.scatterplot(data=df, x='Votes', y='Rating', alpha=0.5, color='darkblue')

sns.regplot(data=df, x='Votes', y='Rating', scatter=False, color='red')

# plt.xscale('log')

plt.title('Зависимость рейтинга фильма от количества проголосовавших', fontsize=15)
plt.xlabel('Количество голосов (Votes)', fontsize=12)
plt.ylabel('Рейтинг (Rating)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()