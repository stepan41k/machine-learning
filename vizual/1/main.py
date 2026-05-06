import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('../../content/IMDB-Movie-Data.csv')

df_2016 = df[df['Year'] == 2016]

metascore_data = df_2016['Metascore'].dropna()

plt.figure(figsize=(10, 6))

sns.histplot(metascore_data, kde=True, color='royalblue', bins=20)

plt.title('Распределение Metascore фильмов 2016 года', fontsize=15)
plt.xlabel('Metascore', fontsize=12)
plt.ylabel('Плотность', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.show()
