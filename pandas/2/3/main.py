import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

comedies_2016 = df.loc[(df['Year'] == 2016) & (df['Genre'].str.contains('Comedy', na=False))]

comedies_2016 = comedies_2016.sort_values(by='Title', ascending=False)

print("Комедии, снятые в 2016 году:")
print(comedies_2016['Title'].to_string(index=False))
# print(len(comedies_2016))