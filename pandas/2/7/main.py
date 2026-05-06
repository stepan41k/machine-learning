import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

comedies = df.loc[df['Genre'].str.contains('Comedy', na=False)].copy()

#series
groupby_result = comedies.groupby('Year')['Title'].count().sort_values(ascending=False)

print("Результат через groupby():")
print(groupby_result)

#df
pivot_result = pd.pivot_table(
    comedies,
    index='Year',
    values='Title',
    aggfunc='count'
).sort_values(by='Title', ascending=False)

print("\nРезультат через pivot_table():")
print(pivot_result)