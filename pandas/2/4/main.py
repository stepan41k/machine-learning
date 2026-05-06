import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

top_10_revenue = df.sort_values(by='Revenue (Millions)', ascending=False).head(10)
top_10_revenue = top_10_revenue[['Title', 'Director', 'Revenue (Millions)']]

print("Топ-10 наиболее прибыльных фильмов:")
print(top_10_revenue.to_string(index=False))