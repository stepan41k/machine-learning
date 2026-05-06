import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

longest_movie = df.loc[df['Runtime (Minutes)'].idxmax()]

shortest_movie = df.loc[df['Runtime (Minutes)'].idxmin()]

print("Самый продолжительный фильм:")
print(f"Название: {longest_movie['Title']}")
print(f"Режиссер: {longest_movie['Director']}")
print(f"Продолжительность: {longest_movie['Runtime (Minutes)']} мин.")

print()

print("Самый короткий фильм:")
print(f"Название: {shortest_movie['Title']}")
print(f"Режиссер: {shortest_movie['Director']}")
print(f"Продолжительность: {shortest_movie['Runtime (Minutes)']} мин.")