import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

dicaprio_movies = df.loc[df['Actors'].str.contains('Leonardo DiCaprio', na=False)]

print("Фильмы, в которых снимался Leonardo DiCaprio:")
print(dicaprio_movies['Title'].to_string(index=False))