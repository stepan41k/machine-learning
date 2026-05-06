import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

sci_fi_movies = df.loc[df['Genre'].str.contains('Sci-Fi', na=False)]

total_sci_fi = len(sci_fi_movies)

high_score_sci_fi = len(sci_fi_movies[sci_fi_movies['Metascore'] > 50])

if total_sci_fi > 0:
    proportion = (high_score_sci_fi / total_sci_fi) * 100
    print(f"Всего Sci-Fi фильмов: {total_sci_fi}")
    print(f"Из них с Metascore > 50: {high_score_sci_fi}")
    print(f"Доля: {proportion:.2f}%")
else:
    print("Фильмы жанра Sci-Fi не найдены.")