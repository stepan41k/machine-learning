import pandas as pd
import numpy as np

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

sci_fi_movies_count = df.loc[df['Genre'].str.contains('Sci-Fi', na=False)].shape[0]

print(f"Количество научно-фантастических фильмов: {sci_fi_movies_count}")