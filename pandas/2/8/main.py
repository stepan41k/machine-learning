import pandas as pd

df = pd.read_csv('../../../content/IMDB-Movie-Data.csv')

comedies = df.loc[df['Genre'].str.contains('Comedy', na=False)]

grouped_by_year = comedies.groupby('Year')

comedies_2016 = grouped_by_year.get_group(2016)

comedies_2016 = comedies_2016.sort_values(by='Title', ascending=False)

print("Названия комедий 2016 года (через get_group):")
print(comedies_2016['Title'].to_string(index=False))
# print(len(comedies_2016))