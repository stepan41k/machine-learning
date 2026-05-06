import pandas as pd

data_left = {
    'col1': ['x', 'y', 'z'],
    'col2': [0, 1, 2]
}
data_right = {
    'col1': ['a', 'b', 'c'],
    'col3': [6, 7, 8]
}

df_left = pd.DataFrame(data_left)
df_right = pd.DataFrame(data_right)

print("Исходный df_left")
print(df_left)
print("\nИсходный df_right")
print(df_right)

df_concat_rows = pd.concat([df_left, df_right], axis=0, ignore_index=True)

print("\nКонкатенация по строкам")
print(df_concat_rows)

df_concat_cols = pd.concat([df_left, df_right], axis=1, ignore_index=True)

print("\nКонкатенация по столбцам")
print(df_concat_cols)