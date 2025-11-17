import pandas as pd
from ast import literal_eval as le

# ================================
# 1. Load CSV (LOCAL SYSTEM)
# ================================
csv_path = "/home/adarsh/Desktop/ml_l_class_ref/files/movies_metadata.csv"   # ← change to your file path

df = pd.read_csv(csv_path, encoding='latin1', on_bad_lines='skip')

# ================================
# 2. Basic Info
# ================================
print(df.columns)
print(df.shape)




print(df.info())

# ================================
# 3. iloc examples
# ================================
print("\nRow 2:")

#  iloc 2,3 2nd row 3rd col only
# [  2:4 , 3:5]  2,3 row ka 3,4 col only 
print(df.iloc[2])



print("\nRows 3–4, Columns 2–3:")

print(df.iloc[3:5, 2:4])

# ================================
# 4. Create smaller DataFrame
# ================================
small_df = df[['title', 'release_date', 'budget', 'revenue', 'runtime']]
print("\nSmall DF (first rows):")
print(small_df.head())

# ================================
# 5. Sort by release date
# ================================
small_df = small_df.sort_values('release_date')
print("\nSorted by release_date:")
print(small_df.head())

# ================================
# 6. Filter by date > 1995
# ================================
result = small_df[small_df['release_date'] > '1995-01-01']
print("\nMovies after 1995:")
print(result)

# ================================
# 7. Sort by runtime descending
# ================================
result = small_df.sort_values('runtime', ascending=False)
print("\nTop movies by runtime:")
print(result.head())

# ================================
# 8. Runtime stats
# ================================
print("\nRuntime Max:", small_df['runtime'].max())
print("Runtime Min:", small_df['runtime'].min())
print("Runtime Mean:", small_df['runtime'].mean())




# ================================
# 9. Example: converting string list to real list
# ================================
data = {
    'id': [1, 2, 3],
    'tags': ["['AI', 'ML']", "['Python']", "['Data', 'Science']"]
}

df2 = pd.DataFrame(data)

df2['tags'] = df2['tags'].apply(le)

print("\nConverted tags column:")
print(df2)
