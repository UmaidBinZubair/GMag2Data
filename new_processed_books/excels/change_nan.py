import pandas as pd
df = pd.read_csv('/home/umaid/Experiments/guitar/GMag2Data/new_processed_books/excels/2001-2020(31.12.19)v2.csv')
df['Features'] = df['Features'].fillna('')
print(df['Features'])
df.to_csv("2001-2020(31.12.19)v2.csv")