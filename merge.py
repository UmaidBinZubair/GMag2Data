import numpy as np
import pandas as pd
import glob
import os

root = '/home/umaid/Experiments/guitar/GMag2Data/processed_books/excels'
files = glob.glob(os.path.join(root,'*.xlsx'))
files = [pd.read_excel(file) for file in files]
merged = pd.concat(files, axis=0,sort= True)
years = np.sort(merged.Year.unique())
columns = list(merged.columns)
columns.remove("High")
columns.remove("Low")
columns.remove("Features")
columns.remove("Year")
columns.remove("Page")
# columns.remove("Model_year")
grouped = merged.sort_values(['Year','Model_year']).groupby(["Manufacturer","Model",'Model_year'])
for year in years:
    columns.append(str(year)+'_Low')
    columns.append(str(year)+'_High')

# groups = [group for _,group in grouped]
excel = []
for name,gr in grouped:
    head = dict.fromkeys(columns, None)
    for row_index, group in gr.iterrows():
        print(group)
        head['Manufacturer'] = group['Manufacturer']
        head['Model'] = group['Model']
        head['Model_year'] = group['Model_year']
        year = group["Year"]
        head[str(year)+'_Low'] = group["Low"] 
        head[str(year)+'_High'] = group["High"] 
        print(head)
    excel.append(head.copy())


# final = pd.concat(groups, axis = 0).drop("Page",axis = 1)

file = pd.DataFrame(excel)
file.to_excel("processed_books/final.xlsx",index = None,columns=list(head.keys()))
   # print(name)
   # print(group)