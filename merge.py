import numpy as np
import pandas as pd
import glob
import os
import re

root = '/home/umaid/Experiments/guitar/GMag2Data/processed_books/excels'
files = glob.glob(os.path.join(root,'*.xlsx'))
files = [pd.read_excel(file) for file in files]
merged = pd.concat(files, axis=0,sort= True)
# new_feature = merged['Features'].map(lambda x: re.sub(r'\W+', '', x))
# merged['New_Feature'] = merged['Features'].str.translate(None, ["_~—-="])
# merged['New_Feature'] = merged['Features'].str.replace(r"[^a-zA-Z ]+", "").str.strip()
# regex = re.compile(r'(\d{4}s?-\d{4}s?)|(\d{4}s?-\d{2}s?)|(\d{4}s?)')
# regex = re.compile(r'\d{4}s?-?[0-9]?[0-9]?[0-9]?[0-9]?s?')
regex = re.compile(r'(\d{4}s?-\d{2,4}s?)|(\d{4}s?)')
def improve_year(x):
    out = regex.findall(re.sub(r'[|\]\[!t\{\}\(LIil\)]', '1', str(x)))
    if len(out):
        if len(out)>1:
            out = out[0][1:]+out[1]
        out = ''.join(out[0])
    else:
        out = x
    return out

merged['year'] = merged['Model_year'].apply(lambda x: improve_year(x))
# print(merged['year'])
# assert False
feat = pd.concat([merged['Model_year'],merged['year']],axis = 1)
feat.to_excel("processed_books/feat.xlsx",index = None)
assert False
merged['New_Feature'] = merged['Features'].apply(lambda x: re.sub(r'[^a-zA-Z.\d\s^$"-\]\[]', '', str(x)).strip())
feat = pd.concat([merged['Features'],merged['New_Feature']],axis = 1)
feat['New_Feature'] = feat['New_Feature'].fillna('')
# print(feat)
feat.to_excel("processed_books/feat.xlsx",index = None)
# print(merged['features'])
assert False
# merged.to_excel("processed_books/merged.xlsx",index = None)
years = np.sort(merged.Year.unique())
print(years)
# assert False
columns = ['Manufacturer','Model','Model_year','Features']
grouped = merged.sort_values(['Year','Model_year']).groupby(["Manufacturer","Model",'Features'])
for year in years:
    columns.append(str(year)+'_Low')
    columns.append(str(year)+'_High')
# groups = [group for _,group in grouped]
excel = []
for name,gr in grouped:
    print(name)
    print(gr)
    head = dict.fromkeys(columns, None)
    for row_index, group in gr.iterrows():
        head['Manufacturer'] = group['Manufacturer']
        head['Model'] = group['Model']
        head['Model_year'] = group['Model_year']
        head['Features'] = group['Features']
        year = group["Year"]
        head[str(year)+'_Low'] = group["Low"] 
        head[str(year)+'_High'] = group["High"] 
    print(head)
    excel.append(head.copy())


# final = pd.concat(groups, axis = 0).drop("Page",axis = 1)

file = pd.DataFrame(excel)
# df.to_csv(index=False)
# file.to_excel("processed_books/final.xlsx",index = None,columns=list(head.keys()))
file.to_csv("processed_books/result.csv",columns=list(head.keys()))
   # print(name)
   # print(group)