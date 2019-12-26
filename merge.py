import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
import re

root = '/home/umaid/Experiments/guitar/GMag2Data/new_processed_books/excels'
files = glob.glob(os.path.join(root,'*.xlsx'))
files = [pd.read_excel(file) for file in files]
merged = pd.concat(files, axis=0,sort= True)
# new_feature = merged['Features'].map(lambda x: re.sub(r'\W+', '', x))
# merged['New_Feature'] = merged['Features'].str.translate(None, ["_~—-="])
# merged['New_Feature'] = merged['Features'].str.replace(r"[^a-zA-Z ]+", "").str.strip()
# regex = re.compile(r'(\d{4}s?-\d{4}s?)|(\d{4}s?-\d{2}s?)|(\d{4}s?)')
# regex = re.compile(r'\d{4}s?-?[0-9]?[0-9]?[0-9]?[0-9]?s?')
# regex = re.compile(r'(\d{4}s?-\d{2,4}s?)|(\d{4}s?)')
def improve_year(x):
    out = []
    # print('start',x)
    x = re.sub(r'[|\]\[!t\{\}\(LIijJl\)]', '1', str(x))
    x = re.sub(r'[OC]', '0', str(x))
    x = re.sub(r'[S]', '5', str(x))
    x = re.sub(r'[^\ds-]','',str(x))
    # print('end',x)
    temp = re.sub(r's','',x).split("-")
    x = x.split('-')
    # print('x',x)
    # print('temp',temp)
    # print(temp,x)
    twe = '200'
    nin = '190'
    # print(temp)
    for i,w in enumerate(temp):
            # print(w[:2] == '19')

        if len(w) > 4:
            if int(w[:4]) > 1700 and int(w[:4]) < 2050:
                temp[i] = w[:4]
            else:
                temp[i] = w[1:]

        if len(w) == 2 and len(temp) > 1:
            if i == 1 and int(temp[i]) < int(temp[0][2:]):
                temp[i] = '20'+ w
            else:
                temp[i] = '19'+ w
            if i == 0 and int(temp[i]) > int(temp[1][2:]):
                temp[i] = '19'+ w
            else:
                temp[i] = '20'+ w

        if len(w) < 4:
            if int(w) < 10:
                temp[i] = twe[:4-len(w)]+ w
            else:
                temp[i] = nin[:4-len(w)]+ w

        out.append(temp[i])
        if 's' in x[i]:
            out[i] = out[i] + 's'
        # if len(w) < 4 and 
    # temp = [w[:4] if len(w) > 4 and int(w[:4]) > 1800 else '' w[1:] if len(w) > 4 and int(w[:4]) <= 1800 else '' for w in temp]
    out = '-'.join(out)
    # # out = regex.findall()
    # if len(out):
    #     if len(out)>1:
    #         out = out[0][1:]+out[1]
    #     out = ''.join(out[0])
    # else:
    #     out = x
    return out

merged['Model_year'] = merged['Model_year'].apply(lambda x: improve_year(x)) 
# merged['year'] = merged['Model_year'].apply(lambda x: improve_year(x))
# print(merged['year'])
# assert False
# feat = pd.concat([merged['Model_year'],merged['year']],axis = 1)
# feat['Model_year'] = merged['Model_year']
# feat2['Model_year'] = merged['year']
# feat.to_excel("processed_books/org.xlsx",index = None)
# feat2.to_excel("processed_books/copy.xlsx",index = None)
# assert False
merged['Features'] = merged['Features'].apply(lambda x: re.sub(r'[^a-zA-Z.\d\s^$"-]', '', str(x)).strip())
merged = merged.drop("Page",axis = 1)
merged['Features'] = merged['New_Feature'].fillna('')
# feat = pd.concat([merged['Features'],merged['New_Feature']],axis = 1)
# feat['New_Feature'] = feat['New_Feature'].fillna('')
# print(feat)
# feat.to_excel("processed_books/feat.xlsx",index = None)
# # print(merged['features'])
# assert False
# merged.to_excel("processed_books/merged.xlsx",index = None)
years = np.sort(merged.Year.unique())
# print(years)
# assert False
columns = ['Type','Manufacturer','Model','Model_year','Features']
grouped = merged.sort_values(['Year','Model_year']).groupby(["Type","Manufacturer","Model",'Features'])
for year in years:
    columns.append(str(year)+'_Low')
    columns.append(str(year)+'_High')
    columns.append(str(year)+'_diff')
# groups = [group for _,group in grouped]
excel = []
head = dict.fromkeys(columns, "")
N = pd.DataFrame(head,index=[0])
head = N.copy()

for name,gr in tqdm(grouped):
    print(name)                       
    print(gr)
    """Temp code"""
    # head = pd.concat([head,gr,N],axis = 0,ignore_index = True)
    # print(head)
    """Temp code"""

    head = dict.fromkeys(columns, None)                      
    for row_index, group in gr.iterrows():
        head['Type'] = group['Type']                        
        head['Manufacturer'] = group['Manufacturer']
        head['Model'] = group['Model']
        head['Model_year'] = group['Model_year']
        head['Features'] = group['Features']
        year = group["Year"]
        head[str(year)+'_Low'] = group["Low"] 
        head[str(year)+'_High'] = group["High"]
        head[str(year)+'_diff'] = group['High'] - group['Low']
    print(head)
    excel.append(head.copy())


# final = pd.concat(groups, axis = 0).drop("Page",axis = 1)

file = pd.DataFrame(excel)                     
# df.to_csv(index=False)
# file.to_excel("processed_books/final.xlsx",index = None,columns=list(head.keys()))
file.to_csv("new_processed_books/excels/group.csv",columns=list(head.keys()))   
# head.to_csv("new_processed_books/excels/group.csv",columns=list(head.keys()))   
   # print(name)
   # print(group)