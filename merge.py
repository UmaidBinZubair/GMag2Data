import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import os
import re

def improve_year(x):
    out = []
    x = re.sub(r'[|\]\[!t\{\}\(LIijJl\)]', '1', str(x))
    x = re.sub(r'[OC]', '0', str(x))
    x = re.sub(r'[S]', '5', str(x))
    x = re.sub(r'[^\ds-]','',str(x))
    temp = re.sub(r's','',x).split("-")
    x = x.split('-')
    twe = '200'
    nin = '190'

    for i,w in enumerate(temp):

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
    
    out = '-'.join(out)
    return out

def preceedingtraits(x):
    char_one = re.compile(r'[a-km-zA-KM-Z][\[\]!{}()|]')
    digit_one = re.compile(r'[\d\s]["\[\]!{}()|"]')
    digit_zero = re.compile(r'\d["Oo"]')
    words = x.split()
    for i,word in enumerate(words):
        if char_one.findall(word):
            word = re.sub(r'[\[\]!{}()|]', 'l', str(word))
        if digit_one.findall(word):
            # print('asdf')
            word = re.sub(r'[\[\]!{}()|]', '1', str(word))
        if digit_zero.findall(word):
            word = re.sub(r'[oO]', '0', str(word))
        words[i] = word
    return ' '.join(words)


def cleaning(df):
    df = df.apply(lambda x: re.sub(r'[^a-zA-Z\d\s^$#\[\]!{}()|]', '', str(x)).strip())
    df = df.apply(lambda x: re.sub(r'[$§]', 'S', str(x)).strip())
    df = df.apply(lambda x: preceedingtraits(x))
    df = df.str.title()
    return df


root = '/home/umaid/Experiments/guitar/GMag2Data/new_processed_books/excels'
files = glob.glob(os.path.join(root,'*.xlsx'))
files = [pd.read_excel(file) for file in files]
merged = pd.concat(files, axis=0,sort= True)


merged['Model_year'] = merged['Model_year'].apply(lambda x: improve_year(x))
# merged['Features'] = cleaning(merged['Features'])
# merged['Model'] = cleaning(merged['Model'])
# merged['Manufacturer'] = cleaning(merged['Manufacturer'])
merged['Type']=merged['Type'].str.title()
merged['Features'] = merged['Features'].apply(lambda x: re.sub(r'Nan', '', str(x)).strip())
merged = merged.drop("Page",axis = 1)

years = np.sort(merged.Year.unique())
columns = ['Type','Manufacturer','Model','Model_year','Features']
# print(merged.Type.unique())
# assert False
grouped = merged.sort_values(['Year','Model_year']).groupby(["Type","Manufacturer","Model",'Features'])
# grouped = merged.groupby(["Type","Manufacturer","Model",'Features'])


"""Temp code"""
merged['Features'] = merged['Features'].apply(lambda x: re.sub(r'nan', '', str(x)).strip())
col = ['Type','Manufacturer','Model','Model_year','Features','High','Low','Year']
head =  [gr[1] for gr in grouped]    
head = pd.concat(head,axis = 0,ignore_index = True)
head = head.fillna('')
head.to_csv("new_processed_books/excels/original.csv",columns=col)
assert False
"""Temp code"""


for year in years:
    columns.append(str(year)+'_Low')
    columns.append(str(year)+'_High')
    columns.append(str(year)+'_diff')

excel = []
head = dict.fromkeys(columns, "")
N = pd.DataFrame(head,index=[0])
head = N.copy()
"""Temp code"""

# # print(head)
# # """Temp code"""
for name,gr in tqdm(grouped):
    # print(name)                       
    # print(gr)

    all_years = [str(group['Model_year']).split('-')[0] for row_index, group in gr.iterrows()]
    unique_years = np.unique(all_years)
    # print(all_years)
    # print(unique_years)
    year_groups = dict()
    for row_index, group in gr.iterrows():
        # print(name)                       
        # print(gr)
        if '-' in str(group['Model_year']):
            split_yr = str(group['Model_year']).split('-')[0]
        else:
            split_yr = str(group['Model_year']) + 'x'
        
        if split_yr in year_groups.keys():
            year_groups[split_yr].append(group)
        else:
            year_groups[split_yr] = [group]

    new_groups = dict()
    current_key = ""
    for key, group_rows in year_groups.items():
        for i, row in enumerate(group_rows):
            # print(row['Model'])
            # print('row',row)
            # print(row['Model_year'],current_key)
            if i == 0:
                current_key = row['Model_year']
                new_groups[current_key] = [row]

            # if current_key not in new_groups.columns:
            #     new_groups[current_key] = list()
            # print(len(group_rows))
            if i < len(group_rows) and i > 0:
                try:
                    # print(i, len(group_rows))
                    # print(group_rows[i]["Model_year"].split("-"))
                    next_year = str(group_rows[i]["Model_year"]).split("-")[1] 
                    # print(current_key,row['Model_year'])
                    # current_key = row['Model_year']
                    prev_year = str(group_rows[i-1]['Model_year']).split("-")[1]
                    next_year = int(next_year.replace("s", ""))
                    prev_year = int(prev_year.replace("s", ""))
                    # print(row['Model'],next_year,prev_year)
                    if next_year - prev_year > 1:
                        current_key = row['Model_year']
                        new_groups[current_key] = [row]
                    else:
                        # current_key = row['Model_year']
                        new_groups[current_key].append(row)
                except Exception as e:
                    # print(e)
                    # print(row['Model'],"0","0")
                    # new_groups
                    new_groups[current_key].append(row)


    # print(year_groups)
    print(new_groups,'\n')
    # assert False
    # exit(0)

    head = dict.fromkeys(columns, None)                      
    for key,data in new_groups.items():
        for group in data:

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
file['Features'] = file['Features'].fillna('')                     
# df.to_csv(index=False)
# file.to_excel("processed_books/final.xlsx",index = None,columns=list(head.keys()))
file.to_csv("new_processed_books/excels/2001-2020(31.12.19)v2",columns=list(head.keys()))   
# head.to_csv("new_processed_books/excels/group.csv",columns=list(head.keys()))   
   # print(name)
   # print(group)