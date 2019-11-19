import pandas as pd
from levenshtein_distance import minimumEditDistance as md

GT = 'GT.xlsx'
check = 'check.xlsx'

gt_data = pd.read_excel(GT)
check_data = pd.read_excel(check)
for col in gt_data.columns: 
    gt_col = gt_data[col]
    check_col = check_data[col]
    if gt_col.size != check_col.size:
        print('size not same')
        break
    for i in range(gt_col.size):
        gt_word = gt_col[i]
        if pd.isna(gt_word):
            continue
        gt_word = gt_word.replace(' ','')
        check_word = check_col[i].replace(' ','')
        value = md(gt_word,check_word)
        if value:
            print(gt_word,check_word,value)
# print(type(data['Low'][0]))g