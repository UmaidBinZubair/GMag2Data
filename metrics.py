import pandas as pd
import argparse
import sys
import os
from excelWriter import excelWrite
from utils import error_rate

def metric(gt_data,check_data,debug,out_dir):

    cells_to_color = []
    acc = {}
    for j,col in enumerate(gt_data.columns): 
        gt_col = gt_data[col]
        check_col = check_data[col]
        if gt_col.size != check_col.size:
            print('size not same')
            break
        words = 0
        chars = 0
        word_error = 0
        cha_error = 0
        print(f"{col}:\n")
        for i in range(gt_col.size):
            gt_word = str(gt_col[i])
            check_word = str(check_col[i])
            if (col == 'Low' or col == 'High') and type(gt_word) == 'float' :
                gt_word = '$'+str(gt_word)
            if pd.isna(gt_word) and pd.isna(check_word):
                continue
            elif pd.isna(gt_word):
                c_error = len(check_word.replace(' ',''))
                w_error = len(check_word.split())
                cha_error += c_error
                word_error += w_error
                words += w_error
                chars += c_error
                if debug:
                    cells_to_color.append([i, j])
            elif pd.isna(check_word):
                cha_error = len(gt_word.replace(' ',''))
                word_error = len(gt_word.split())
                cha_error += c_error
                word_error += w_error
                words += w_error
                chars += c_error
                if debug:
                    cells_to_color.append([i, j])
            else:
                gt_word_list = gt_word.split()
                check_word = str(check_word)
                check_word_list = check_word.split()

                gt_word_ns = gt_word.replace(' ','')
                check_word_ns = check_word.replace(' ','')

                c_error = error_rate(gt_word_ns,check_word_ns)
                w_error = error_rate(gt_word_list,check_word_list)
                word_error += w_error
                cha_error += c_error
                words += len(gt_word_list)
                chars += len(gt_word)
                if w_error and debug:
                    cells_to_color.append([i, j])

        print( 
            f"Cha Acc: {((chars-cha_error)/chars):.4f}\n"
            f"Word Acc: {((words-word_error)/words):.4f}\n\n"
            )

        acc[col] =( 
            f"Cha Acc: {((chars-cha_error)/chars):.5f}\n"
            f"Word Acc: {((words-word_error)/words):.5f}\n\n"
            )

    def colorize(x):
        color = 'background-color: #E45334'
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        for elem in cells_to_color:
            df1.iloc[elem[0]][elem[1]] = color
        return df1

    check_data = check_data.append(acc,ignore_index = True)
    final_data = check_data.style.apply(colorize, axis=None)
    final_data.to_excel(os.path.join(out_dir,"result.xlsx"),index = None)


if __name__== "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--debug', help = 'Enable debug', action='store_true')
    parser.add_argument('-g','--gt_file', help = 'Path of ground truth excel file')
    parser.add_argument('-f','--output_file', help = 'Path of output excel file')
    parser.add_argument('-o',"--out_dir", help = "Directory of evaluation output", default = './')

    args = parser.parse_args()

    if not len(sys.argv) > 1 :
        print ('No input has been provided')
    else:
        GT = args.gt_file
        output = args.output_file
        gt_data = pd.read_excel(GT)
        output_data = pd.read_excel(output)
        debug = args.debug
        out_dir = args.out_dir
        metric(gt_data.drop(columns = 'Page'),output_data.drop(columns = 'Page'),debug,out_dir)