import pandas as pd
import argparse
import sys
import os
from utils import error_rate

def print_detail(gt_word,check_word,c_error,w_error,file):
    print(f"REF: {gt_word}\n"
        f"HYP: {check_word}\n"
        f"cer: {c_error}\n"
        f"wer: {w_error}\n\n")

    file.write(f"REF: {gt_word}\n"
        f"HYP: {check_word}\n"
        f"cer: {c_error}\n"
        f"wer: {w_error}\n\n")


def metric(gt_data,check_data,debug,out_dir):

    result = os.path.join(out_dir,'result.txt')
    with open(result, "w") as file:
        for col in gt_data.columns: 
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
            file.write(f"{col}:\n")
            for i in range(gt_col.size):
                gt_word = gt_col[i]
                check_word = check_col[i]
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
                        print_detail(gt_word,check_word,c_error,w_error,file)
                elif pd.isna(check_word):
                    cha_error = len(gt_word.replace(' ',''))
                    word_error = len(gt_word.split())
                    cha_error += c_error
                    word_error += w_error
                    words += w_error
                    chars += c_error
                    if debug:
                        print_detail(gt_word,check_word,c_error,w_error,file)
                else:
                    gt_word_list = gt_word.split()
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
                        print_detail(gt_word,check_word,c_error,w_error,file)

            print(
                f"character Accuracy: {(chars-cha_error)/chars}\n"
                f"Word Accuracy: {(words-word_error)/words}\n\n"
                )
            file.write(
                f"character Accuracy: {(chars-cha_error)/chars}\n"
                f"Word Accuracy: {(words-word_error)/words}\n\n"
                )

# def text_detection(txt_file)

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
        metric(gt_data,output_data,debug,out_dir)
