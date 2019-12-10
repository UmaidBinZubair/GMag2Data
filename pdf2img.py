import os
import sys
import tempfile
import argparse
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

def convert(start,batch,diff,file_path,output_path):
    for i in tqdm(range(diff)):
        images = convert_from_path(file_path,300,first_page = start+(batch*i)+1, last_page = start + (batch * (i+1)))
        temp_images = []
        page_num =start+(batch*i)+1
        for i in range(len(images)):
            image_path = os.path.join(output_path,'page_'+str(page_num))
            print(image_path)
            images[i].save(image_path+'.jpg', 'JPEG')
            page_num+=1

def convert_pdf(file_path, output_path,start,end,batch):
    os.makedirs(output_path,exist_ok = True)
    diff = int((end - start)/batch)
    convert(start ,batch,diff,file_path,output_path)
    check_end = start+(batch*diff)
    check_batch = end - check_end
    if (check_batch):
        convert(check_end,check_batch,1,file_path,output_path)

if __name__== "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', help = 'Path to the pdf file')
    parser.add_argument('-o',"--out_dir", help = "Directory of output images", default = './')
    parser.add_argument('-s','--start', help = 'Path of ground truth excel file',type = int)
    parser.add_argument('-e','--end', help = 'Path of ground truth excel file',type = int)
    parser.add_argument('-b','--batch', help = 'Path of ground truth excel file',type = int, default = 20)

    args = parser.parse_args()

    if not len(sys.argv) > 1 :
        print ('No input has been provided')
    else:    
        file = args.file
        out_dir = args.out_dir
        root = os.getcwd()
        os.makedirs(os.path.join(root,out_dir),exist_ok = True)
        start = args.start
        end = args.end
        batch = args.batch
        convert_pdf(file,out_dir,start-1,end,batch)
