import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
def convert(start,batch,diff,file_path,output_path):
    for i in tqdm(range( diff - 1 )):
        images = convert_from_path(file_path,300,first_page = start+(batch*i)+1, last_page = start + (batch * (i+1)))
        temp_images = []
        page_num =start+(batch*i)
        for i in range(len(images)):
            page_num+=1
            image_path = os.path.join(output_path,'page_'+str(page_num))
            print(image_path)
            images[i].save(image_path, 'JPEG')

def convert_pdf(file_path, output_path):
    os.makedirs(output_path,exist_ok = True)
    end = 625
    start = 30
    batch = 20
    diff = int((end - start)/batch)
    convert(start,batch,diff,file_path,output_path)
    check_end = start+(batch*diff)
    check_batch = end - check_end
    if (check_batch):
        convert(check_end,check_batch,2,file_path,output_path)

convert_pdf('../guitar.pdf','pdf_pages')
