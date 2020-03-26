import pytesseract
import cv2
import os
import re
import sys
import pandas as pd
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from utils import iou,bb,error_rate
from excelWriter import excelWrite

pd.options.display.width = 0

types = ['GUITARS','BASSES','AMPS','EFFECTS','STEEL & LAP STEEL','MANDOLINS','UKULELES','BANJOS']

lin = {}
lin['Page'] = ''
lin['Manufacturer'] = ''
lin['Model'] = ''
lin['Model_year'] = ''
lin['Features'] = ''
lin['Low'] = ''
lin['High'] = ''
lin['Year'] = ''
lin['Type'] = ''

num = 0

def constant_aspect_resize(image, width=2500, height=None):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    if dim[0] < w:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
    else:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def apply_ocr(path, image):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        ocr = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config = '--psm 4')

        with open(path, "wb") as f:
            pickle.dump(ocr, f)

        return ocr

def removeLine(img):
    
    _h,_w = img.shape
    ret,thresh = cv2.threshold(img,10,255,cv2.THRESH_BINARY_INV)
    cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    widths = {}
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 0.08*(_w*_h):
            return None
        perimeter = cv2.arcLength(c, False)
        if area > 1 and perimeter > 1:
            x,y,w,h = cv2.boundingRect(c)
            widths[w] = [x,y,w,h]
    try:
        max_wid = max(widths.keys()) 
        if max_wid > 0.75 * _w:
            # print(max_wid,0.75*_w)
            # print(widths[max_wid])
            x,y,w,h = widths[max_wid]
            img[0:y+h+3,:] = 255
            # img[y-3:y+h+3,x-3:x+w+3] = 255
        # print('sdglsjdgliijojo')
        # cv2.rectangle(img,(x,y),(x+w,h+y),(0,0,255),3)
        # cv2.imshow('',constant_aspect_resize(img, width=None, height=600))
        # cv2.waitKey(0)
        return img
    except:
        pass
    
def verticalProj(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        pass
    # thresh, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)    
    # image = 255 - image
    thresh,image = cv2.threshold(image,200,255,cv2.THRESH_BINARY_INV)
    h,w = image.shape
    image = image[int(0.065*h):int(0.95*h),:]
    proj = np.sum(image,0) 
    brk = np.where(proj==0)
    # print(brk)
    cha = (np.diff(brk[0])>1)
    final = np.where(cha ==True)
    result = brk[0][final[0]]
    if len(final[0]):
        gaps = np.insert(brk[0][final[0]],-1,brk[0][-1])
    else:
        gaps = []
    return gaps

def horizontalProj(image):    

    image = cv2.threshold(image,0,255,cv2.THRESH_BINARY)[1]
    image = 255 - image
    h,w = image.shape
    proj = np.sum(image,1)
    proj[proj<(255*(w/16))] = 0 
    brk = np.where(proj==0)
    cha = (np.diff(brk[0])>1)
    final = np.where(cha ==True)
    if len(final[0]):
        gaps = np.insert(brk[0][final[0]],-1,brk[0][-1])
    else:
        gaps = []
    return gaps

def line_align(data,w):
    i = 0
    while i < len(data):
        r1 = data[i]['box']
        a = [0,r1[1],w,r1[1]+r1[3]]
        j = 0
        while j < len(data):
            r2 = data[j]['box']
            b = [0,r2[1],w,r2[1]+r2[3]]
            if bb(a,b) > 0 and r2[0] > r1[0]:
                # print(bb(a,b),data[i]['words'],data[j]['words'])
                data[i]['words'] += data[j]['words']
                data[i]['box'][0] = min(data[i]['box'][0],data[j]['box'][0])
                data[i]['box'][1] = min(data[i]['box'][1],data[j]['box'][1])
                data[i]['box'][2] = data[j]['box'][0] + data[j]['box'][2] - data[i]['box'][0]  
                data[i]['box'][3] = data[j]['box'][1] + data[j]['box'][3] - data[i]['box'][1]
                del data[j]
                j-=1
            j+=1
        i+=1
    return sorted(data, key = lambda i: i['box'][1]) 

def findamount(data,w):
    i = 0
    # print(data)
    # print(len(data))
    while i < len(data):
        # print(i,len(data))
        line = data[i]
        dollor_num = 0
        for elem in line['words']:
            if '$' in elem:
                dollor_num += 1
            if dollor_num > 1:
                line['amount'] = 1
                if line['box'][0] > 0.09*w:
                    data[i-1]['words'] = data[i-1]['words'] + line['words']
                    data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3] + 10
                    data[i-1]['box'][2] = (line['box'][0]+line['box'][2]) - data[i-1]['box'][0]   
                    data[i-1]['amount'] = 1
                    del data[i]
                    i -= 2
        i+=1

def findHeader(data,image,i,save,out_dir,head_pkl,itr):

    img = image.copy()
    image = findHeaderblob(image)
    # cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(i)+'_header.jpg')),constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        b = line['box']
        crop = image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
        _crop = img[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
        if crop.size == 0:
            continue
        summed = np.sum(crop)/(crop.size)
        widths = findContourWidth(_crop)
        wid = np.mean(widths) + (2*np.std(widths))
        # print(summed*wid,line['words'])

        if (summed*wid) > 700:
            if data[i-1]['header']:
                data[i-1]['words'] = data[i-1]['words'] + line['words']
                data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
                del data[i]
            else:
                max_common = 0
                # print(line['words'],line['org_box'][0],line['org_box'][1],line['org_box'][0]+line['org_box'][2],line['org_box'][1]+line['org_box'][3])
                line['header'] = 1
                # print(line['words'])
                for head in head_pkl[0][itr]:
                    
                    common = iou(line['org_box'],[head['box'][0],head['box'][1]/1.01,head['box'][2] - head['box'][0],head['box'][3] - head['box'][1]/1.01])
                    # print(line['words'],head['word'])
                    # print(line['org_box'],[head['box'][0],head['box'][1],head['box'][2] - head['box'][0],head['box'][3] - head['box'][1]])
                    
                    if common > max_common:
                        print(line['words'],head['word'])
                        # print('I am in')
                        max_common = common
                        # print(head['word'])
                        line['words'] = head['word'].split()
                        # print(head['word'])
                        # print(line['org_box'],[head['box'][0],head['box'][1]/1.01,head['box'][2] - head['box'][0],head['box'][3] - head['box'][1]])
                        # print(common)
                # assert False

def findContourWidth(crop):
    gray_image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,10,255,cv2.THRESH_BINARY_INV)
    cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    widths = []
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, False)
        if area > 1 and perimeter > 1:
            x,y,w,h = cv2.boundingRect(c)
            widths.append(w)

    return widths


def findSubHeader(data,image,i,save,out_dir):
    img = image.copy()
    image = findSubHeaderblob(image)
    h,w,_ = image.shape
    # cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(i)+'_subheader.jpg')),constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        if line['amount'] or line['header']:
            continue
        b = line['box']
        crop = image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
        _crop = img[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
        if crop.size == 0:
            continue
        summed = np.sum(crop)/(crop.size)
        # _summed = np.sum(_crop)/(_crop.size)
        widths = findContourWidth(_crop)
        wid = np.mean(widths) + (6*np.std(widths))
        # print(summed*wid,line['words'])
        if (summed*wid) > 2000:
            if data[i-1]['sub']:
                data[i-1]['words'] = data[i-1]['words'] + line['words']
                data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
                del data[i]
            else:
                line['sub'] = 1

def findHeaderblob(image):
    image = 255-image
    kernel = np.ones((5,6),np.uint8)
    image= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=1)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=2)
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=2)

    return image

def findSubHeaderblob(im):
    im = 255-im
    kernel = np.ones((5,6),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=1)
    kernel = np.ones((4,37),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel,iterations=4)
    kernel = np.ones((3,7),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=4)  

    return im

def findType(img):
    threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 9)
    kernel = np.ones((7,3), dtype=np.uint8)
    threshed = cv2.erode(threshed, kernel, iterations=3)
    threshed = cv2.dilate(threshed, kernel, iterations=3)
    kernel = np.ones((1,3), dtype=np.uint8)
    threshed = cv2.dilate(threshed, kernel, iterations=1)
    
    ret, labels = cv2.connectedComponents(255-threshed)

    left = labels[:, 10:30]
    left = left[left!=0]
    right = labels[:, -30:-10]
    right = right[right!=0]

    max_freq_left, max_freq_right = 0, 0
    if left.shape[0] > 0:
        (values,counts) = np.unique(left, return_counts=True)
        max_freq_left = np.amax(counts)
        max_freq_label_left = values[np.argmax(counts)]
    if right.shape[0] > 0:
        (values,counts) = np.unique(right, return_counts=True)
        max_freq_right = np.amax(counts)
        max_freq_label_right = values[np.argmax(counts)]

    if max_freq_left == 0 and max_freq_right == 0:
        print("Label not found ")
        return -1

    mask = labels!=0
    selected_component = -1
    if max_freq_left > max_freq_right:
        indices = np.where(labels==max_freq_label_left)
        crop = img[np.amin(indices[0]): np.amax(indices[0]), np.amin(indices[1]): np.amax(indices[1])].transpose()
        mask = mask[np.amin(indices[0]): np.amax(indices[0]), np.amin(indices[1]): np.amax(indices[1])].transpose()
        crop = np.flip(crop, axis=1)
        mask = np.flip(mask, axis=1)
    else:
        indices = np.where(labels==max_freq_label_right)
        crop = img[np.amin(indices[0]): np.amax(indices[0]), np.amin(indices[1]): np.amax(indices[1])].transpose()
        mask = mask[np.amin(indices[0]): np.amax(indices[0]), np.amin(indices[1]): np.amax(indices[1])].transpose()
        crop = np.flip(crop, axis=0)
        mask = np.flip(mask, axis=0)

    ret,crop = cv2.threshold(crop,200,255,cv2.THRESH_BINARY_INV)
    # cv2.imshow('',crop)
    # cv2.waitKey(0)
    ocr = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, config = '--psm 8')
    
    _type = ''.join(ocr['text'])
    # print('detected',_type)
    error = [error_rate(_type,typ) for typ in types]
    return (np.argmin(error))

def draw(data,img):
    for line in data:
        if line['words']:
            if line['header']:
                b = line['box']
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,0, 255), 3)
            elif line['amount']:
                b = line['box']
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 0), 3)
            elif line['sub']:
                b = line['box']
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255,0, 0), 3)
            else:
                b = line['box']
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255,255, 0), 3)

def temp_draw(head_pkl,img,itr):

    for line in head_pkl[0][itr]:
        if line['word']:
            if line['header']:
                b = line['box']
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,255, 255), 3)
            # elif line['amount']:
            #     b = line['box']
            #     cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 255, 0), 3)
            # elif line['sub']:
            #     b = line['box']
            #     cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255,0, 0), 3)
            # else:
            #     b = line['box']
            #     cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255,255, 0), 3)


def data2excel (data,file):
    global num
    startrow = num
    excel_data = []
    amount_reg = re.compile(r'\d+')
    special_one = [']','|','[','H']
    special_zero = ['o','O','C']
    for line in data:
        lin['Page'] = line['page']
        lin['Type'] = line['type']
        if line['header']:
            lin['Manufacturer'] = ' '.join(line['words'])
        if line['sub']:
            lin['Model'] = ' '.join(line['words'])
        if line['amount']:
            if len(line['words'])<2 or not (line['words'][0][:2].isnumeric()):
                continue
            lin['High'] = 0                              
            lin['Low'] = 0                              
            lin['Model_year'] = line['words'][0]
            amount_line = line['words']
            amount_line.reverse()
            for i,word in enumerate(amount_line):
                if '$' in word:
                    # print(word)
                    if not lin['High']:
                        price = word.split('$')
                        lin['High'] = price[1]

                        for sp in special_one:
                            if sp in lin['High']:
                                if sp == 'H':
                                    lin['High'] = lin['High'].replace(sp,'11')
                                else:    
                                    lin['High'] = lin['High'].replace(sp,'1')

                        for sp in special_zero:
                            if sp in lin['High']:
                                lin['High'] = lin['High'].replace(sp,'0')

                        lin['High'] = lin['High'].replace('.','').replace(',','')
                        # print(lin['High'])
                        lin['High'] = int(''.join(amount_reg.findall(lin['High'])))
                    elif not lin['Low']:
                        price = word.split('$')
                        lin['Low'] = price[1]
                        if sp in lin['Low']:
                            lin['Low'].replace(sp,'1')
                        lin['Low'] = lin['Low'].replace('.','').replace(',','')
                        lin['Low'] = int(''.join(amount_reg.findall(lin['Low'])))
                        if i < len(line['words']) - 1:
                            features = line['words'][i+1:len(line['words'])-i]
                            features.reverse()
                            lin['Features'] = ' '.join(features)
                        else:
                            lin['Features'] = ''
                        lin['Features'] = lin['Features'] + price[0]
                        lin['Features'] = re.sub(r'[^a-zA-Z&\[\]\d\s^$"]', '', str(lin['Features'])).strip()
                        
            excel_data.append(lin.copy())
            num += 1
    df = pd.DataFrame(excel_data)
    if not df.empty:
        col = ['Page','Type','Manufacturer','Model','Model_year','Features','Low','High','Year']
        df = df[col]
        print(df)
        excelWrite(df,filename = file)



def readpickle(path):
    objects = []
    try:
        with (open(path, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
    except:
        objects = None
    return objects

temp_num = 0
type_num = -1
check_num = -1


def sortByColumn(image,columns,save,file,out_dir,itr,factor):

    im = image.copy()

    global temp_num
    global type_num
    global check_num

    h,w = image.shape
    image = image[int(0.054*h):,:]
    cut = 0
    j = 0
    columns = np.sort(columns)
    col_num = 0


    while j < len(columns) - 1:
        
        col = columns[j]
        # print(col)
        if col > 10:
            start = col - 10
        elif col < 5:
            start = col + 5
        else:
            start = col

        end = columns[j+1]
        width = end -start
        j += 1

        if width < 0.1*w:    
            continue 
        if width > 0.4*w:
            temp_num = 1
            continue

        if temp_num:
            type_num = findType(im)
            print(type_num)
            if type_num == -1:
                check_num += 1
                type_num = check_num
            if type_num > -1:
                temp_num = 0

        print(types[type_num])

        img = image[:,start:end]
        # cv2.imshow('sdg',constant_aspect_resize(img, width=None, height=600))
        img = removeLine(img)        
        if img is None:
            continue

        cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(col_num)+'.jpg')),constant_aspect_resize(img, width=None, height=700))
        vert = verticalProj(img.copy())
        vert = np.sort(vert)
        horz = horizontalProj(img.copy())

        try:                      
            if vert[0] <= 15:
                pass
            else:
                img = img[:,vert[0]-15:]
        except:
            pass
        try:
            if cut == 0:
                cut = horz[-1]
            img = img[:cut-5,:]
        except:
            pass

        # cv2.imshow('',constant_aspect_resize(img, width=None, height=600))
        # cv2.waitKey(0)
        # continue

        ocr = apply_ocr(os.path.join(out_dir,'OCR_new',str(save)+'_'+str(col_num)),img)
    
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        data = []
        for i in tqdm(range(len(ocr['level']))):
            if ocr['level'][i] == 4:
                flag = 0
                line = {}
                line['words'] = []
                line['amount'] = 0
                line['header'] = 0
                line['sub'] = 0
                line['type'] = types[type_num]
                line['page'] = save
                rect = [
                    ocr['left'][i],
                    ocr['top'][i],
                    ocr['width'][i],
                    ocr['height'][i],
                ]
                org_rect = [
                    (ocr['left'][i]+col)*factor,
                    (ocr['top'][i]+int(0.054*h))*factor,
                    (ocr['width'][i])*factor,
                    (ocr['height'][i])*factor,
                ] 
                line['org_box'] = org_rect
                line['box'] = rect
                data.append(line) 
            if ocr['level'][i] == 5:
                line['words'].append(ocr['text'][i])

        
        head_pkl = readpickle(os.path.join(out_dir,file.split('/')[-1].split('.')[0]+'.pickle'))
        # print(data)
        findamount(data,end - start)
        findHeader(data,img,j,save,out_dir,head_pkl,itr)
        findSubHeader(data,img,j,save,out_dir)
        draw(data,img)
        # temp_draw(head_pkl,img,itr)
        cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(col_num)+'_'+str(col_num)+'_.jpg')),constant_aspect_resize(img, width=None, height=600))
        col_num+=1
        data2excel(data,file)


def process(path,out_dir,file,name,i):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    height, width = gray.shape
    img = constant_aspect_resize(gray)
    factor = float(width) / 2500
    h,w = img.shape
    columns = verticalProj(img.copy())
    sortByColumn(img.copy(),columns,name,file,out_dir,i,factor)


if __name__== "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--debug', help = 'Enable debug', action='store_true')
    parser.add_argument('-i','--input_dir', help = 'Directory containing images')
    parser.add_argument('-f','--excel_name', help = 'Output excel filename')
    parser.add_argument('-o',"--out_dir", help = "Directory of evaluation output", default = './')
    parser.add_argument('-s',"--specific_page", help = "Path to a specific image to be processed")
    parser.add_argument('-y',"--year", help = "Year, in which the magazine was published in")

    args = parser.parse_args()

    if not len(sys.argv) > 1 :
        print ('No input has been provided')
    else:
        lin['Year'] = args.year
        debug = args.debug
        out_dir = args.out_dir
        file = os.path.join('/home/umaid/Experiments/guitar/GMag2Data/new_processed_books/new_excels',args.excel_name)
        os.makedirs(os.path.join(out_dir,'processed'),exist_ok = True)
        os.makedirs(os.path.join(out_dir,'OCR_new'),exist_ok = True)

        i = 0
        if args.specific_page:
            img_path = args.specific_page
            name = img_path.split('/')[-1].split('_')[-1].split('.')[0]
            process(img_path,out_dir,file,name)
        else:
            root = args.input_dir
            images = os.listdir(root)
            images.sort(key = lambda i: int((i.split('_')[-1]).split('.')[0]))

            i = 0 
            # i = 544  - 30
            # i = 22
            while i < len(images):
                name = images[i] 
                print(images[i])
                path = os.path.join(root,name)
                print(path)
                process(path,out_dir,file,str(name.split('_')[-1].split('.')[0]),i)
                i+=1