import pytesseract
import cv2
import os
import sys
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm
from utils import iou,bb
from excelWriter import excelWrite

pd.options.display.width = 0

lin = {}
lin['Page'] = ''
lin['Manufacturer'] = ''
lin['Model'] = ''
lin['Model_year'] = ''
lin['Features'] = ''
lin['Low'] = ''
lin['High'] = ''
lin['Year'] = ''

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

def removeLine(img):
    _h,_w = img.shape
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,10,255,cv2.THRESH_BINARY_INV)
    cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    widths = {}
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 0.1*(_w*_h):
            return None
        perimeter = cv2.arcLength(c, False)
        if area > 1 and perimeter > 1:
            x,y,w,h = cv2.boundingRect(c)
            widths[w] = [x,y,w,h]
    # print(widths.keys())
    # image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for i in widths:
    #     x,y,w,h = widths[i]
    #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    #     cv2.imshow('header',constant_aspect_resize(image, width=None, height=768))
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #       cv2.destroyAllWindows()
    max_wid = max(widths.keys()) 
    if max_wid > 0.7 * w:
        # print(widths[max_wid])
        x,y,w,h = widths[max_wid]
        img[y-3:y+h+3,x-3:x+w+3] = 255
    return img

def removeHeader(data):
    def common_member(a, b): 
        a_set = set(a) 
        b_set = set(b) 
        if len(a_set.intersection(b_set)) > 0: 
            return(True)  
        return(False)

    a = ['YEAR', 'FEATURES', 'LOW', 'HIGH','MODEL', 'EXC.' 'COND. ', 'EXC. ', 'COND.','Low','High']
    i=0
    while i < len(data):
        line = data[i]
        if len(line['words']):
            if common_member(a,line['words']) or line['words'][0].isspace() or line['words'][0] == '':
                del data[i]
                i-=1 
        i+=1

    
def verticalProj(image):    
    image = 255 - image
    h,w = image.shape
    image = image[int(0.07*h):int(0.95*h),:]
    proj = np.sum(image,0) 
    brk = np.where(proj==0)
    cha = (np.diff(brk[0])>1)
    final = np.where(cha ==True)
    return brk[0][final[0]]

def horizontalProj(image):    
    image = 255 - image
    h,w = image.shape
    proj = np.sum(image,1) 
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
    while i < len(data):
        line = data[i]
        dollor_num = 0
        for elem in line['words']:
            if '$' in elem:
                dollor_num += 1
            if dollor_num > 1:
                line['amount'] = 1
                if line['box'][0] > 0.07*w:
                    data[i-1]['words'] = data[i-1]['words'] + line['words']
                    data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3] + 10
                    data[i-1]['box'][2] = (line['box'][0]+line['box'][2]) - data[i-1]['box'][0]   
                    data[i-1]['amount'] = 1
                    del data[i]
                    i -= 2
        i+=1

def findHeader(data,image,i,save,out_dir):
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
                line['header'] = 1

        # summed = np.sum(image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]])

        # print(summed,line['words'])
        # if summed > 900000:
        #     if data[i-1]['header']:
        #         data[i-1]['words'] = data[i-1]['words'] + line['words']
        #         data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
        #         del data[i]
        #     else:
        #         line['header'] = 1

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
        wid = np.mean(widths) + (3*np.std(widths))
        # print(summed*wid,line['words'])

        if (summed*wid) > 600:
            if data[i-1]['sub']:
                data[i-1]['words'] = data[i-1]['words'] + line['words']
                data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
                del data[i]
            else:
                line['sub'] = 1
        # try:
        #     if summed > 20 and ((data[i+1]['box'][0]) > 0.01*w or data[i+1]['amount']) :
        #         line['sub'] = 1
        #         if data[i-1]['amount'] or data[i-1]['header'] or data[i-1]['box'][0] > 0.01*w:
        #             continue
        #         check_b = data[i-1]['box']
        #         check_crop = image[check_b[1]:check_b[1]+check_b[3],check_b[0]:check_b[0]+check_b[2]]
        #         if np.sum(check_crop)/np.size(check_crop) > 20 :
        #             data[i-1]['words'] = data[i-1]['words'] + line['words']
        #             data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
        #             data[i-1]['sub'] = 1
        #             del data[i]
        # except:
        #     pass

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
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel,iterations=5)
    kernel = np.ones((3,7),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=4)  
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(im,(5,5),0)
    # _,im = cv2.threshold(blur,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    return im

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


def data2excel (data,file):
    global num
    startrow = num
    excel_data = []
    for line in data:
        lin['Page'] = line['page']
        if line['header']:
            lin['Manufacturer'] = ' '.join(line['words'])
        if line['sub']:
            lin['Model'] = ' '.join(line['words'])
        if line['amount']:
            if len(line['words'])<2 or not (line['words'][0][:2].isnumeric()):
                continue
            lin['Low'] = 0
            lin['Model_year'] = line['words'][0]
            for i,word in enumerate(line['words']):
                if '$' in word:
                    if not lin['Low']:
                        lin['Low'] = word
                        if 1 != i:
                            lin['Features'] = ' '.join(line['words'][1:i])
                        else:
                            lin['Features'] = ''
                    else:
                        lin['High'] = word
            excel_data.append(lin.copy())
            num += 1
    df = pd.DataFrame(excel_data)
    if not df.empty:
        col = ['Page','Manufacturer','Model','Model_year','Features','Low','High','Year']
        df = df[col]
        print(df)
        excelWrite(df,filename = file)

def sortByColumn(image,columns,save,file,out_dir):
    h,w= image.shape
    image = image[int(0.07*h):int(0.955*h),:]
    for j,col in enumerate(columns):
        start = col - 20
        if j+1 == len(columns):
            end = w 
        else:
            end = columns[j+1]
        if (end - start) < 0.1*w:
            continue 
        img = image[:,start:end]
        img = removeLine(img)
        if img is None:
            continue
        vert = verticalProj(img.copy())
        try:
            if vert[0] <= 15:
                continue
            else:
                img = img[:,vert[0]-15:]
        except:
            pass
        ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config = '--psm 4')
        # value = [ocr['height'][i] for i,level in enumerate(ocr['level']) if level == 4]
        # if value:
        #     value = max(value)
        #     if value > 0.1*h:
        #         continue
        # else:
        #     continue
            
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
                line['page'] = save
                rect = [
                    ocr['left'][i],
                    ocr['top'][i],
                    ocr['width'][i],
                    ocr['height'][i],
                ]
                line['box'] = rect
                data.append(line) 
            if ocr['level'][i] == 5:
                # print(ocr['text'][i],ocr['conf'][i],ocr['width'][i],0.7 * (end-start))
                # print(data[-1])
                # if ocr['width'][i] > 0.7 * (end-start) and not flag and ocr['conf'][i] > 70:
                # print(ocr['text'][i])
                # if ocr['width'][i] > 0.7 * (end-start) and not flag:
                #     print(ocr['text'][i],ocr['width'][i],(end-start))
                # # if ocr['width'][i] > 0.7 * (end-start) and ocr['conf'][i] > 70:
                #     flag = 1
                #     # print(ocr['text'][i])
                #     del data[-1]
                #     # continue
                # else:
                    # try:
                        # print(data[-1])
                line['words'].append(ocr['text'][i])
                    # except:
                    #     pass
        # for line in data:
        #     print(line['words'])
        removeHeader(data)
        findamount(data,end - start)
        findHeader(data,img,j,save,out_dir)
        findSubHeader(data,img,j,save,out_dir)
        draw(data,img)
        data2excel(data,file)
        cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(j)+'.jpg')),constant_aspect_resize(img, width=None, height=700))


def process(path,out_dir,file,name):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = constant_aspect_resize(gray)
    h,w = img.shape
    columns = verticalProj(img.copy())
    sortByColumn(img.copy(),columns,name,file,out_dir)


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
        file = os.path.join(out_dir,args.excel_name)
        os.makedirs(os.path.join(out_dir,'processed'),exist_ok = True)
        i = 0
        if args.specific_page:
            img_path = args.specific_page
            name = img_path.split('/')[-1].split('_')[-1].split('.')[0]
            process(img_path,out_dir,file,name)
        else:
            root = args.input_dir
            images = os.listdir(root)
            images.sort(key = lambda i: int((i.split('_')[-1]).split('.')[0]))
            # i = 24 
            i = 0           
            while i < len(images):
                name = images[i] 
                print(images[i])
                path = os.path.join(root,name)
                process(path,out_dir,file,str(name.split('_')[-1].split('.')[0]))
                i+=1
