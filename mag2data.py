import pytesseract
import cv2
import os
import sys
# from PIL import Image
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
                print(bb(a,b),data[i]['words'],data[j]['words'])
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
        # print(line['words'])
        # if not line['amount']:
        #     i+=1
        #     continue
        dollor_num = 0
        for elem in line['words']:
            if '$' in elem:
                dollor_num += 1
            if dollor_num > 1:
                line['amount'] = 1
                if line['box'][0] > 0.07*w:
                    # print(line['words'])
                    # print(data[i-1]['words'])
                    data[i-1]['words'] = data[i-1]['words'] + line['words']
                    data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3] + 10
                    data[i-1]['box'][2] = (line['box'][0]+line['box'][2]) - data[i-1]['box'][0]   
                    data[i-1]['amount'] = 1
                    del data[i]
                    i -= 2
        i+=1

def findHeader(data,image,i,save,out_dir):
    image = findHeaderblob(image)
    cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(i)+'_header.jpg')),constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        b = line['box']
        summed = np.sum(image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]])
        # print(summed,line['words'])
        if summed > 1000000:
            # print(data)
            # if line['words'][0].isspace() or line['words'][0] == '':
            #     continue
            if data[i-1]['header']:
                data[i-1]['words'] = data[i-1]['words'] + line['words']
                data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
                del data[i]
            else:
                line['header'] = 1                

def findSubHeader(data,image,i,save,out_dir):
    image = findSubHeaderblob(image)
    h,w = image.shape
    cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(i)+'_subheader.jpg')),constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        if line['amount'] or line['header']:
            continue
        b = line['box']
        crop = image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
        if crop.size == 0:
            continue
        summed = np.sum(crop)/(crop.size)
        try:
            # print(summed,line['words'])
            if summed > 20 and ((data[i+1]['box'][0]) > 0.01*w or data[i+1]['amount']) :
                line['sub'] = 1
                if data[i-1]['amount'] or data[i-1]['header'] or data[i-1]['box'][0] > 0.01*w:
                    continue
                check_b = data[i-1]['box']
                check_crop = image[check_b[1]:check_b[1]+check_b[3],check_b[0]:check_b[0]+check_b[2]]
                if np.sum(check_crop)/np.size(check_crop) > 20 :
                    data[i-1]['words'] = data[i-1]['words'] + line['words']
                    data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
                    data[i-1]['sub'] = 1
                    del data[i]
        except:
            pass
            # if data[i-1]['sub']:
            #   data[i-1]['words'] = data[i-1]['words'] + line['words']
            #   data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
            #   del data[i]
            # else:
            #   line['sub'] = 1

    # paras = []
    # kk = 0
    # gaps = sorted(gaps)
    # for k in range(len(gaps)-1):
    #     start = gaps[k]
    #     end = gaps[k+1]
    #     para = []
    #     while kk < len(data):
    #         line_y = data[kk]['box'][1]
    #         if line_y > start and line_y < end:
    #             para.append(data[kk])
    #         else:
    #             break
    #         kk+=1
    #     paras.append(para)
    # for para in paras:
    #     if len(para):
    #         para[0]['header'] = 1
    #         if len(para) > 1:                    
    #             if para[0]['box'][2] > para[1]['box'][2] :
    #                 para[0]['words'] = para[0]['words'] + para[1]['words']
    #                 para[0]['box'][3] = para[0]['box'][3] + para[1]['box'][3]



def findHeaderblob(image):
    image = 255-image
    kernel = np.ones((5,6),np.uint8)
    image= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=1)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=2)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    # _,image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = np.ones((3,3),np.uint8)
    # # kernel = np.ones((7,7),np.uint8)
    # # image = cv2.erode(image,kernel,iterations = 1)
    kernel = np.ones((5,5),np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=2)
    # cv2.imshow('header',constant_aspect_resize(image, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()

    return image

def findSubHeaderblob(im):
    # cv2.imshow('firsr',constant_aspect_resize(im, width=None, height=768))
    im = 255-im
    kernel = np.ones((5,6),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=1)
    # cv2.imshow('firsr',constant_aspect_resize(im, width=None, height=768))
    # cv2.imshow('first',constant_aspect_resize(im, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()
    kernel = np.ones((4,37),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel,iterations=5)
    # cv2.imshow('second',constant_aspect_resize(im, width=None, height=768))
    kernel = np.ones((3,7),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=4)  
    # cv2.imshow('third',constant_aspect_resize(im, width=None, height=768))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im,(5,5),0)
    _,im = cv2.threshold(blur,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow('final',constant_aspect_resize(im, width=None, height=768))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

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
        # print(lin['Manufacturer'])
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
        col = ['Page','Manufacturer','Model','Model_year','Features','Low','High']
        df = df[col]
        print(df)
        excelWrite(df,filename = file)

def sortByColumn(image,columns,save,file,out_dir):
    h,w= image.shape
    # image = image[int(0.07*h):int(0.95*h),:]
    image = image[int(0.106*h):int(0.93*h),:]
    # cv2.imshow('',constant_aspect_resize(image, width=None, height=600))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()
    # col_data = []
    for j,col in enumerate(columns):
        start = col - 20
        if j+1 == len(columns):
            end = w 
        else:
            end = columns[j+1]
        if (end - start) < 0.1*w:
            continue 
        img = image[:,start:end]

        # cv2.imshow(str(j),constant_aspect_resize(img, width=None, height=768))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()        

        # gaps = horizontalProj(img.copy())
        # if list(gaps):
        #     if gaps[0] > 0.33*h:
        #         continue
        #     img = img [gaps[0]:,:]
        # else:
        #     continue
        vert = verticalProj(img.copy())
        try:
            if vert[0] <= 15:
                continue
            else:
                img = img[:,vert[0]-15:]
        except:
            pass
        # cv2.line(img,(int(0.01*w),0),(int(0.01*w),h),(0,0,255),3)
        # # print(len(gaps),'gaps')
        # # if len(gaps) < 10 :
        # #     continue
        # # cv2.line(img,(int(0.02*w),0),(int(0.02*w),h),(0,0,255),3)
        # for gap in vert:
        #     cv2.line(img,(int(gap),0),(int(gap),h),(0,0,255),3)
        # cv2.imshow(str(j),constant_aspect_resize(img, width=None, height=768))
        # # cv2.imshow('prcess_'+str(j),constant_aspect_resize(img, width=None, height=768))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
      # im = check(img)
    #   cv2.imshow(str(j),constant_aspect_resize(im, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()
    # assert False

        # conf = """--oem 0 -c tessedit_char_blacklist=|_*! """
        # ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,config=conf)
        ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config = '--psm 4')
        value = [ocr['height'][i] for i,level in enumerate(ocr['level']) if level == 4]
        if value:
            value = max(value)
            if value > 0.1*h:
                continue
        else:
            continue
            
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
                # if flag:
                #     # print(ocr['text'][i],ocr['width'][i],ocr['conf'][i],end-start)
                #     continue
                if ocr['width'][i] > 0.7 * (end-start) and not flag and ocr['conf'][i] > 70:
                    # print(ocr['text'][i],ocr['width'][i],ocr['conf'][i],end-start)
                    flag = 1
                    del data[-1]
                else:
                # line['words'].append(ocr['text'][i])
                    data[-1]['words'].append(ocr['text'][i])

                # if '$' in ocr['text'][i] or flag:
                # if '$' in ocr['text'][i]:
                #     if flag:
                #         line['amount'] = 1
                #     else:
                #         flag = 1 
                    # if len(line['words'])<2:
                    #     continue
                    # print('first line',line['words'])
                    # if line['box'][0] > 0.05*(end - start) and flag:
                    #     print('previous line',data[-2]['words'])
                    #     data[-2]['words'] = data[-2]['words'] + line['words']
                    #     data[-2]['box'][3] = data[-2]['box'][3] + line['box'][3] +5
                    #     data[-2]['box'][2] = (line['box'][0]+line['box'][2]) - data[-2]['box'][0]   
                    #     data[-2]['amount'] = 1
                    #     del data[-1]
                    #     print('second line',data[-1]['words'])
                    #     if data[-1]['box'][0] > 0.05*(end - start):
                    #         data[-2]['words'] = data[-2]['words'] + data[-1]['words']
                    #         data[-2]['box'][3] = data[-2]['box'][3] + data[-1]['box'][3] + 5
                    #         data[-2]['box'][2] = (data[-1]['box'][0]+data[-1]['box'][2]) - data[-2]['box'][0]   
                    #         data[-2]['amount'] = 1
                    #         del data[-1]
                    # else:
                    #     flag = 1
                    #     line['amount'] = 1
                    # print(data[-1])
        # cv2.imshow(str(j), constant_aspect_resize(img, width=None, height=700))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        # assert False
        # w_ = end - start
        # data = line_align(data,end - start)
        # cv2.line(img,(int(0.05*w_),0),(int(0.05*w_),h),(0,0,255),3)
        # cv2.imshow(str(j),constant_aspect_resize(img, width=None, height=768))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        # for line in data:
        #     print(line['words'])
        # for line in data:
        #     b = line['box']
        #     cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,0, 255), 3)
        # cv2.imshow(str(j), constant_aspect_resize(img, width=None, height=700))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        # # assert False
        # print(data)
        # assert False
        # data = line_align(data,end-start)
        # for line in data:
        #     print(line['words'])
        # assert False
        findamount(data,end - start)
        findHeader(data,img,j,save,out_dir)
        findSubHeader(data,img,j,save,out_dir)
        draw(data,img)
        data2excel(data,file)
        cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(j)+'.jpg')),constant_aspect_resize(img, width=None, height=700))


        # cv2.line(img,(int(0.015*w),0),(int(0.015*w),h),(0,0,255),3)
    #   cv2.imshow(str(j),constant_aspect_resize(img, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()


def process(path,out_dir,file,name):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = constant_aspect_resize(gray)
    h,w = img.shape
    columns = verticalProj(img.copy())
    # for col in columns:
    #     cv2.line(img,(col,0),(col,h),(0,0,255),2)
    # cv2.imshow('sdgd',constant_aspect_resize(img, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()

    sortByColumn(img.copy(),columns,name,file,out_dir)


if __name__== "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--debug', help = 'Enable debug', action='store_true')
    parser.add_argument('-i','--input_dir', help = 'Directory containing images')
    parser.add_argument('-f','--excel_name', help = 'Output excel filename')
    parser.add_argument('-o',"--out_dir", help = "Directory of evaluation output", default = './')
    parser.add_argument('-s',"--specific_page", help = "Path to a specific image to be processed")

    args = parser.parse_args()

    if not len(sys.argv) > 1 :
        print ('No input has been provided')
    else:
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
