import pytesseract
import cv2
import os
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from excelWriter import excelWrite

pd.options.display.width = 0

lin = {}
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
    image = image[int(0.05*h):int(0.95*h),:]
    # cv2.imshow('',constant_aspect_resize(image, width=None, height=700))
    proj = np.sum(image,0) 
    brk = np.where(proj==0)
    cha = (np.diff(brk[0])>1)
    final = np.where(cha ==True)
    return brk[0][final[0]]

# def horizontalProj(image):    
#     image = 255 - image
#     h,w = image.shape
#     kernel = np.ones((35,20),np.int8)
#     image = cv2.dilate(image, kernel)
#     proj = np.sum(image,1) 
#     brk = np.where(proj==0)
#     cha = (np.diff(brk[0])>1)
#     final = np.where(cha ==True)
#     if len(final[0]):
#         gaps = np.insert(brk[0][final[0]],-1,brk[0][-1])
#     else:
#         gaps = []
#     return gaps

def findHeader(data,image,i,save):
    image = findHeaderblob(image)
    cv2.imwrite('output/'+save+'_'+str(i)+'_header.jpg',constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        b = line['box']
        summed = np.sum(image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]])
        if summed > 12000:
            if line['words'][0].isspace() or line['words'][0] == '':
                continue
            if data[i-1]['header']:
                data[i-1]['words'] = data[i-1]['words'] + line['words']
                data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
                del data[i]
            else:
                line['header'] = 1

def findSubHeader(data,image,i,save):
    image = findSubHeaderblob(image)
    h,w = image.shape
    cv2.imwrite('output/'+save+'_'+str(i)+'_subheader.jpg',constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        if line['amount']:
            continue
        b = line['box']
        crop = image[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
        if crop.size == 0:
            continue
        summed = np.sum(crop)/(crop.size)
        try:
            if summed > 20 and not line['header'] and ((data[i+1]['box'][0]) > 0.015*w or data[i+1]['amount']) :
                line['sub'] = 1
                if data[i-1]['amount'] or data[i-1]['header'] :
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
    kernel = np.ones((3,3),np.uint8)
    image = cv2.erode(image,kernel,iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=4)
    return image

def findSubHeaderblob(im):
    im = 255-im
    kernel = np.ones((5,6),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=1)
    kernel = np.ones((5,33),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel,iterations=6)
    kernel = np.ones((3,7),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel,iterations=4)  
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im,(5,5),0)
    _,im = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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


def data2excel (data):
    global num
    startrow = num
    excel_data = []
    for line in data:
        # print(line)
        if line['header']:
            lin['Manufacturer'] = ' '.join(line['words'])
        if line['sub']:
            lin['Model'] = ' '.join(line['words'])
        if line['amount']:
            # print(len(line['words']))
            if len(line['words'])<2:
                continue
            lin['Low'] = 0
            if not (line['words'][0][:2].isnumeric()):
                continue
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
        col = ['Manufacturer','Model','Model_year','Features','Low','High']
        df = df[col]
        print(df)
        excelWrite(df)

def sortByColumn(image,columns,save):
    h,w= image.shape
    image = image[int(0.105*h):int(0.95*h),:]
    col_data = []
    for j,col in enumerate(columns):
        start = col
        if j+1 == len(columns):
            end = w 
        else:
            end = columns[j+1]
        if (end - start) < 0.1*w:
            continue 
        img = image[:,start:end]
    #   im = check(img)
    #   cv2.imshow(str(j),constant_aspect_resize(im, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()
    # assert False
        ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        value = [ocr['height'][i] for i,level in enumerate(ocr['level']) if level == 4]
        if value:
            value = max(value)
            if value > 0.33*h:
                continue
        else:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        data = []
        for i in tqdm(range(len(ocr['level']))):
            if ocr['level'][i] == 4:
                line = {}
                line['words'] = []
                line['amount'] = 0
                line['header'] = 0
                line['sub'] = 0
                rect = [
                    ocr['left'][i],
                    ocr['top'][i],
                    ocr['width'][i],
                    ocr['height'][i],
                ]
                line['box'] = rect
                # b = rect
                # cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0,0, 255), 3)
                data.append(line)
                flag = 0           
            if ocr['level'][i] == 5:
                line['words'].append(ocr['text'][i])
                if '$' in ocr['text'][i] or flag:
                    if line['box'][0] > 0.05*(end - start) and flag:
                        data[-2]['words'] = data[-2]['words'] + line['words']
                        data[-2]['box'][3] = data[-2]['box'][3] + line['box'][3] +5
                        data[-2]['box'][2] = (line['box'][0]+line['box'][2]) - data[-2]['box'][0]   
                        data[-2]['amount'] = 1
                        del data[-1]
                        if data[-1]['box'][0] > 0.05*(end - start):
                            data[-2]['words'] = data[-2]['words'] + data[-1]['words']
                            data[-2]['box'][3] = data[-2]['box'][3] + data[-1]['box'][3] +5
                            data[-2]['box'][2] = (data[-1]['box'][0]+data[-1]['box'][2]) - data[-2]['box'][0]   
                            data[-2]['amount'] = 1
                            del data[-1]
                    else:
                        flag = 1
                        line['amount'] = 1
                    # print(data[-1])
        # cv2.imshow(str(j), constant_aspect_resize(img, width=None, height=700))
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        # assert False
        findHeader(data,img,j,save)
        findSubHeader(data,img,j,save)
        draw(data,img)
        data2excel(data)
        cv2.imwrite('output/'+save+'_'+str(j)+'.jpg',constant_aspect_resize(img, width=None, height=700))
        # cv2.line(img,(int(0.015*w),0),(int(0.015*w),h),(0,0,255),3)
    #   cv2.imshow(str(j),constant_aspect_resize(img, width=None, height=768))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()

os.makedirs('output',exist_ok = True)
i = 0
root = "./pdf_pages/"
images = os.listdir(root)
images.sort(key = lambda i: int(i.split('_')[-1]))
# i = 380-32s
while i < len(images):
    name = images[i] 
    print(images[i])
    path = os.path.join(root,name)
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img = constant_aspect_resize(gray)
    h,w = img.shape
    columns = verticalProj(img.copy())
    # for col in columns:
    #     image = cv2.line(image,(col,0),(col,h),(0,0,255))
    # cv2.imshow('sfew', constant_aspect_resize(image, width=None, height=700))
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # assert False
    sortByColumn(img.copy(),columns,name)
    i+=1
