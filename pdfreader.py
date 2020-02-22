import xml.etree.ElementTree as ET
import cv2
import os
import argparse
from pdf2image import convert_from_path
from mag2data import verticalProj,findHeaderblob 
import numpy as np
import pandas as pd
import pickle


# inp_dir = os.path.join(inp_dir,'xmls')

# def verticalProj(image):    
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   _,image = cv2.threshold(image,100,255,cv2.THRESH_BINARY_INV)
#   # image = 255 - image
#   h,w = image.shape
#   image = image[int(0.25*h):int(0.75*h),:]
#   proj = np.sum(image,0)
#   brk = np.where(proj==0)
#   cha = (np.diff(brk[0])>1)
#   cha = np.append(cha,False)
    
    # return brk[0][cha]

def column_sort(inps,cols,f):
    final = []
    for j,col in enumerate(cols):
        data = []
        for inp in inps:
            box = inp['bbox']
            if j == 0:
                if box[0][0]*f < col and box[0][0]*f > 0:
                    data.append(inp)
            elif box[0][0]*f < col and box[0][0]*f > cols[j-1]:
                data.append(inp)
        final.append(data)
    return final

def removeLine(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols = verticalProj(img)
    _h,_w = img.shape



# def findHeaderblob(image):
#   image = 255-image
#   kernel = np.ones((5,6),np.uint8)
#   image= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=1)
#   kernel = np.ones((2,2),np.uint8)
#   image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=2)
#   kernel = np.ones((5,5),np.uint8)
#   image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=2)
#   kernel = np.ones((5,5),np.uint8)
#   dilation = cv2.dilate(image,kernel,iterations = 1)
    

#   return image

def findContourWidth(crop):
    gray_image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY_INV)
    # cv2.imshow('sdg',thresh)
    cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    widths = []
    for c in cnts:
        area = cv2.contourArea(c)
        # print(area)
        perimeter = cv2.arcLength(c, False)
        if area > 1 and perimeter > 1:
            x,y,w,h = cv2.boundingRect(c)
            widths.append(w)

    return widths

def findHeader(data,image):
    img = image.copy()
    image = findHeaderblob(image)
    # cv2.imwrite(os.path.join(inp_dir,'processed',(save+'_'+str(i)+'_header.jpg')),constant_aspect_resize(image, width=None, height=700))
    for i,line in enumerate(data):
        line['header'] = 0
        b = line['box']
        crop = image[b[1]:b[3],b[0]:b[2]]
        _crop = img[b[1]:b[3],b[0]:b[2]]
        # cv2.imshow('',crop)
        # cv2.imshow('sdfwe',_crop)
        # cv2.waitKey(0)
        if crop.size == 0:
            continue
        summed = np.sum(crop)/(crop.size)
        widths = findContourWidth(_crop)
        # cv2.waitKey(0)
        wid = 5*np.mean(widths) + (7*np.std(widths))
        # print(summed*wid,line['word'])
        # print(line['word'],(summed*wid))
        if (summed*wid) > 3200:
            line['header'] = 1
            # if data[i-1]['header']:
            #     data[i-1]['words'] = data[i-1]['words'] + line['words']
            #     data[i-1]['box'][3] = data[i-1]['box'][3] + line['box'][3]
            #     del data[i]
            # else:
            #     line['header'] = 1

def mergeheader(words):
    for line in words:
        if line['header']:
            # print('first',line['word'])
            dots = line['box']
            rise = dots[3] - dots[1]
            i = 0
            while i < len(words):
                word = words[i]
                if (line['word'] == word['word']) or (not word['header']):
                    i+=1
                    continue
                s_dots = word['box']

                if np.abs(s_dots[1] - dots[1]) < 0.75*rise :
                    if (np.abs(s_dots[0] - dots[2]) < 0.1*rise) :
                        line['word'] = ''.join([line['word'].strip(),word['word'].strip()])
                        line['box'] = [min(dots[0],s_dots[0]),min(dots[1],s_dots[1]),max(dots[2],s_dots[2]),min(dots[3],s_dots[3])]
                        del words[i]
                        i-=1
                    elif np.abs(s_dots[0] - dots[2]) < 6*rise :
                        # print([line['word'].split(),word['word'].split()])
                        line['word'] = ' '.join([line['word'].strip(),word['word'].strip()])
                        line['box'] = [min(dots[0],s_dots[0]),min(dots[1],s_dots[1]),max(dots[2],s_dots[2]),min(dots[3],s_dots[3])]
                        del words[i]
                        i-=1
                i+=1


def topheadermerge(data):
    i = 0
    while i < len(data)-1:
        # if i == 0:
        #     i+=1
        #     continue
        if data[i+1]['header'] and data[i]['header']:

            dots = data[i]['box']
            s_dots = data[i+1]['box']
            data[i]['word'] = data[i]['word'] +' '+data[i+1]['word']
            data[i]['box']=[dots[0],dots[1],dots[2],(dots[3]+(s_dots[3] - s_dots[1]))] 
            del data[i+1]
        i+=1


def columns(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cols = verticalProj(img)
    cols = np.sort(cols)
    h,w,_ = image.shape

    j = 0
    out = (None,None)
    # print(len(cols))
    while j < len(cols) - 1:
        start = cols[j]
        end = cols[j+1]

        crop = image[:,start:end]
        _h,_w,_ = crop.shape

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(crop,200,255,cv2.THRESH_BINARY_INV)
        # cv2.imshow('sdfg',thresh)
        cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        widths = {}
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 0.05*(w*h):
                # print(j)
                # print(area)
                # print(start,end)
                if start > 0.5*w:
                    out =  (None,start)
                else:
                    out =  (end,None)

        # cv2.imshow('',cv2.resize(crop,(700,300)))
        # cv2.waitKey(0)
        j+=1
    return out

def removegarbage(words,start,end):

     i=0
     while i < len(words):
        box = words[i]['box']
        if words[i]['header']:
            # print(words[i]['word'],box,start,end)
            if start:
                start = start-15
                if box[0] < start:
                    words[i]['header']=0
            elif end:
                end = end+15
                if box[0] > end:
                    words[i]['header']=0
        i+=1
headers = []

def pdfreader(root):
    j = 1
    for i in range(2,568):
    # for i in (0,1):

        # j = 

        # print(image_dir)
        xml_path = os.path.join(out_dir,'xmls',"page_"+str(j)+".xml")
        img_path = os.path.join(image_dir,"page_"+str(j)+".jpg")

        if os.path.exists(img_path):
            image = cv2.imread(img_path)
        else:
            j+=1
            continue

        # print(img_path,xml_path)
        # print(xml_path)

        if not os.path.exists(xml_path):
            os.system("pdf2txt.py "+str(root)+" -p "+str(j)+" -o "+xml_path)
        print('\n\n\n'+'page_'+str(j))
        mytree = ET.parse(xml_path)

        # start,end = findsides(image)

        myroot = mytree.getroot()

        page = myroot.findall('.//page')
        image_size = page[0].attrib['bbox'].split(',')
        img_w = int(float(image_size[2]))
        img_h = int(float(image_size[3]))
        h,w,_ = image.shape
        # image = image[int(0.11*h):,:]
        f = w/img_w
        # print(f)
        fy = h/img_h
        h_= img_h
        words = []
        word = ''
        i = 0
        # id = 0
        textline = myroot.findall('.//text')
        for cha in textline:
            try:
                box = [x for x in cha.attrib['bbox'].split(',')]
                # print(h_-int(float(box[3])),int(0.11*h))
                if h_-int(float(box[3]))<int(0.024*h):
                    continue
                if i == 0:
                    p1 = (int(float(box[0])),h_-int(float(box[3])))
                    i+=1
                word+=cha.text
                font = cha.attrib['font']
                size = cha.attrib['size']
                if cha.text == ' ' :
                    temp = {}
                    temp['word'] = word
                    temp['font'] = font
                    # temp['box'] = [p1,p2]
                    temp['box'] = [int(p1[0]*f),int(p1[1]*1.01*fy),int(p2[0]*f),int(p2[1]*fy)]
                    temp['size'] = size
                    word = ''
                    words.append(temp)
                    i = 0
                p2 = (int(float(box[2])),h_-int(float(box[1])))
            except:
                continue

        findHeader(words,image)
        mergeheader(words)
        topheadermerge(words)
        # start,end = None,None
        # columns(image)
        # assert False
        # print(j)
        start,end = columns(image)
        removegarbage(words,start,end)
        # print(start,end)
        # if start:
        #     print('start')
        #     cv2.line(image,(start,0),(start,h),(0,0,255),3)
        # elif end:
        #     cv2.line(image,(end,0),(end,h),(0,0,255),3)
        # else:
        #     pass

        # cv2.imshow('',cv2.resize(image,(500,700)))
        # cv2.waitKey(0)


        # print(cols)

        # for col in cols:
        #     cv2.line(image,(col,0),(col,h),(0,0,255),3)
        # cv2.imshow('',cv2.resize(image,(500,700)))
        # cv2.waitKey(0)
        
        # j+=1
        # continue
        # print(words)
        
        for word in words:
            if word['header'] == 1:
                headers.append(word)
                print(word['word'])
                b = word['box']
                cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0,0, 255), 3)
        # assert False
        # print(os.path.join(inp_dir,'headers','page_'+str(i),'.jpg'))
        cv2.imwrite(os.path.join(out_dir,'headers','page_'+str(j)+'.jpg'),image)
        j+=1

    with open(os.path.join(out_dir,name+'.pickle'), 'wb') as f:
        pickle.dump(headers, f)
    # file = pd.DataFrame(headers)
    # file.column = 'header'
    # file.to_csv(os.path.join(out_dir,'pdfreaderHeaders.csv'))


        # cv2.imshow('',cv2.resize(image,(500,700)))
        # cv2.waitKey(0)
            # for word in line:
            #   print(word)

        # assert False
if __name__== "__main__" :

    parser = argparse.ArgumentParser()

    parser.add_argument('-r','--root', help = 'root directory for the book')
    parser.add_argument('-n','--name', help = 'name of the output file')
    parser.add_argument('-o','--out_dir', help = 'output directory')
    parser.add_argument('-i','--inp_dir', help = 'input directory for page images')


    # parser.add_argument('-n','--name', help = 'file name', default = '2020')
    args = parser.parse_args()
    
    name = args.name
    root = args.root
    inp_dir = args.inp_dir
    out_dir = args.out_dir

    image_dir = os.path.join(inp_dir,'pages')

    os.makedirs(os.path.join(out_dir,'xmls'),exist_ok = True)
    os.makedirs(os.path.join(out_dir,'headers'),exist_ok = True)
    
    pdfreader(root)