import xml.etree.ElementTree as ET
import cv2
import os
from pdf2image import convert_from_path
import numpy as np
import pandas as pd

name = '2020'
root = '/home/umaid/Experiments/guitar/GMag2Data/Books/Price\ Guide\ '+name+'\ EDITED.pdf'
out_dir = '/home/umaid/Experiments/guitar/GMag2Data/processed_books/'+name
image_dir = os.path.join(out_dir,'pages')

os.makedirs(os.path.join(out_dir,'xmls'),exist_ok = True)
out_dir = os.path.join(out_dir,'xmls')

def verticalProj(image):    
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_,image = cv2.threshold(image,10,255,cv2.THRESH_BINARY_INV)
	# image = 255 - image
	h,w = image.shape
	image = image[int(0.25*h):int(0.75*h),:]
	proj = np.sum(image,0)
	brk = np.where(proj==0)
	cha = (np.diff(brk[0])>1)
	cha = np.append(cha,False)
	
	return brk[0][cha]

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

def findHeaderblob(image):
	image = 255-image
	kernel = np.ones((5,6),np.uint8)
	image= cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=1)
	kernel = np.ones((2,2),np.uint8)
	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=2)
	kernel = np.ones((5,5),np.uint8)
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=2)

	return image

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
	# cv2.imwrite(os.path.join(out_dir,'processed',(save+'_'+str(i)+'_header.jpg')),constant_aspect_resize(image, width=None, height=700))
	for i,line in enumerate(data):
		line['header'] = 0
		b = line['box']
		# print(b)
		# continue
		# crop = image[b[1]*int(f):b[1]*int(f)+b[3]*int(f),b[0]*int(f):b[0]*int(f)+b[2]*int(f)]
		crop = image[b[1]:b[3],b[0]:b[2]]
		_crop = img[b[1]:b[3],b[0]:b[2]]
		# cv2.imshow('',_crop)
		if crop.size == 0:
			continue
		summed = np.sum(crop)/(crop.size)
		widths = findContourWidth(_crop)
		# cv2.waitKey(0)
		wid = np.mean(widths) + (2*np.std(widths))
		# print(summed*wid,line['word'])

		if (summed*wid) > 700:
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

				if np.abs(s_dots[1] - dots[1])<0.5*rise:
					if (np.abs(s_dots[0] - dots[2]) < 0.1*rise) :
						line['word'] = ''.join([line['word'].strip(),word['word'].strip()])
						line['box'] = [min(dots[0],s_dots[0]),min(dots[1],s_dots[1]),max(dots[2],s_dots[2]),min(dots[3],s_dots[3])]
						del words[i]
						i-=1
					elif (np.abs(s_dots[0] - dots[2]) < 4*rise) :
						# print([line['word'].split(),word['word'].split()])
						line['word'] = ' '.join([line['word'].strip(),word['word'].strip()])
						line['box'] = [min(dots[0],s_dots[0]),min(dots[1],s_dots[1]),max(dots[2],s_dots[2]),min(dots[3],s_dots[3])]
						del words[i]
						i-=1
				i+=1

def pdfreader(root):
	for i in range(2,567):
		i = 541

		xml_path = os.path.join(out_dir,"page_"+str(i)+".xml")
		img_path = os.path.join(image_dir,"page_"+str(i)+".jpg")

		if not os.path.exists(xml_path):
			os.system("pdf2txt.py "+str(root)+" -p "+str(i)+" -o "+xml_path)
		print(xml_path)
		mytree = ET.parse(xml_path)

		image = cv2.imread(img_path)

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
				if h_-int(float(box[3]))<int(0.025*h):
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
					temp['box'] = [int(p1[0]*f),int(p1[1]*fy),int(p2[0]*f),int(p2[1]*fy)]
					temp['size'] = size
					word = ''
					words.append(temp)
					i = 0
				p2 = (int(float(box[2])),h_-int(float(box[1])))
			except:
				continue
		# columns = verticalProj(image[:])
		# words = column_sort(words,columns,f)
		findHeader(words,image)
		mergeheader(words)
		# print(words)
		
		for word in words:
			if word['header'] == 1:
				print(word['word'])
				b = word['box']
				cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0,0, 255), 3)
		cv2.imshow('',cv2.resize(image,(500,700)))
		cv2.waitKey(0)
			# for word in line:
			#   print(word)

		assert False

pdfreader(root)