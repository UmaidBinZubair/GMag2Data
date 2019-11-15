# from PyPDF2 import PdfFileWriter, PdfFileReader

# inputpdf = PdfFileReader(open("guitar.pdf", "rb"))

# for i in range(inputpdf.numPages):
#     output = PdfFileWriter()
#     output.addPage(inputpdf.getPage(i))
#     with open("pdf_pages/document-page%s.jpg" % i, "wb") as outputStream:
#         output.write(outputStream)

# from PyPDF2 import PdfFileReader
# from pprint import pprint

# def walk(obj, fnt, emb):
#     '''
#     If there is a key called 'BaseFont', that is a font that is used in the document.
#     If there is a key called 'FontName' and another key in the same dictionary object
#     that is called 'FontFilex' (where x is null, 2, or 3), then that fontname is 
#     embedded.
    
#     We create and add to two sets, fnt = fonts used and emb = fonts embedded.
#     '''
#     if not hasattr(obj, 'keys'):
#         return None, None
#     fontkeys = set(['/FontFile', '/FontFile2', '/FontFile3'])
#     if '/BaseFont' in obj:
#         fnt.add(obj['/BaseFont'])
#     if '/FontName' in obj:
#         if [x for x in fontkeys if x in obj]:# test to see if there is FontFile
#             emb.add(obj['/FontName'])

#     for k in obj.keys():
#         walk(obj[k], fnt, emb)

#     return fnt, emb# return the sets for each page

# if __name__ == '__main__':
#     fname = 'pdf_pages/document-page33.pdf'
#     pdf = PdfFileReader(fname)
#     fonts = set()
#     embedded = set()
#     for page in pdf.pages:
#         obj = page.getObject()
#         f, e = walk(obj['/Resources'], fonts, embedded)
#         fonts = fonts.union(f)
#         embedded = embedded.union(e)

#     unembedded = fonts - embedded
#     print ('Font List')
#     print(sorted(list(fonts)))
#     if unembedded:
#         print ('\nUnembedded Fonts')
#         print(unembedded)


import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
def convert_pdf(file_path, output_path):
    # save temp image files in temp dir, delete them after we are finished
        # convert pdf to multiple image
    print('about to convert')
    images = convert_from_path(file_path,300,first_page = 31, last_page = 33)
    # save images to temporary directory
    temp_images = []
    for i in range(len(images)):
        image_path = os.path.join(output_path,'page_'+str(i-3))
        print(image_path)
        images[i].save(image_path, 'JPEG')
        # temp_images.append(image_path)
        # read images into pillow.Image
    #     imgs = list(map(Image.open, temp_images))
    # # find minimum width of images
    # min_img_width = min(i.width for i in imgs)
    # # find total height of all images
    # total_height = 0
    # for i, img in enumerate(imgs):
    #     total_height += imgs[i].height
    # # create new image object with width and total height
    # merged_image = Image.new(imgs[0].mode, (min_img_width, total_height))
    # # paste images together one by one
    # y = 0
    # for img in imgs:
    #     merged_image.paste(img, (0, y))
    #     y += img.height
    # # save merged image
    # merged_image.save(output_path)
    # return output_path

convert_pdf('guitar.pdf','./pdf_pages/')
