import os
import csv
import shutil
import array
import numpy as np
from scipy import misc

# label_file = 'E:/Works/temp/model/label_list_re.csv'
# file_folder = 'E:/Works/temp/model/mann_dataset_folder'
#
# f = open(label_file, 'r', encoding='utf-8')
# rdr = csv.reader(f)
#
# category_set = set()
# category_dict = dict()
#
# for line in rdr:
#     category = line[1]
#     category_set.add(category)
#
# for category in category_set:
#     path = os.path.join(file_folder, category)
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# f2 = open(label_file, 'r', encoding='utf-8')
# rdr2 = csv.reader(f2)
#
# for line in rdr2:
#     md5 = line[0]
#     category_folder = line[1]
#     src = os.path.join(file_folder, md5)
#     dest = os.path.join(file_folder, category_folder)
#     shutil.move(src, dest)
#     print('ok')

# from gensim.models import doc2vec
# import numpy as np
# from PIL import Image
# import numpy as np
# import zipfile
# from scipy import misc
#
# path = 'E:/Project/PycharmProjects/dogs/model/bin_model/basicblock_by_file_bin2vec.model'
# label_file = 'E:/Works/temp/model/label_list_re.csv'
#
# f = open(label_file, 'r', encoding='utf-8')
# rdr = csv.reader(f)
#
# hash_list = list()
# model = doc2vec.Doc2Vec.load(path)

'''
doc2vec -> Image
'''
#
# count = 1
# for line in rdr:
#     hash = line[0]
#     x = np.array(model.docvecs[hash])
#     w, h = 20, 20
#     data = np.zeros((h, w), dtype=np.uint8)
#     data = np.reshape(x, (-1, 20))
#     img = Image.fromarray(data, 'L')
#     filename = hash+'.png'
#     img.save('E:/Works/png/'+filename)
#     print('complete', count,'/3200')
#     count = count + 1


'''
Binary -> Image
'''

# data_folder = 'E:/Works/Data2/tt/'
# save_folder = 'E:/Works/Data2/tt2/'
# # mann_dataset_filename = 'mann_dataset.zip'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         path = data_folder+filename
#         f = open(path, 'rb')
#         ln = os.path.getsize(path)
#
#         width = int(ln**0.5)
#         # width = 256
#
#         rem = ln % width
#
#         a = array.array("B")
#
#         a.fromfile(f, ln-rem)
#         # print(len(a)/width)
#         f.close()
#
#         g = np.reshape(a, (int(len(a)/width), width))
#         g = np.uint8(g)
#         sp = save_folder+filename+'.png'
#         # print(path)
#         misc.imsave(sp, g)
#         print('complete', count, '/1676')
#         count = count + 1


'''
temp code
'''
# label_file = 'E:/Works/Data2/label.csv'
#
# # all_path = 'E:/Works/Data/backup/all/all'
# sample_path = 'E:/Works/Data2/sample_png_gray'
# sample_folder_path = 'E:/Works/Data2/sample_png_gray_folder'
#
# f = open(label_file, 'r', encoding='utf-8')
# rdr = csv.reader(f)
#
# file_dict = dict()
# for (path, dir, files) in os.walk(sample_path):
#     for file in files:
#         s_ex_ext = os.path.splitext(file)
#         p = os.path.join(path, file)
#         file_dict[s_ex_ext[0]] = p
#         # print(p)
#
# # print(file_dict)
#
#
# for line in rdr:
#     hash = line[0]
#     category = line[1]
#     directory = os.path.join(sample_folder_path, category)
#
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     dest_path = os.path.join(sample_folder_path, category, hash+'.png')
#     if file_dict.get(hash) != None:
#         src_path = file_dict[hash]
#         shutil.move(src_path, dest_path)
#

# print(len(file_list))



# for line in rdr:
#     hash = line[0]
#     category = line[1]
#     directory = os.path.join(png_dataset, category)
#
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#
#     src = os.path.join(png_dataset, hash+'.png')
#     dest = os.path.join(png_dataset, directory, hash)
#     shutil.move(src, dest)

'''
change gray -> binary image
'''

# import cv2
#
# data_folder = 'E:/Works/Data2/tt/'
# save_folder = 'E:/Works/Data2/tt2/'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         path = data_folder+filename
#         im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         thresh = 127
#         im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#         sp = save_folder + filename
#         cv2.imwrite(sp, im_bw)
#         print('complete', count, '/1676')
#         count = count + 1


'''
change Binary -> binary image file
'''
import cv2

# data_folder = 'E:/Works/Data2/sample'
# save_folder = 'E:/Works/Data2/sample_png'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         category = path.split('\\')[1]
#         p = os.path.join(path, filename)
#         f = open(p, 'rb')
#         ln = os.path.getsize(p)
#
#         width = int(ln**0.5)
#         # width = 256
#
#         rem = ln % width
#
#         a = array.array("B")
#
#         a.fromfile(f, ln-rem)
#         # print(len(a)/width)
#         f.close()
#
#         g = np.reshape(a, (int(len(a)/width), width))
#         g = np.uint8(g)
#
#         sp = os.path.join(save_folder, category)
#         sp2 = os.path.join(save_folder, category, filename+'.png')
#         if not os.path.exists(sp):
#             os.makedirs(sp)
#
#         # sp = save_folder+filename+'.png'
#         # print(path)
#         misc.imsave(sp2, g)
#         print('complete', count, '/', '1200')
#         count = count + 1



# data_folder = 'E:/Works/Data2/sample_png'
# save_folder = 'E:/Works/Data2/sample_binary_100'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         category = path.split('\\')[1]
#         p = os.path.join(path, filename)
#
#         im_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
#         (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         thresh = 100
#         im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#
#         sp = os.path.join(save_folder, category)
#         if not os.path.exists(sp):
#             os.makedirs(sp)
#         sp2 = os.path.join(save_folder, category, filename)
#         # sp_t = os.path.join(save_folder, filename + '.png')
#         cv2.imwrite(sp2, im_bw)
#         print('complete', count, '/1200')
#         count = count + 1


# data_folder = 'E:/Works/Data2/sample_binary'
# save_folder = 'E:/Works/Data2/sample_binary_resize'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         category = path.split('\\')[1]
#         p = os.path.join(path, filename)
#
#         img = cv2.imread(p)
#         shrink = cv2.resize(img, dsize=(20, 20), interpolation=cv2.INTER_AREA)
#
#         sp_t = os.path.join(save_folder, filename + '.png')
#         cv2.imwrite(sp_t, shrink)
#         print('complete', count, '/420')
#         count = count + 1