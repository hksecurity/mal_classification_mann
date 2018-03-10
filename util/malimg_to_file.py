import os
import csv
import shutil
import array
import numpy as np
from scipy import misc
import cv2
from sklearn.preprocessing import StandardScaler

#
# data_folder = 'E:/Works/Data_MS/train/'
# save_folder = 'E:/Works/Data_MS/output_gray/'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#
#         path = data_folder+filename
#         fname, ext = os.path.splitext(path)
#         if ext == '.asm':
#             f = open(path, 'rb')
#             ln = os.path.getsize(path)
#
#             width = 256
#             width = int(ln**0.5)
#
#             rem = ln % width
#
#             a = array.array("B")
#
#             a.fromfile(f, ln-rem)
#             # print(len(a)/width)
#             f.close()
#
#             g = np.reshape(a, (int(len(a)/width), width))
#             g = np.uint8(g)
#             sp = save_folder+filename+'.png'
#             # print(path)
#             misc.imsave(sp, g)
#             print('complete', count)
#             count = count + 1


#
#
# data_folder = 'E:/Works/Data_MS/output_gray/'
# save_folder = 'E:/Works/Data_MS/output_binary/'
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
#         print('complete', count)
#         count = count + 1


def malimg_to_gray():
    save_folder = 'E:/Works/Data_MS/output_gray/'
    dataset = np.load('E:/Works/malimg/malimg.npz')

    images = dataset['arr'][:, 0]
    images = np.array([image for image in images])

    labels = dataset['arr'][:, 1]
    labels = np.array([label for label in labels])

    for i, label in enumerate(labels):

        path = os.path.join(save_folder, str(label))
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, str(label)+'_'+str(i)+'.png')
        misc.imsave(path, images[i])

# malimg_to_gray()


def gray_to_binaryimg():

    data_folder = 'E:/Works/Data_MS/output_gray_choice_20/'
    save_folder = 'E:/Works/Data_MS/output_binary_0'

    count = 1
    for (path, dir, files) in os.walk(data_folder):
        for filename in files:
            category = path.split('/')[4]

            p = os.path.join(path, filename)

            im_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = 0
            im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

            sp = os.path.join(save_folder, category)
            if not os.path.exists(sp):
                os.makedirs(sp)
            sp2 = os.path.join(save_folder, category, filename)
            # sp_t = os.path.join(save_folder, filename + '.png')
            cv2.imwrite(sp2, im_bw)
            print('complete', count)
            count = count + 1

gray_to_binaryimg()