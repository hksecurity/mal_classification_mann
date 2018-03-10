import cv2
import numpy as np
from scipy import misc
import os
from PIL import Image
import PIL.ImageOps
import shutil

FNAME = 'E:/Works/Data/paper/mnist/sample/digits.npz'
GRAYFOLDER = 'E:/Works/Data/paper/mnist/gray'
GRAYIFOLDER = 'E:/Works/Data/paper/mnist/gray_invert'

# convert digits.png -> digits.npz
def learningDigit():
    img = cv2.imread('E:/Works/Data/paper/mnist/sample/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

    # print(cells)

    x = np.array(cells)
    # print(x.shape)
    train = x[:, :].reshape(-1, 400).astype(np.float32)
    print(train.shape)
    k = np.arange(10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]

    np.savez(FNAME, train=train, train_labels=train_labels)

# learningDigit()

# load digits.npz
def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

# learningDigit()
# train, train_labels = loadTrainData(FNAME)

# print(train_labels[499])


# npz file -> grayscale png file
def saveGrayScale(fname):
    save_folder = 'E:/Works/Data/paper/mnist/gray'
    dataset = np.load(fname)

    images = dataset['train']
    # images = np.array([image for image in images])
    images = np.array([image.reshape(20,20) for image in images])
    # print(images[0])

    labels = dataset['train_labels']
    labels = np.array([label[0] for label in labels])

    for i, label in enumerate(labels):
        path = os.path.join(save_folder, str(label))
        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, str(label) + '_' + str(i) + '.png')
        misc.imsave(path, images[i])

# saveGrayScale(FNAME)

# image invert
def imageInvert(image_folder):
    save_folder = 'E:/Works/Data/paper/mnist/gray_invert'

    for (path, dir, files) in os.walk(image_folder):
        for filename in files:
            full_filename = os.path.join(path, filename)
            image = Image.open(full_filename)
            inverted_image = PIL.ImageOps.invert(image)
            save_file = os.path.join(save_folder, filename)
            inverted_image.save(save_file)

# imageInvert(GRAYFOLDER)

# order by folder
def imageInvertPlusFolder(image_folder):
    for (path, dir, files) in os.walk(image_folder):
        for filename in files:
            front = filename.split('_')[0]
            old_full_path = os.path.join(path, filename)
            new_path = os.path.join(path, front)
            if not os.path.exists(new_path):
                os.mkdir(new_path)

            new_full_path = os.path.join(new_path, filename)

            shutil.move(old_full_path, new_full_path)

# imageInvertPlusFolder('E:/Works/Data/paper/mnist/gray_invert')

# gray to binary image ( 0 ~ 100)
def gray_to_binaryimg(source_folder):
    save_folder = 'E:/Works/Data/paper/mnist/binary'

    count = 1
    for (path, dir, files) in os.walk(source_folder):
        for filename in files:
            category = path.split('\\')[-1]

            p = os.path.join(path, filename)

            im_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = 100
            im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]

            sp = os.path.join(save_folder, category)
            if not os.path.exists(sp):
                os.makedirs(sp)
            sp2 = os.path.join(save_folder, category, filename)
            # sp_t = os.path.join(save_folder, filename + '.png')
            cv2.imwrite(sp2, im_bw)
            print('complete', count)
            count = count + 1

# gray_to_binaryimg(GRAYIFOLDER)


# def resize20(pimg):
#     img = cv2.imread(pimg)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     grayResize = cv2.resize(gray,(20,20))
#     ret, thresh = cv2.threshold(grayResize, 125, 255,cv2.THRESH_BINARY_INV)
#
#     cv2.imshow('num',thresh)
#     return thresh.reshape(-1,400).astype(np.float32)