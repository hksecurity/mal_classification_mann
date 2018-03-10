import cv2
import numpy as np
import os

FNAME = 'E:/Works/Data/paper/malimg/npz/malimg100.npz'

def learningDigit(image_folder):

    value_list = []
    label_list = []
    for (path, dir, files) in os.walk(image_folder):
        for filename in files:
            image_path = os.path.join(path, filename)
            category = path.split('\\')[-1]
            label_list.append(category)

            # img = cv2.imread(image_path)
            im_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            grayResize = cv2.resize(im_gray, (20, 20))
            (thresh, im_bw) = cv2.threshold(grayResize, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            thresh = 100
            im_bw = cv2.threshold(grayResize, thresh, 255, cv2.THRESH_BINARY)[1]
            im_bw = im_bw.flatten()
            value_list.append(im_bw)

    train = np.vstack(value_list)
    train_labels = np.vstack(label_list)
    np.savez(FNAME, train=train, train_labels=train_labels)

learningDigit('E:/Works/Data/paper/malimg/add_data/output_gray_choice_20')

# load digits.npz
def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

# train, train_labels = loadTrainData(FNAME)
# print(train)