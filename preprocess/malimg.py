import cv2
import numpy as np
from scipy import misc
import os

FNAME = 'E:/Works/Data/paper/malimg/sample/malimg.npz'

def saveGrayScale(fname):
    save_folder = 'E:/Works/Data/paper/malimg/gray/'
    dataset = np.load(fname)

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

saveGrayScale(FNAME)