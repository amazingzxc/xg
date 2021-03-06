#*-coding: UTF-8 -*-

import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

prefix = './data/'
def get_img(img_path, resize_h=128):
    img=scipy.misc.imread(img_path).astype(np.float)
    # crop resize
    # crop_w = crop_h
    #resize_h = 64
    resize_w = resize_h
    # h, w = img.shape[:2]
    # j = int(round((h - crop_h)/2.))
    # i = int(round((w - crop_w)/2.))
    # cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])
    cropped_image = scipy.misc.imresize(img, [resize_h, resize_w])
    # plt.imshow(cropped_image[:, :,0])
    # plt.show()
    return np.array(cropped_image)/127.5 - 1.

def show_img(img):
    plt.imshow(img)
    plt.show()

if __name__=='__main__':
    test_path = '../data/y_domain'
    # test_path='../data/x_domain'
    img_path=os.path.join(test_path,os.listdir(test_path)[0])
    # print(get_img(img_path,200,200))
    img = get_img(img_path,128)
    show_img(img)
    show_img((img+1)/2)

