import os
import sys
import cv2
import argparse

import numpy as np

from matplotlib import pyplot as plt

def create_hystograms(args):

    input_img = args.input_image
    mask_str = args.mask_coords.split(",")
    class_name = args.classname

    mask = [int(x) for x in mask_str]

    image = cv2.imread(input_img, 0)
    img_mask = np.zeros(image.shape[:2], np.uint8)
    img_mask[mask[0]:mask[2], mask[1]:mask[3]] = 255
    masked_img = cv2.bitwise_and(image,image,mask = img_mask)

    hist_mask = cv2.calcHist([image],[0],img_mask,[255],[1,255])
    print type(hist_mask)

    # plt.subplot(211), plt.imshow(masked_img, 'gray')
    # plt.title(class_name)
    # plt.subplot(212), plt.plot(hist_mask), plt.plot(hist_mask)
    # plt.xlim([0,255])

    # plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create hystograms")

    parser.add_argument("-input", "--input_image", type = str)
    parser.add_argument("-mask_coords", "--mask_coords", type = str)
    parser.add_argument("-class", "--classname", type = str)

    args = parser.parse_args()

    create_hystograms(args)