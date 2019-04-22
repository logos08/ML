import os
import sys
import cv2
import argparse

import numpy as np

from matplotlib import pyplot as plt

def create_hystograms(args):

    input_dir = args.input_dir
    class_name = args.classname

    img_list = sorted(list(os.walk(input_dir))[0][2])

    plot_index = 1

    for img in img_list:

        image = cv2.imread("{}/{}".format(input_dir, img), 0)
        img_mask = np.zeros(image.shape[:2], np.uint8)

        if args.mask_coords == "0":

            img_mask[0 : image.shape[0], 0 : image.shape[1]] = 255

        else:

            mask_str = args.mask_coords.split(",")
            mask = [int(x) for x in mask_str]
            img_mask[mask[0] : mask[2], mask[1] : mask[3]] = 255

        masked_img = cv2.bitwise_and(image, image, mask = img_mask)

        hist_mask = cv2.calcHist([image],[0],img_mask,[255],[1,255])

        # plt.subplot(2, 10, plot_index), plt.imshow(masked_img, 'gray')
        # plt.title("{}: {}".format(class_name, plot_index))
        # plt.subplot(2, 10, plot_index + 10), plt.plot(hist_mask), plt.plot(hist_mask)
        # plot_index += 1
        # plt.xlim([0,255])

        # plt.subplot(10, 2, plot_index), plt.imshow(masked_img, 'gray')
        # plt.subplot(10, 2, plot_index + 1), plt.plot(hist_mask), plt.plot(hist_mask)
        # # plt.title("{}: {}".format(class_name, plot_index))
        # plot_index += 2
        # plt.xlim([0,255])

        plt.subplot(2, 10, plot_index), plt.imshow(masked_img, 'gray')
        plt.title("{}: {}".format(class_name, plot_index))
        plt.subplot(2, 1, 2), plt.plot(hist_mask), plt.plot(hist_mask)
        plot_index += 1
        plt.xlim([0,255])

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create hystograms")

    parser.add_argument("-input", "--input_dir", type = str)
    parser.add_argument("-mask_coords", "--mask_coords", type = str, default = "0")
    parser.add_argument("-class", "--classname", type = str)

    args = parser.parse_args()

    create_hystograms(args)