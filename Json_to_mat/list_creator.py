import os
import sys
import numpy as np

containing_dir_img = "dataset/inst"

list_img_files = list(os.walk(containing_dir_img))[0][2]

part_train = 0.8
part_valid = 0.2

count_train = int(round(len(list_img_files) * part_train))
count_valid = len(list_img_files) - count_train

train_list = list_img_files[:count_train]
valid_list = list_img_files[-count_valid:]

print "Images for train: {}".format(len(train_list))
print "Images for valid: {}".format(len(valid_list))

with open("dataset/train.txt", "a") as f:
    for x in train_list:
        f.write("{}{}".format(x.split(".")[0], "\n"))

with open("dataset/val.txt", "a") as f:
    for x in valid_list:
        f.write("{}{}".format(x.split(".")[0], "\n"))

print "Done!"
