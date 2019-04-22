import os
import sys
import json
import scipy.io
import numpy as np

from PIL import Image, ImageDraw
from scipy import sparse

json_dir = "/home/maxim/file/library/mnc_net/data/new_json/"

# Create mask for segmentation -----------------------------------------------------------------------
list_json = sorted(list(os.walk(json_dir))[0][2])

total_pics = 0
total_obj = 0

for j in list_json:

    json_string = open(json_dir + j)
    data = json.load(json_string)

    files_list = sorted(data["_via_img_metadata"].keys())

    for file in files_list:

        img_name = "{0}".format(file.split(".")[0])
        total_pics += 1

        coords = data["_via_img_metadata"][file]["regions"]

        for region in coords:

            for name, value in region.items():

                if name == "region_attributes":
                    pass
                else:
                    coords_x = []
                    coords_y = []
                    polygon = []

                    coords_x = region[name]["all_points_x"]
                    coords_y = region[name]["all_points_y"]

                    if len(coords_x) > 2 and len(coords_y) > 2:
                        total_obj += 1

print "Done!"
print "Total images: {}".format(total_pics)
print "Total objects: {}".format(total_obj)