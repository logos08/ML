import cv2
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw, ImageColor

json_string = open("pig_annotate.json")
data = json.load(json_string)

files_list = sorted(data["_via_img_metadata"].keys())
height = 720
width = 1280

for file in files_list:
    img_name = "{}".format(file.split(".")[0])
    coords = data["_via_img_metadata"][file]["regions"]
    blank_image = img = Image.open("/home/maxim/file/project/pig_v5_affordance_net/data/background.jpg").convert('RGBA')
    count = 1
    for region in coords:
        for name, value in region.items():

            if name == "region_attributes":
                pass
            else:

                coords_x = []
                coords_y = []

                coords_x = region[name]["all_points_x"]
                coords_y = region[name]["all_points_y"]
                blank_image2 = blank_image.copy()
                draw = ImageDraw.Draw(blank_image2)
                # white = ImageColor.getrgb("white")
                draw.polygon(zip(coords_x,coords_y), fill = (255, 255, 255), outline = (255, 255, 255))
                img3 = Image.blend(blank_image, blank_image2, 0.5)
                img3.save("/home/maxim/file/project/pig_v5_affordance_net/data/supervise_data/{}_{}.png".format(img_name, count))
        count += 1

    print img_name            
print "Done"





