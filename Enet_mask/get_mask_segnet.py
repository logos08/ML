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
    blank_image = Image.open("/home/maxim/file/project/pig_v5_affordance_net/data_convert/background.jpg").convert('RGBA')
    for region in coords:
        for name, value in region.items():

            if name == "region_attributes":
                pass
            else:

                coords_x = []
                coords_y = []

                coords_x = region[name]["all_points_x"]
                coords_y = region[name]["all_points_y"]
                draw = ImageDraw.Draw(blank_image)
                # white = ImageColor.getrgb("white")
                draw.polygon(zip(coords_x,coords_y), fill = (255, 255, 255), outline = (255, 255, 255))
    blank_image.save("/home/maxim/file/project/pig_v5_affordance_net/data_convert/supervise_data/{}.png".format(img_name))

    print img_name            
print "Done"





