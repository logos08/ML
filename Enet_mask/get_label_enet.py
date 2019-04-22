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
    out_dir = "/home/maxim/file/project/pig_v5_affordance_net/label_img/"

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

                coords = zip(coords_x, coords_y)

                for coord in coords:
                    polygon.append(list(coord))
                
                size = ( width , height )
                encoding = "ids"

                
                if encoding == "ids":
                    background = name2label['unlabeled'].id
                elif encoding == "trainIds":
                    background = name2label['unlabeled'].trainId
                elif encoding == "color":
                    background = name2label['unlabeled'].color
                else:
                    print("Unknown encoding '{}'".format(encoding))
                    return None

                # this is the image that we want to create
                if encoding == "color":
                labelImg = Image.new("RGBA", size, background)
                else:
                labelImg = Image.new("L", size, background)

                # a drawer to draw into the image
                drawer = ImageDraw.Draw( labelImg )

                # loop over all objects
                label   = "pig"

                if ( not label in name2label ) and label.endswith('group'):
                    label = label[:-len('group')]

                if not label in name2label:
                    printError( "Label '{}' not known.".format(label) )

                # If the ID is negative that polygon should not be drawn
                if name2label[label].id < 0:
                    continue

                if encoding == "ids":
                    val = name2label[label].id
                elif encoding == "trainIds":
                    val = name2label[label].trainId
                elif encoding == "color":
                    val = name2label[label].color

                try:
                    if outline:
                        drawer.polygon( polygon, fill=val, outline=outline )
                    else:
                        drawer.polygon( polygon, fill=val )
                except:
                    print("Failed to draw polygon with label {}".format(label))
                    raise

            labelImg.save(out_dir + file)

    print img_name

print "Done"





