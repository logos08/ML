import os
import cv2
import imutils

from xml import etree
from shapely.geometry import box

import xml.etree.cElementTree as ET


filenames_jpg = sorted(list(os.walk("/home/maxim/file/project/pig_v5_fully_rotated_data/data/002_fsize_img/"))[0][2])
filenames_xml = sorted(list(os.walk("/home/maxim/file/project/pig_v5_fully_rotated_data/data/002_fsize_xml/"))[0][2])

count = 3403

def coords(width, height, coords, angle):

    x1 = coords[0]
    y1 = coords[1]
    x2 = coords[2]
    y2 = coords[3]

    if angle == 270:

        _x1 = width - y2
        _y1 = x1
        _x2 = width - y1
        _y2 = x2

        return [_x1, _y1, _x2, _y2]

    if angle == 180:

        _x1 = width - x2
        _y1 = height - y2
        _x2 = width - x1
        _y2 = height - y1

        return [_x1, _y1, _x2, _y2]

    if angle == 90:

        _x1 = y1
        _y1 = height - x2
        _x2 = y2
        _y2 = height - x1

        return [_x1, _y1, _x2, _y2]

# MAIN -----------------------------------------------------------------------------------------------------------

for angle in range(90, 360, 90):
    
    for f_jpg, f_xml in zip(filenames_jpg, filenames_xml):
        
        old_jpg_path = "/home/maxim/file/project/pig_v5_fully_rotated_data/data/002_fsize_img/{0}".format(f_jpg)
        
        original_img = cv2.imread(old_jpg_path)
        rotate_img = imutils.rotate(original_img, -angle)

        height_img, width_img = rotate_img.shape[:2]

        new_jpg_path = "/home/maxim/file/project/pig_v5_fully_rotated_data/data/007_fsize_img/{0:07}.jpg".format(count)
    
        old_path_xml = "/home/maxim/file/project/pig_v5_fully_rotated_data/data/002_fsize_xml/{0}".format(f_xml)
        
        tree = ET.parse(old_path_xml)
        root = tree.getroot()

        for filename in root.iter("filename"):

            filename.text = "{0:07}.jpg".format(count)

        for width in root.iter("width"):

            width.text = str(width_img)

        for height in root.iter("height"):

            height.text = str(height_img)

        for _box in root.iter("bndbox"):

            x1 = float(_box.find("xmin").text)
            y1 = float(_box.find("ymin").text)
            x2 = float(_box.find("xmax").text)
            y2 = float(_box.find("ymax").text)

            coords_list = [x1, y1, x2, y2]
            angle_coords = coords(width_img, height_img, coords_list, angle)

            for x in _box.iter("xmin"):

                x.text = str(int(angle_coords[0]))

            for y in _box.iter("ymin"):

                y.text = str(int(angle_coords[1]))

            for x in _box.iter("xmax"):

                x.text = str(int(angle_coords[2]))

            for y in _box.iter("ymax"):

                y.text = str(int(angle_coords[3]))

        tree = ET.ElementTree(root)
    
        new_xml_file = "/home/maxim/file/project/pig_v5_fully_rotated_data/data/007_fsize_xml/{0:07}.xml".format(count)
        tree.write(new_xml_file)
        cv2.imwrite(new_jpg_path, rotate_img)
    
        print "Processing: {}, angle: {}".format(count, angle)
        count += 1
    break

print"Done"