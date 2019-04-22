import os
import sys
import cv2
import json
import imutils

import xml.etree.cElementTree as ET
import numpy as np
import lxml.etree as etree

from shapely import affinity
from shapely.geometry import Polygon
from xml.dom import minidom

json_dir = "/home/maxim/file/project/pig_v5_affordance_net/json_people/"
images_dir = "/home/maxim/file/project/pig_v5_affordance_net/img_people/"
template_xml = "/home/maxim/file/project/pig_v5_affordance_net/full_image_rotate/template.xml"

new_images_dir = "/home/maxim/file/project/pig_v5_affordance_net/001_fsize_img/"
new_xml_dir = "/home/maxim/file/project/pig_v5_affordance_net/001_fsize_xml/"

angle_step = 90

count = 0

list_json = sorted(list(os.walk(json_dir))[0][2])

loss_images = []

for angle in range(0,360,angle_step):

    for j in list_json:

        json_string = open(json_dir + j)
        data = json.load(json_string)

        files_list = sorted(data["_via_img_metadata"].keys())

        for file in files_list:

            total_obj = {}

            img_name = "{}".format(file.split(".")[0])
            original_img = cv2.imread(images_dir + img_name + ".jpg")
            rotate_img = imutils.rotate(original_img, angle)

            height, width = rotate_img.shape[:2]

            coords = data["_via_img_metadata"][file]["regions"]
            bound_boxes = []

            for region in coords:

                for name, value in region.items():

                    if name == "region_attributes":
                        pass
                    else:
                        coords_x = []
                        coords_y = []
                        polygon_coords = []

                        coords_x = region[name]["all_points_x"]
                        max_coord_x = max(coords_x)
                        min_coord_x = min(coords_x)
                        coords_y = region[name]["all_points_y"]
                        max_coord_y = max(coords_y)
                        min_coord_y = min(coords_y)

                        coords = zip(coords_x, coords_y)

                        for coord in coords:
                            polygon_coords.append(tuple(coord))

                        polygon = Polygon(polygon_coords)

                        rotate_center_x = width / 2
                        rotate_center_y = height / 2

                        rotate_polygon = affinity.rotate(polygon, -angle, (rotate_center_x, rotate_center_y))

                        if 0 <= rotate_polygon.bounds[0] <= width and 0 <= rotate_polygon.bounds[2] <= width and 0 <= rotate_polygon.bounds[1] <= height and 0 <= rotate_polygon.bounds[3] <= height:
                                box = [int(x) for x in rotate_polygon.bounds]
                                if region.items()[1][1] == {}:
                                    name = "pig"
                                else:
                                    name = region.items()[1][1]["name"].lower()
                                box.append(name)
                                bound_boxes.append(box)

            if len(bound_boxes) == 0:
                loss_images.append(img_name)
                continue

            
            cv2.imwrite("{0}{1:07}.jpg".format(new_images_dir, count), rotate_img)

            tree = ET.parse(template_xml)
            root = tree.getroot()

            for filename in root.iter("filename"):
                filename.text = "{0:07}.jpg".format(count)

            for box in bound_boxes:
                xml_object = ET.Element("object")
                ET.SubElement(xml_object, 'name').text = box[4]
                ET.SubElement(xml_object, 'pose').text = 'Unspecified'
                ET.SubElement(xml_object, 'truncated').text = '0'
                ET.SubElement(xml_object, 'difficult').text = '0'

                item = ET.SubElement(xml_object, 'bndbox')
                ET.SubElement(item, 'xmin').text = str(box[0])
                ET.SubElement(item, 'ymin').text = str(box[1])
                ET.SubElement(item, 'xmax').text = str(box[2])
                ET.SubElement(item, 'ymax').text = str(box[3])
                root.append(xml_object)
                
                xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
                xml_object = ET.fromstring(xmlstr)

            tree = ET.ElementTree(xml_object)
            root = tree.getroot()
            for w in root.iter("width"):
                w.text = str(width)
            for h in root.iter("height"):
                h.text = str(height)
            tree = ET.ElementTree(root)

            new_xml_file = "{0}{1:07}.xml".format(new_xml_dir, count)
            tree.write(new_xml_file)

            count += 1

            print "{0:07} rotate at {1} and contains {2} objects".format(count, angle, len(bound_boxes))

print "loss_images: {0} \n".format(len(loss_images))
print "\nDone!"