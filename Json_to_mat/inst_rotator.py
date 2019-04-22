import os
import sys
import json
import scipy.io
import numpy as np

from shapely import affinity
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from scipy import sparse

# Set work directories ========================================================================

json_dir = "/home/maxim/file/library/mnc_net/data/json`s/" # directory with json files
cls_mat_dir = "/home/maxim/file/library/mnc_net/data/dataset/cls/" # directory with cls .mat files
inst_mat_dir = "/home/maxim/file/library/mnc_net/data/dataset/inst/" # directory with inst .mat files
image_fullsize_dir = "/home/maxim/file/library/mnc_net/data/dataset/img_fullsize/" # directory that contains fullsize (1280x720 and another) images
image_dir = "/home/maxim/file/library/mnc_net/data/dataset/img/" # directory for resized images (1024x512)

mat_template_dir = "/home/maxim/file/library/mnc_net/data/inst_template.mat" # mat-file template

# Set script parameters =======================================================================

angle_step = 30 # angle, which introduce to step for data rotation
count = 1 # count for naming

img_size = ( 1024 , 512 ) # final image size

classes_ids = {"pig": 1} # dictionary with classes. Structure - {"class" : id}

# MAIN ========================================================================================

list_json = sorted(list(os.walk(json_dir))[0][2])

loss_images = []

for angle in range(0,91,angle_step):

    for j in list_json:

        json_string = open(json_dir + j)
        data = json.load(json_string)

        files_list = sorted(data["_via_img_metadata"].keys())

        for file in files_list:

            img_name = "{0}".format(file.split(".")[0])
            img = Image.open("{0}{1}.png".format(image_fullsize_dir, img_name))
            mat_template = scipy.io.loadmat('{0}'.format(mat_template_dir))

            coords = data["_via_img_metadata"][file]["regions"]
            bound_boxes = []

            orig_width = img.size[0] 
            orig_height = img.size[1]

            labelSegmentation = None
            drawerSegmentation = None

            boundaries = []
            id_inst = 1
            segmentation_mask_inst = None

            for region in coords:

                for name, value in region.items():

                    if name == "region_attributes":
                        pass
                    else:
                        coords_x = []
                        coords_y = []
                        polygon_coords = []

                        coords_x = region[name]["all_points_x"]
                        coords_y = region[name]["all_points_y"]

                        coords = zip(coords_x, coords_y)

                        for coord in coords:
                            polygon_coords.append(tuple(coord))

                        background = 0 # background pixels always 0
                        val = classes_ids['pig']

                        if labelSegmentation == None:
                            labelSegmentation = Image.new("L", img_size, background)

                        if drawerSegmentation == None:
                            drawerSegmentation = ImageDraw.Draw(labelSegmentation)

                        label_actual_boundary = Image.new("L", img_size, background)
                        actual_boundary = ImageDraw.Draw(label_actual_boundary)

                        polygon = Polygon(polygon_coords)

                        x_fact = float(img_size[0]) / float(orig_width) # calculate x scale factor
                        y_fact = float(img_size[1]) / float(orig_height) # calculate y scale factor

                        polygon_scaled = affinity.scale(polygon, xfact=x_fact, yfact=y_fact, origin=(0,0)) #scale polygon from(1280x720)to(1024x512)

                        # calculate rotate cener coordinates
                        rotate_center_x = img_size[0] / 2
                        rotate_center_y = img_size[1] / 2

                        rotate_polygon = affinity.rotate(polygon_scaled, -angle, (rotate_center_x, rotate_center_y)) # rotate polygon

                        if 0 <= rotate_polygon.bounds[0] <= img_size[0] and 0 <= rotate_polygon.bounds[2] <= img_size[0] and 0 <= rotate_polygon.bounds[1] <= img_size[1] and 0 <= rotate_polygon.bounds[3] <= img_size[1]:
                            box = [int(x) for x in rotate_polygon.bounds]
                            bound_boxes.append(box)

                        try:
                            drawerSegmentation.polygon(rotate_polygon.exterior.coords, fill = id_inst) # try draw segmentation objects
                            actual_boundary.line(rotate_polygon.exterior.coords, fill = 1, width = 1) # try draw segmentation boundaries
                            id_inst += 1
                        except:
                            print("Failed to draw polygon")
                            raise

                        actual_boundary_inst = np.array(label_actual_boundary)
                        boundaries.append([sparse.csr_matrix(actual_boundary_inst)])

            if len(bound_boxes) == 0:
                    loss_images.append(img_name) # if turned image have no objects - pass him
                    continue

            # create array from image
            segmentation_mask_inst = np.array(labelSegmentation)

            # reset mat file fields
            mat_template['GTinst'][0]['Segmentation'][0] = segmentation_mask_inst
            mat_template['GTinst'][0]['Boundaries'][0] = np.asarray(boundaries)
            mat_template['GTinst'][0]['Categories'][0] = []
            ids = []

            for b in boundaries:
                ids.append([classes_ids['pig']])
            id_array = np.asarray(ids, dtype = np.uint8)
            mat_template['GTinst'][0]['Categories'][0] = id_array

            # save mat file
            scipy.io.savemat("{0}2008_{1:06}.mat".format(inst_mat_dir, count), mat_template)
            print "Save file: {0}2008_{1:06}.mat".format(inst_mat_dir, count)
            count += 1

print "loss_images: {} \n".format(len(loss_images))
print "\nDone!"