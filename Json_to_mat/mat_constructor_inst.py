import os
import sys
import json
import scipy.io
import numpy as np

from PIL import Image, ImageDraw
from scipy import sparse

np.set_printoptions(threshold=np.nan)

json_dir = "/home/maxim/file/library/mnc_net/data/json`s/"
inst_mat_dir = "/home/maxim/file/library/mnc_net/data/dataset/inst/"
image_fullsize_dir = "/home/maxim/file/library/mnc_net/data/dataset/img_fullsize/"
image_dir = "/home/maxim/file/library/mnc_net/data/dataset/img/"

mat_template_dir = "/home/maxim/file/library/mnc_net/data/inst_template.mat"

classes_ids = {"pig": 1}

num_of_classes = 1

mat_template = scipy.io.loadmat('{0}'.format(mat_template_dir))

# Create mask for segmentation -----------------------------------------------------------------------
list_json = sorted(list(os.walk(json_dir))[0][2])

for j in list_json:

    json_string = open(json_dir + j)
    data = json.load(json_string)

    files_list = sorted(data["_via_img_metadata"].keys())

    for file in files_list:

        img_name = "{0}".format(file.split(".")[0])
        img = Image.open("{0}{1}.png".format(image_fullsize_dir, img_name))
        mat_template = scipy.io.loadmat('{0}'.format(mat_template_dir))

        coords = data["_via_img_metadata"][file]["regions"]

        width = 512 
        height = 512

        labelSegmentation = None
        drawerSegmentation = None

        boundaries = []
        count = 1
        segmentation_mask_inst = None

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
                        polygon.append(tuple(coord))
                
                    size = ( 1280 , 720 )
                    background = 0
                    val = classes_ids['pig']

                    if labelSegmentation == None:
                        labelSegmentation = Image.new("L", size, background)

                    if drawerSegmentation == None:
                        drawerSegmentation = ImageDraw.Draw(labelSegmentation)

                    label_actual_boundary = Image.new("L", size, background)
                    actual_boundary = ImageDraw.Draw(label_actual_boundary)

                    try:
                        drawerSegmentation.polygon(polygon, fill = count)
                        actual_boundary.line(polygon, fill = 1, width = 1)
                        count += 1
                    except:
                        print("Failed to draw polygon")
                        raise

                    label_actual_boundary = label_actual_boundary.resize((width, height), Image.LANCZOS)

                    actual_boundary_inst = np.array(label_actual_boundary)
                    boundaries.append([sparse.csr_matrix(actual_boundary_inst)])

        labelSegmentation = labelSegmentation.resize((width, height), Image.LANCZOS)
        segmentation_mask_inst = np.array(labelSegmentation)

        mat_template['GTinst'][0]['Segmentation'][0] = segmentation_mask_inst
        mat_template['GTinst'][0]['Boundaries'][0] = np.asarray(boundaries)
        mat_template['GTinst'][0]['Categories'][0] = []
        ids = []

        for b in boundaries:
            ids.append([classes_ids['pig']])
            id_array = np.asarray(ids, dtype = np.uint8)
        mat_template['GTinst'][0]['Categories'][0] = id_array

        scipy.io.savemat("{0}2008_{1:06}.mat".format(inst_mat_dir, int(img_name[2:])), mat_template)
        img = img.resize((width, height), Image.LANCZOS)
        img.save("{}2008_{:06}.png".format(image_dir, int(img_name[2:])))
        print "Save file: {0}2008_{1:06}.mat".format(inst_mat_dir, int(img_name[2:]))

print "Done!"