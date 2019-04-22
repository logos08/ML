import os
import sys
import json
import scipy.io
import numpy as np

from PIL import Image, ImageDraw
from scipy import sparse

json_dir = "/home/maxim/file/library/mnc_net/data/json`s/"
cls_mat_dir = "/home/maxim/file/library/mnc_net/data/dataset/cls/"
inst_mat_dir = "/home/maxim/file/library/mnc_net/data/dataset/inst/"
image_fullsize_dir = "/home/maxim/file/library/mnc_net/data/dataset/img_fullsize/"
image_dir = "/home/maxim/file/library/mnc_net/data/dataset/img/"


mat_template_dir = "/home/maxim/file/library/mnc_net/data/cls_template.mat"

classes_ids = {"pig": 1}

num_of_classes = 1

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
        labelBoundaries = None
        drawerBoundaries = None

        segmentation_mask_cls = None
        new_boundaries = None

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

                    if labelBoundaries == None:
                        labelBoundaries = Image.new("L", size, background)

                    if drawerSegmentation == None:
                        drawerSegmentation = ImageDraw.Draw(labelSegmentation)

                    if drawerBoundaries == None:
                        drawerBoundaries = ImageDraw.Draw(labelBoundaries)

                    try:
                        drawerSegmentation.polygon( polygon, fill = val)
                        drawerBoundaries.line(polygon, fill = 1, width = 1)
                    except:
                        print("Failed to draw polygon")
                        raise

        labelSegmentation = labelSegmentation.resize((width, height), Image.LANCZOS)
        labelBoundaries = labelBoundaries.resize((width, height), Image.LANCZOS)

        segmentation_mask_cls = np.array(labelSegmentation)
        boundaries_mask_cls = np.array(labelBoundaries)
        empty_boundary_mask = np.array(Image.new("L", (width, height), background))

        new_boundaries = mat_template['GTcls'][0]['Boundaries'][0][:2]
        new_boundaries[0] = sparse.csr_matrix(empty_boundary_mask)
        new_boundaries[1] = sparse.csr_matrix(boundaries_mask_cls)
        
        mat_template['GTcls'][0]['Segmentation'][0] = segmentation_mask_cls
        mat_template['GTcls'][0]['Boundaries'][0] = new_boundaries

        mat_template['GTcls'][0]['CategoriesPresent'][0][0] = classes_ids['pig']

        scipy.io.savemat("{0}2008_{1:06}.mat".format(cls_mat_dir, int(img_name[2:])), mat_template)
        img = img.resize((width, height), Image.LANCZOS)
        img.save("{}2008_{:06}.png".format(image_dir, int(img_name[2:])))
        print "Save file: {0}2008_{1:06}.mat".format(cls_mat_dir, int(img_name[2:]))

print "Done!"