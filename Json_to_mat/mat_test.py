import os
import sys
import json
import scipy.io
import numpy as np

from PIL import Image

projectPath = os.path.dirname(os.path.realpath(__file__))

np.set_printoptions(threshold=np.nan)

# Set path to the test files -----------------------------------------------------------------------
# cls_mat_name = "/home/maxim/file/library/mnc_net/data/test/cls_2008_000001.mat"
# inst_mat_name = "/home/maxim/file/library/mnc_net/data/test/2008_000001.mat"

# cls_mat_name = "/home/maxim/file/library/mnc_net/data/test/cls_2008_000019.mat"
# inst_mat_name = "/home/maxim/file/library/mnc_net/data/test/inst_2008_000019.mat"

#---------------------------------------------------------------------------------------------------

cls_prefix = "cls"
inst_prefix = "inst"

cls_mat_template = scipy.io.loadmat('{0}'.format(cls_mat_name))
inst_mat_template = scipy.io.loadmat('{0}'.format(inst_mat_name))

# Cls segmentation check ---------------------------------------------------------------------------
cls_segmentation = cls_mat_template["GTcls"][0]["Segmentation"][0]
cls_segmentation[cls_segmentation > 0] = 200
cls_segmentation_img = Image.fromarray(cls_segmentation)
cls_segmentation_img.save("img_example/{}_segmentation.png".format(cls_prefix))

# Cls boundaries check -----------------------------------------------------------------------------
cls_boundaries = cls_mat_template["GTcls"][0]["Boundaries"][0]
for boundary in cls_boundaries:
    arr_boundary = boundary[0].toarray()
    if arr_boundary.any():
        arr_boundary[arr_boundary > 0] = 200
        cls_boundary_img = Image.fromarray(arr_boundary)
        cls_boundary_img.save("img_example/{}_boundaries.png".format(cls_prefix))

# Cls Categories check -----------------------------------------------------------------------------
cls_categories = cls_mat_template["GTcls"][0]["CategoriesPresent"]
print cls_categories
print "Cls check done!"


# Inst segmentation check --------------------------------------------------------------------------
inst_segmentation = inst_mat_template["GTinst"][0]["Segmentation"][0]
inst_segmentation[inst_segmentation == 1] = 200
inst_segmentation[inst_segmentation == 2] = 100
inst_segmentation[inst_segmentation == 3] = 50
inst_segmentation[inst_segmentation == 4] = 150
inst_segmentation[inst_segmentation == 5] = 250
inst_segmentation_img = Image.fromarray(inst_segmentation)
inst_segmentation_img.save("img_example/{}_segmentation.png".format(inst_prefix))

# Inst boundaries check ----------------------------------------------------------------------------
inst_boundaries = inst_mat_template["GTinst"][0]["Boundaries"][0]

for i, boundary in enumerate(inst_boundaries):
    arr_boundary = boundary[0].toarray()
    if arr_boundary.any():
        arr_boundary[arr_boundary > 0] = 200
        inst_boundary_img = Image.fromarray(arr_boundary)
        inst_boundary_img.save("img_example/{}_boundaries_{:03}.png".format(inst_prefix, i))

# Inst Categories check ----------------------------------------------------------------------------
inst_categories = inst_mat_template["GTinst"][0]["Categories"]
print inst_categories
print "Inst check done!"