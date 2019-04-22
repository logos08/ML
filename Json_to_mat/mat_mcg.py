import os
import sys
import json
import scipy.io
import numpy as np

from PIL import Image

projectPath = os.path.dirname(os.path.realpath(__file__))

np.set_printoptions(threshold=np.nan)

# Set path to the test files -----------------------------------------------------------------------
mcg_mat_name = "/home/maxim/file/library/mnc_net/data/test/2008_000116.mat"

#---------------------------------------------------------------------------------------------------

mcg_mat_name = scipy.io.loadmat('{0}'.format(mcg_mat_name))

print mcg_mat_name.keys()
# print type(mcg_mat_name['labels'])

array = mcg_mat_name['superpixels']
array[array > 900] = 200
print array
img = Image.fromarray(array)
img.save("1.png")

