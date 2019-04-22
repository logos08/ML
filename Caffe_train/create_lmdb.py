import shutil
import os

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MAIN
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

projectPath = os.path.dirname(os.path.realpath(__file__))

caffePath   = "/home/maxim/file/library/caffe/SSD_weiliu89"

#------------------------------------------------------------------------------

for stage in ["train", "valid"]:

    if os.path.exists("{}/data/lmdb_{}".format(projectPath, stage)):

        shutil.rmtree("{}/data/lmdb_{}".format(projectPath, stage))

    argsString = str()

    argsString += "{}/build/tools/convert_annoset".format(caffePath)
    argsString += " --anno_type=detection"
    argsString += " --label_type=xml"
    argsString += " --label_map_file={}/prototxt/labelmaps.prototxt".format(projectPath)
    argsString += " --check_label=True"
    argsString += " --min_dim=0"
    argsString += " --max_dim=0"
    argsString += " --resize_height=0"
    argsString += " --resize_width=0"
    argsString += " --backend=lmdb"
    argsString += " --shuffle=False"
    argsString += " --check_size=False"
    argsString += " --encode_type=jpg"
    argsString += " --encoded=True"
    argsString += " --gray=False"
    argsString += " {}/".format(projectPath)
    argsString += " {}/data/list_{}.txt".format(projectPath, stage)
    argsString += " {}/data/lmdb_{}".format(projectPath, stage)

    process = os.system(argsString)