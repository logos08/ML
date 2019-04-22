import numpy as np
import random
import scipy.misc
import sys
import os

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MAIN
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

projectPath = os.path.dirname(os.path.realpath(__file__))

classDict = {
    "001" : 3412,
    "002" : 2200,
}

trainPart = 0.8
validPart = 0.2

trainList = []
validList = []

#------------------------------------------------------------------------------

trainPath = "data/list_train.txt" # output
validPath = "data/list_valid.txt" # output

for key, item in classDict.iteritems():

    imgFldPath = "data/{}_rsize_img".format(key)
    xmlFldPath = "data/{}_rsize_xml".format(key)

    nameList = [xml.split(".")[0] for xml in sorted(os.listdir(xmlFldPath))]
    randList = random.sample(range(len(nameList)), len(nameList))[0:item]
    nameList = [nameList[i] for i in randList]

    pathList = [
        "{}/{}.png {}/{}.xml".format(imgFldPath, name, xmlFldPath, name)
        for name in nameList
    ]

    p0 = 0
    p1 = int(p0 + trainPart * len(pathList))
    p2 = len(pathList)

    trainList.extend(sorted(pathList[p0:p1]))
    validList.extend(sorted(pathList[p1:p2]))

#------------------------------------------------------------------------------
# Print len lists
#------------------------------------------------------------------------------

print "list_train.txt: {}".format(len(trainList))
print "list_valid.txt: {}".format(len(validList))

#------------------------------------------------------------------------------
# Write lists
#------------------------------------------------------------------------------

with open(trainPath, "w") as f:
    for item in trainList:
        print >> f, item

with open(validPath, "w") as f:
    for item in validList:
        print >> f, item

#------------------------------------------------------------------------------
# Create size files
#------------------------------------------------------------------------------

for stage in ["train", "valid"]:

    imgListPath = "data/list_{}.txt".format(stage) # input
    imgSizeFile = "data/size_{}.txt".format(stage) # output

    with open(imgListPath) as f:
        imgPathList = f.readlines()

    imgPathList = [name.split(" ")[0] for name in imgPathList]
    imgSizeFile = open(imgSizeFile, "w")

    for imgPath in imgPathList:
        imgName = imgPath.split("/")[-1].split(".")[0]
        imgArray = scipy.misc.imread(imgPath)
        imgShape = imgArray.shape
        imgSizeString = "{} {} {}\n".format(imgName, imgShape[0], imgShape[1])
        imgSizeFile.write(imgSizeString)

    imgSizeFile.close()
