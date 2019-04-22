import numpy as np
import sys
import cv2
import os

from lxml import etree
from PIL import Image, ImageDraw

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# MAIN
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

projectPath = os.path.dirname(os.path.realpath(__file__))

className_1 = "car"
className_2 = "truck"
className_3 = "bus"

part = "001"

#------------------------------------------------------------------------------

imgFldPath = "data/{}_rsize_img".format(part)
xmlFldPath = "data/{}_rsize_xml".format(part)
visFldPath = "data/{}_rsize_vis".format(part)

xmlList = sorted(os.listdir(xmlFldPath))

for xmlName in xmlList:

    name = xmlName.split(".")[0]

    imgPath = "{}/{}.png".format(imgFldPath, name)
    xmlPath = "{}/{}.xml".format(xmlFldPath, name)
    visPath = "{}/{}.png".format(visFldPath, name)

    visImage = cv2.imread(imgPath)

    #--------------------------------------------------------------------------
    # xml
    #--------------------------------------------------------------------------

    root = etree.parse(xmlPath)

    objectXpath = root.xpath('//annotation/object')

    nameXpath = root.xpath('//annotation/object/name')

    xminXpath = root.xpath('//annotation/object/bndbox/xmin')
    yminXpath = root.xpath('//annotation/object/bndbox/ymin')
    xmaxXpath = root.xpath('//annotation/object/bndbox/xmax')
    ymaxXpath = root.xpath('//annotation/object/bndbox/ymax')

    if(len(objectXpath) == 0):
        print "empty         : {}".format(xmlPath)

    for j in range(len(objectXpath)):
        name = str(nameXpath[j].text)
        xmin = int(xminXpath[j].text)
        ymin = int(yminXpath[j].text)
        xmax = int(xmaxXpath[j].text)
        ymax = int(ymaxXpath[j].text)

        if(name == className_1):
            print "{}            : {} [{:04}:{:04}, {:04}:{:04}]".format(
                className_1,
                xmlPath,
                ymin,
                ymax,
                xmin,
                xmax
            )

            cv2.rectangle(
                visImage,
                (xmin, ymin),
                (xmax, ymax),
                (0, 0, 255),
                2
            )
        elif (name == className_2):
            print "{}            : {} [{:04}:{:04}, {:04}:{:04}]".format(
                className_2,
                xmlPath,
                ymin,
                ymax,
                xmin,
                xmax
            )

            cv2.rectangle(
                visImage,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 255),
                2
            )
        elif (name == className_3):
            print "{}            : {} [{:04}:{:04}, {:04}:{:04}]".format(
                className_3,
                xmlPath,
                ymin,
                ymax,
                xmin,
                xmax
            )

            cv2.rectangle(
                visImage,
                (xmin, ymin),
                (xmax, ymax),
                (255, 255, 0),
                2
            )

        else:
            print "ERROR         : {}".format(xmlPath)

    cv2.imwrite(visPath, visImage)
