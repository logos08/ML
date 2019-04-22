import numpy as np
import sys
import cv2
import os

from lxml import etree

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

projectPath = os.path.dirname(os.path.realpath(__file__))

new_cols = 1280
new_rows = 720

#------------------------------------------------------------------------------

nameList = sorted(os.listdir("dataset/img_fullsize"))
nameList = [name.split(".")[0] for name in nameList]

#------------------------------------------------------------------------------

for name in nameList:

    #--------------------------------------------------------------------------

    print name

    #--------------------------------------------------------------------------
    # Defined paths
    #--------------------------------------------------------------------------

    img_fsize_path = "dataset/img_fullsize/{}.png".format(name)
    # xml_fsize_path = "data/{}_fsize_xml/{}.xml".format(part, name)

    img_rsize_path = "dataset/img_resize/{}.png".format(name)
    # xml_rsize_path = "data/{}_rsize_xml/{}.xml".format(part, name)

    #--------------------------------------------------------------------------
    # Change xml
    #--------------------------------------------------------------------------

    # root = etree.parse(xml_fsize_path)

    # filenameXPath = root.xpath("//annotation/filename")

    # filenameXPath[0].text = "{}.png".format(name)

    # colsXPath = root.xpath("//annotation/size/width")
    # rowsXPath = root.xpath("//annotation/size/height")

    # old_cols = int(colsXPath[0].text)
    # old_rows = int(rowsXPath[0].text)

    # colsXPath[0].text = str(new_cols)
    # rowsXPath[0].text = str(new_rows)

    # nameXPath = root.xpath("//annotation/object/name")

    # xminXPath = root.xpath("//annotation/object/bndbox/xmin")
    # yminXPath = root.xpath("//annotation/object/bndbox/ymin")
    # xmaxXPath = root.xpath("//annotation/object/bndbox/xmax")
    # ymaxXPath = root.xpath("//annotation/object/bndbox/ymax")

    # scaleX = new_cols / float(old_cols)
    # scaleY = new_rows / float(old_rows)

    # for i in range(len(nameXPath)):
    #     xminXPath[i].text = str(int(scaleX * int(xminXPath[i].text)))
    #     yminXPath[i].text = str(int(scaleY * int(yminXPath[i].text)))
    #     xmaxXPath[i].text = str(int(scaleX * int(xmaxXPath[i].text)))
    #     ymaxXPath[i].text = str(int(scaleY * int(ymaxXPath[i].text)))

    # #--------------------------------------------------------------------------
    # # Write xml
    # #--------------------------------------------------------------------------

    # root.write(xml_rsize_path, pretty_print = True)

    # #--------------------------------------------------------------------------
    # # Change and Write img
    # #--------------------------------------------------------------------------

    old_img = cv2.imread(img_fsize_path)
    new_img = cv2.resize(old_img, (new_cols, new_rows))
    cv2.imwrite(img_rsize_path, new_img)
