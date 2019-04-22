import cv2
import os
from scipy.ndimage import rotate
from xml import etree
import xml.etree.cElementTree as ET


wrong_path = "/home/maxim/file/project/pig_v5_affordance_net/xml_wrong/"
right_path = "/home/maxim/file/project/pig_v5_affordance_net/xml_right/"

filenames_xml = sorted(list(os.walk(wrong_path))[0][2])

foldername = "IIT2017"

def generator_xml():

    count = 5
    
    for name in filenames_xml:

        old_path_xml = wrong_path + name

        tree = ET.parse(old_path_xml)
        root = tree.getroot()

        for folder in root.iter("folder"):
            folder.text = foldername

        source = tree.find('source')
        root.remove(source)

        segmented = tree.find('segmented')
        root.remove(segmented)
            
        tree = ET.ElementTree(root)
    
        new_xml_file = "{0}{1}.xml".format(right_path, name.split(".")[0])
        tree.write(new_xml_file)

        body = ""
        with open(new_xml_file, "r") as f:
            content = f.read()
            body = content.replace("    ", "  ")

        with open(new_xml_file, "w") as f:
            f.write(body)
            
        count += 1

if __name__ == "__main__":
    generator_xml()

print("Done")  


