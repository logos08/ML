import cv2
import os
from scipy.ndimage import rotate
from xml import etree
import xml.etree.cElementTree as ET

count = 1
angles = [
    0,
    ]

for ang in angles:

    filenames_jpg = sorted(list(os.walk("people_loss/{}/".format(ang)))[0][2])
    filename_background = "background.png"
    filename_xml = "test.xml"

    #ext_x = 130
    ext_y = 70

    def generator_data(offset_x, offset_y):
    
        for name in filenames_jpg:
        
            for angle in range(4):
            
                background = cv2.imread(filename_background)
        
                old_jpg_path = "people_loss/{0}/{1}".format(ang, name)
        
                img = cv2.imread(old_jpg_path, -1)
                rotate_img = rotate(img, 90 * angle)

                y1, y2 = offset_y, offset_y + rotate_img.shape[0]
                x1, x2 = offset_x, offset_x + rotate_img.shape[1]

                alpha_s = rotate_img[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(3):
                    background[y1:y2, x1:x2, c] = (alpha_s * rotate_img[:, :, c] + alpha_l * background[y1:y2, x1:x2, c])

                global count
                new_jpg_path = "002_fsize_img/{0:07}.jpg".format(count)
            

                cv2.imwrite(new_jpg_path, background)
            
                old_path_xml = filename_xml
                tree = ET.parse(old_path_xml)
                root = tree.getroot()
                for filename in root.iter("filename"):
                    filename.text = "{0:07}.jpg".format(count)
                for width in root.iter("width"):
                    width.text = "1280"
                for height in root.iter("height"):
                    height.text = "720"
                for box in root.iter("bndbox"):
                    for x in box.iter("xmin"):
                        x.text = str(x1)
                    for y in box.iter("ymin"):
                        y.text = str(y1)
                    for x in box.iter("xmax"):
                        x.text = str(x2)
                    for y in box.iter("ymax"):
                        y.text = str(y2)
            
                tree = ET.ElementTree(root)
    
                new_xml_file = "002_fsize_xml/{0:07}.xml".format(count)  
                tree.write(new_xml_file)
            
                print(str(count) + ", " + str(ang))
                count += 1
                   
    for x in range(5):
        ext_x = 100
        for x in range(5):
            generator_data(ext_x, ext_y)
            ext_x += 200
        ext_y += 60

print("Done")  


