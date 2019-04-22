import os
import sys
import cv2
import json

json_dir = "json`s/"
new_images_dir = "viz_points/"

parts_dict = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel",
}

colors = {
    "Nose": (255, 255, 255),
    "Neck": (128, 128, 128),
    "RShoulder": (0, 255, 255),
    "RElbow": (0, 170, 170),
    "RWrist": (0, 85, 85),
    "LShoulder": (0, 255, 0),
    "LElbow": (0, 170, 0),
    "LWrist": (0, 85, 0),
    "MidHip": (255, 128, 0),
    "RHip": (255, 255, 0),
    "RKnee": (213, 213, 0),
    "RAnkle": (170, 170, 0),
    "LHip": (255, 0, 0),
    "LKnee": (213, 0, 0),
    "LAnkle": (170, 0, 0),
    "REye": (0, 0, 255),
    "LEye": (255, 0, 255),
    "REar": (0, 0, 123),
    "LEar": (123, 0, 123),
    "LBigToe": (128, 0, 0),
    "LSmallToe": (85, 0, 0),
    "LHeel": (43, 0, 0),
    "RBigToe": (128, 128, 0),
    "RSmallToe": (85, 85, 0),
    "RHeel": (43, 43, 0),
}

parts_pairs = (
    ("Neck", "MidHip"),
    ("Neck", "RShoulder"),
    ("Neck", "LShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("MidHip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("MidHip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("Neck", "Nose"),
    ("Nose", "REye"),
    ("REye", "REar"),
    ("Nose", "LEye"),
    ("LEye", "LEar"),
    ("LAnkle", "LBigToe"),
    ("LBigToe", "LSmallToe"),
    ("LAnkle", "LHeel"),
    ("RAnkle", "RBigToe"),
    ("RBigToe", "RSmallToe"),
    ("RAnkle", "RHeel"),
)


# MAIN --------------------------------------------------------------------------------------------------

list_json = sorted(list(os.walk(json_dir))[0][2])

for x,j in enumerate(list_json):

    errors = []

    json_string = open(json_dir + j)
    data = json.load(json_string)

    files_list = sorted(data["_via_img_metadata"].keys())

    for file in files_list:

        total_coords = {}
        person_ids = []

        img_name = "{}".format(file.split(".")[0])
        original_img = cv2.imread("{}/{}.jpg".format(x + 1, img_name))

        for shape_attr in data["_via_img_metadata"][file]["regions"]:

            try:

                person_id = shape_attr["region_attributes"]["PeopleID"].strip()
            
                if person_id not in total_coords.keys():
                    total_coords[person_id] = {}
                    person_ids.append(person_id)

                part = parts_dict[int(shape_attr["region_attributes"]["TypePoint"])]
                color = colors[part]

                cv2.circle(
                    original_img,
                    (shape_attr["shape_attributes"]["cx"],shape_attr["shape_attributes"]["cy"]), 
                    5, 
                    color, 
                    -1
                    )
            
                fontType  = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
            
                cv2.putText(
                    original_img, 
                    part, 
                    (shape_attr["shape_attributes"]["cx"] + 5,shape_attr["shape_attributes"]["cy"] + 5), 
                    fontType, 
                    fontScale, 
                    color
                    )

                total_coords[person_id][part] = (
                    shape_attr["shape_attributes"]["cx"], 
                    shape_attr["shape_attributes"]["cy"]
                    )

                print "filename: {}.jpg, x: {}, y: {}, TypePoint: {}".format(
                    file, 
                    shape_attr["shape_attributes"]["cx"], 
                    shape_attr["shape_attributes"]["cy"], 
                    shape_attr["region_attributes"]["TypePoint"]
                    )
            except KeyError as e:

                errors.append(e.message + " in file {}.jpg".format(img_name))
                continue


        for _id in person_ids:

            for pair in parts_pairs:

                if pair[0] in total_coords[_id] and pair[1] in total_coords[_id]:

                    coord_1 = total_coords[_id][pair[0]]
                    coord_2 = total_coords[_id][pair[1]]
                    color = colors[pair[1]]

                    cv2.line(original_img,coord_1,coord_2,color,5)

        print "\n========================Next file===============================\n"
        
        cv2.imwrite("{}/{}_{}.png".format(new_images_dir, img_name, x + 1), original_img)

for er in errors:

    print "Check: {}".format(er)

print "\nDone!"