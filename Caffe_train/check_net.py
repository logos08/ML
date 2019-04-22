import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import caffe
import time
import cv2
import os
import gc

#------------------------------------------------------------------------------

from math import sqrt
from lxml import etree
from PIL import Image, ImageDraw
from google.protobuf import text_format
from caffe.proto import caffe_pb2



#------------------------------------------------------------------------------

def get_labelname(labelmap, labels):

    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        # assert found == True
    return labelnames

#------------------------------------------------------------------------------

class CaffeDetection:

    def __init__(self, gpu_id, model_def, model_weights, nx, ny, labelmap_file):

        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.nx = nx
        self.ny = ny

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        # self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, frame, t = 0.3, topn = 999):

        '''
        SSD detection
        '''

        image = frame.astype(np.float32) / 255.0

        self.net.blobs['data'].reshape(1, 3, self.ny, self.nx)

        #Run the net and examine the top_k results
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        startT = time.time()

        # Forward pass.
        detections = self.net.forward()['detection_out']

        print time.time() - startT

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= t]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            if len(top_labels) > 0:
                label_name = top_labels[i]
            else:
                continue
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

#------------------------------------------------------------------------------
# Function compute metrics and write log
#------------------------------------------------------------------------------

def log(namePath, it, logPath, viewPath, classNames, t):

    prototxt = "prototxt/deploy.prototxt"
    labelmap = "prototxt/labelmaps.prototxt"
    snapshot = "snapshot/{}_iter_{}.caffemodel".format(classNames[0], it)

    nx = 512
    ny = 512

    classNum = 1

    #------------------------------------------------------------------------------

    with open(namePath, "r") as f:
        nameList = sorted([line[:-1] for line in f.readlines()])

    imgList = [name.split(" ")[0] for name in nameList]

    #------------------------------------------------------------------------------

    detect = CaffeDetection(0, prototxt, snapshot, nx, ny, labelmap)

    #------------------------------------------------------------------------------

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    fontType  = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7

    prev_value = 0

    for imgPath in imgList:

        print imgPath

        #----------------------------------------------------------------------

        imgArray = cv2.imread("{}".format(imgPath))

        height = imgArray.shape[0]
        width = imgArray.shape[1]

        cropNum = 16

        step_h = int(height / sqrt(cropNum))
        # step_w = 612
        step_w = int(width / sqrt(cropNum))

        coords_y = [[y, y + step_h] for y in range(height) if y % step_h == 0]
        coords_x = [[x, x + step_w] for x in range(width) if x % step_w == 0]

        warning_coords = [225, 270, 1020, 510]

        cv2.rectangle(
            imgArray,
            (warning_coords[0], warning_coords[1]),
            (warning_coords[2], warning_coords[3]),
            (0,0,255),
            3
            )

        total_rect = 0

        for x_pair in coords_x:

            for y_pair in coords_y:

                print "Processing crop: [{}:{}, {}:{}]".format(y_pair[0], y_pair[1], x_pair[0], x_pair[1])

                crop_img = imgArray[y_pair[0]:y_pair[1], x_pair[0]:x_pair[1]]

                res_crop_img = cv2.resize(crop_img, (512, 512))

                ny = res_crop_img.shape[0]
                nx = res_crop_img.shape[1]
                ch = res_crop_img.shape[2]

                #----------------------------------------------------------------------
                # Net
                #----------------------------------------------------------------------

                try:

                    resultList = detect.detect(res_crop_img, t = t)

                except IndexError:

                    continue

                #----------------------------------------------------------------------

                netArray = np.zeros(shape = (classNum, ny, nx))

                netVisArray = np.zeros(shape = res_crop_img.shape)

                for hItem in resultList:

                    name = hItem[-1]

                    xmin = int(round(hItem[0] * nx))
                    ymin = int(round(hItem[1] * ny))
                    xmax = int(round(hItem[2] * nx))
                    ymax = int(round(hItem[3] * ny))

                    xmin = xmin if(xmin >= 0) else 0
                    ymin = ymin if(ymin >= 0) else 0
                    xmax = xmax if(xmax < nx) else (nx - 1)
                    ymax = ymax if(ymax < ny) else (ny - 1)

                    P = hItem[-2]

                    if (name == classNames[0]):

                        netArray[0, ymin:ymax, xmin:xmax] = 1

                        center_x, center_y = (x_pair[0] + xmin / 1.6) + (((x_pair[0] + xmax / 1.6) - (x_pair[0] + xmin / 1.6)) / 2), (y_pair[0] + ymin / 2.84) + (((y_pair[0] + ymax / 2.84) - (y_pair[0] + ymin / 2.84)) / 2)

                        if center_x > warning_coords[0] and center_x < warning_coords[2] and center_y > warning_coords[1] and center_y < warning_coords[3]:

                            cv2.rectangle(
                                res_crop_img,
                                (xmin, ymin),
                                (xmax, ymax),
                                (0,0,255),
                                3
                                )

                            text = "P{0:.2f}".format(P)

                            cv2.putText(
                                res_crop_img,
                                text,
                                (xmin + 5, ymin + 15),
                                fontType,
                                fontScale,
                                (0,0,255),
                                2
                            )

                            total_rect += 1

                    # elif (name == classNames[1]):

                    #     netArray[0, ymin:ymax, xmin:xmax] = 1

                    #     cv2.rectangle(
                    #         res_crop_img,
                    #         (xmin, ymin),
                    #         (xmax, ymax),
                    #         (50,205,50),
                    #         2
                    #     )

                    #     # d = ((xmax - xmin) * (ymax - ymin)) / float(512 * 512)

                    #     # text = "L{0:.2f} d{1:.2f}".format(P, d)
                    #     text = "P{0:.2f}".format(P)

                    #     cv2.putText(
                    #         res_crop_img,
                    #         text,
                    #         (xmin + 5, ymin + 15),
                    #         fontType,
                    #         fontScale,
                    #         (50,205,50),
                    #         2
                    #     )

                        continue

                res_imgArray = cv2.resize(res_crop_img, (step_w, step_h))

                imgArray[y_pair[0]:y_pair[1], x_pair[0]:x_pair[1]] = res_imgArray
        
        total = "Peoples total: {}".format(total_rect)

        diff = "Previous frame difference: {}".format(0 - (prev_value - total_rect))

        prev_value = total_rect

        cv2.putText(
            imgArray,
            total,
            (1000, 690),
            fontType,
            fontScale,
            (0,0,255),
            2
            )

        cv2.putText(
            imgArray,
            diff,
            (600, 690),
            fontType,
            fontScale,
            (0,0,255),
            2
            )

        outPath = "{}/{}".format(viewPath, imgPath.split("/")[-1])

        cv2.imwrite(outPath, imgArray)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MAIN
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

caffe.set_mode_gpu()
caffe.set_device(0)

# caffe.set_mode_cpu()

classNames = [
    "people",
    ]

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

projectPath = os.path.dirname(os.path.realpath(__file__))

# input
solverPath    = "prototxt/solver.prototxt"
# weightPath    = "snapshot/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel"
# weightPath    = "snapshot/no_helmet_iter_0.caffemodel"
trainListPath = "data/list_train.txt"
validListPath = "data/list_valid.txt"

# output
trainLogPath  = "data/log_train.txt"
validLogPath  = "data/log_valid.txt"
hitLogPath = "data/log_hit.txt"
trainViewPath = "data/view_train"
validViewPath = "data/view_valid"

#------------------------------------------------------------------------------
# Load caffe model and weights
#------------------------------------------------------------------------------

# solver = caffe.get_solver(solverPath)
# solver.net.copy_from(weightPath)

#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------

log(
    namePath  = validListPath,
    it        = 0,
    logPath   = validLogPath,
    viewPath  = validViewPath,
    classNames = classNames,
    t         = 0.6
)
