import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import caffe
import time
import cv2
import os
import gc

#------------------------------------------------------------------------------

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
            try:
                label_name = top_labels[i]
            except IndexError:
                if len(top_labels) > 0:
                    label_name = top_labels[-1]
                else:
                    label_name = "unknown_labelname"

            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

# def hit_log(hits_dict, hitLogPath, it, stage):
#     with open(hitLogPath, "a") as f:
#         total_loss = 0
#         loss_jpg = []
#         for img, status in hits_dict.items():
#             if status > 2:
#                 total_loss += 1
#                 loss_jpg.append(img)
#         f.write("iteration: {} net loss is {} on {} stage\n".format(it, total_loss, stage))
#         f.write("loss on jpg`s: {}\n".format(loss_jpg))
#         f.write("NEXT STEP {} \n\n\n".format("========" * 20))

# def intersection(a,b):
#     x = max(a[0], b[0])
#     y = max(a[1], b[1])
#     w = min(a[0]+a[2], b[0]+b[2]) - x
#     h = min(a[1]+a[3], b[1]+b[3]) - y
#     if w<0 or h<0: 
#         return 0
#     return 1

# def step_cutter(list_of_rects, step, total_steps):
#     first_width = 512
#     first_height = 512
#     returned_list = []

#     img_step = first_width / (total_steps * 2)

#     x1, y1, x2, y2 = 0 + (img_step * step), 0, first_width - (img_step * step), first_height
#     main_frame = [x1, y1, x2, y2]
#     for rect in list_of_rects:
#         if intersection(rect, main_frame):
#             returned_list.append(rect)
#         else:
#             pass
#     return returned_list


# def check_hits(rectangles_xml, rectangles_net, hitLogPath, it, stages):
#     for stage in range(stages):
#         hits = {}
#         for key_xml in rectangles_xml.keys():
#             intersect = 0
#             status = 0
#             list_of_xmls = step_cutter(rectangles_xml[key_xml], stage, stages)
#             list_of_net = step_cutter(rectangles_net[key_xml], stage, stages)
#             total_obj = len(list_of_xmls)
#             for xml_rect in list_of_xmls:
#                 for net_rect in list_of_net:
#                     i = intersection(xml_rect, net_rect)
#                     if i > 0 :
#                         intersect += i
#                         break
#             if intersect < total_obj:
#                 status = total_obj - intersect + 4
#             elif intersect == total_obj:
#                 status = 1
#             elif intersect > total_obj:
#                 status = (intersect - total_obj) * -1
#             hits[key_xml] = status
#         hit_log(hits, hitLogPath, it, stage)



#------------------------------------------------------------------------------
# Function compute metrics and write log
#------------------------------------------------------------------------------

def log(namePath, it, logPath, viewPath, classNames, t):

    prototxt = "prototxt/deploy.prototxt"
    labelmap = "prototxt/labelmaps.prototxt"
    snapshot = "snapshot/{}_iter_{}.caffemodel".format(classNames[0], it)

    nx = 512
    ny = 512

    classNum = 2
    rectangles_xml = {}
    rectangles_net = {}
    num_of_stages = 5

    class_1 = 0
    class_2 = 0

    #------------------------------------------------------------------------------

    with open(namePath, "r") as f:
        nameList = sorted([line[:-1] for line in f.readlines()])

    imgList = [name.split(" ")[0] for name in nameList]
    xmlList = [name.split(" ")[1] for name in nameList]

    #------------------------------------------------------------------------------

    detect = CaffeDetection(1, prototxt, snapshot, nx, ny, labelmap)

    #------------------------------------------------------------------------------

    fp = 0
    fn = 0
    tp = 0
    tn = 0

    fontType  = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    for imgPath, xmlPath in zip(imgList, xmlList):

        print imgPath, xmlPath

        #----------------------------------------------------------------------

        imgArray = cv2.imread("{}".format(imgPath))

        ny = imgArray.shape[0]
        nx = imgArray.shape[1]
        ch = imgArray.shape[2]

        total_rect_xml = []
        total_rect_net = []

        #----------------------------------------------------------------------
        # XML
        #----------------------------------------------------------------------

        root = etree.parse("{}".format(xmlPath))

        objectXpath = root.xpath('//annotation/object')

        nameXpath = root.xpath('//annotation/object/name')

        xminXpath = root.xpath('//annotation/object/bndbox/xmin')
        yminXpath = root.xpath('//annotation/object/bndbox/ymin')
        xmaxXpath = root.xpath('//annotation/object/bndbox/xmax')
        ymaxXpath = root.xpath('//annotation/object/bndbox/ymax')

        xmlArray = np.zeros(shape = (classNum, ny, nx))

        for j in range(len(objectXpath)):

            name = str(nameXpath[j].text)

            xmin = int(xminXpath[j].text)
            ymin = int(yminXpath[j].text)
            xmax = int(xmaxXpath[j].text)
            ymax = int(ymaxXpath[j].text)

            xmin = xmin if(xmin >= 0) else 0
            ymin = ymin if(ymin >= 0) else 0
            xmax = xmax if(xmax < nx) else (nx - 1)
            ymax = ymax if(ymax < ny) else (ny - 1)

            text = "xml"

            if(name == classNames[0]):

                xmlArray[0, ymin:ymax, xmin:xmax] = 1

                total_rect_xml.append([xmin, ymin, xmax, ymax])
                
                cv2.rectangle(
                    imgArray,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    imgArray,
                    text,
                    (xmin + 5, ymin + 15),
                    fontType,
                    fontScale,
                    (0, 255, 0),
                    2
                )

                continue

            if (name == classNames[1]):

                xmlArray[1, ymin:ymax, xmin:xmax] = 1

                total_rect_xml.append([xmin, ymin, xmax, ymax])
                
                cv2.rectangle(
                    imgArray,
                    (xmin, ymin),
                    (xmax, ymax),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    imgArray,
                    text,
                    (xmin + 5, ymin + 15),
                    fontType,
                    fontScale,
                    (0, 255, 255),
                    2
                )

                continue

        rectangles_xml[imgPath.split("/")[-1]] = total_rect_xml
        #----------------------------------------------------------------------
        # Net
        #----------------------------------------------------------------------

        resultList = detect.detect(imgArray, t = t)

        #----------------------------------------------------------------------

        netArray = np.zeros(shape = (classNum, ny, nx))

        netVisArray = np.zeros(shape = imgArray.shape)

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

            text = "net: P {:.2f}".format(P)

            if (name == classNames[0]):

                class_1 += 1

                netArray[0, ymin:ymax, xmin:xmax] = 1

                total_rect_net.append([xmin, ymin, xmax, ymax])

                cv2.rectangle(
                    imgArray,
                    (xmin, ymin),
                    (xmax, ymax),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    imgArray,
                    text,
                    (xmin + 5, ymin + 15),
                    fontType,
                    fontScale,
                    (255,   0,   0),
                    2
                )

                continue

            if (name == classNames[1]):

                class_2 += 1

                netArray[1, ymin:ymax, xmin:xmax] = 1

                total_rect_net.append([xmin, ymin, xmax, ymax])

                cv2.rectangle(
                    imgArray,
                    (xmin, ymin),
                    (xmax, ymax),
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    imgArray,
                    text,
                    (xmin + 5, ymin + 15),
                    fontType,
                    fontScale,
                    (255,   0,   255),
                    2
                )

                continue

        rectangles_net[imgPath.split("/")[-1]] = total_rect_net


        #----------------------------------------------------------------------
        # Compute metrics
        #----------------------------------------------------------------------

        for i in range(classNum):

            xmlArrayNot = np.logical_not(xmlArray[i])
            netArrayNot = np.logical_not(netArray[i])

            fpArray = np.logical_and(xmlArrayNot, netArray[i])
            fnArray = np.logical_and(xmlArray[i], netArrayNot)
            tpArray = np.logical_and(xmlArray[i], netArray[i])
            tnArray = np.logical_and(xmlArrayNot, netArrayNot)

            fp += np.sum(fpArray) / float(classNum * nx * ny)
            fn += np.sum(fnArray) / float(classNum * nx * ny)
            tp += np.sum(tpArray) / float(classNum * nx * ny)
            tn += np.sum(tnArray) / float(classNum * nx * ny)

        #----------------------------------------------------------------------
        # View save
        #----------------------------------------------------------------------

        outPath = "{}/{}".format(viewPath, imgPath.split("/")[-1])

        cv2.imwrite(outPath, imgArray)

    #--------------------------------------------------------------------------
    # Compute metrics
    #--------------------------------------------------------------------------

    fp /= float(len(nameList))
    fn /= float(len(nameList))
    tp /= float(len(nameList))
    tn /= float(len(nameList))

    ac = tp + tn
    pr = tp / (tp + fp + 1e-10)
    rc = tp / (tp + fn + 1e-10)
    iu = tp / (tp + fp + fn + 1e-10)

    #--------------------------------------------------------------------------
    # Write log
    #--------------------------------------------------------------------------

    with open(logPath, "a") as f:
        f.write("it {:06} ".format(it))
        f.write("fp {:.3f} ".format(fp))
        f.write("fn {:.3f} ".format(fn))
        f.write("tp {:.3f} ".format(tp))
        f.write("tn {:.3f} ".format(tn))
        f.write("ac {:.3f} ".format(ac))
        f.write("pr {:.3f} ".format(pr))
        f.write("rc {:.3f} ".format(rc))
        f.write("iu {:.3f}\n".format(iu))
        f.write("Class 'pig': {} ".format(class_1))
        f.write("Class 'people': {}\n".format(class_2))

    # check_hits(rectangles_xml, rectangles_net, hitLogPath, it, num_of_stages)


#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# MAIN
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

caffe.set_mode_gpu()
caffe.set_device(1)

# caffe.set_mode_cpu()

classNames = [
    "pig",
    "people",
    ]

#------------------------------------------------------------------------------
# Paths
#------------------------------------------------------------------------------

projectPath = os.path.dirname(os.path.realpath(__file__))

# input
solverPath    = "prototxt/solver.prototxt"
# weightPath    = "snapshot/VGG_VOC0712_SSD_512x512_ft_iter_120000.caffemodel"
weightPath    = "snapshot/pig_32.caffemodel"
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

solver = caffe.get_solver(solverPath)
solver.net.copy_from(weightPath)

#------------------------------------------------------------------------------
# Train
#------------------------------------------------------------------------------

# log(
#     namePath  = trainListPath,
#     it        = 0,
#     logPath   = trainLogPath,
#     viewPath  = trainViewPath,
#     className = className,
#     t         = 0.6
# )

# log(
#     namePath  = validListPath,
#     it        = 0,
#     logPath   = validLogPath,
#     viewPath  = validViewPath,
#     className = className,
#     t         = 0.6
# )

for i in range(200):

    solver.step(1000)

    it = 1000 * (i + 1)

    log(
        namePath  = trainListPath,
        it        = it,
        logPath   = trainLogPath,
        viewPath  = trainViewPath,
        classNames = classNames,
        t         = 0.5
    )

    log(
        namePath  = validListPath,
        it        = it,
        logPath   = validLogPath,
        viewPath  = validViewPath,
        classNames = classNames,
        t         = 0.5
    )
