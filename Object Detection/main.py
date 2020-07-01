from cv2 import cv2
import numpy as np

# webCamFeed = False
# path = '1.jpg'
cap = cv2.VideoCapture(0)
wht = 320
confThreshold = 0.5
nmsThreshold = 0.2

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(f'Class Names are:- {classNames}')
# print(f'Length of Class Names are:- {len(classNames)}')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(f'Lenght of the bounding box:- {len(bbox)}.')
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x, y, w, h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    # if webCamFeed:
    success, img = cap.read()
    # else: img = cv2.imread(path)
    blob = cv2.dnn.blobFromImage(img, 1/255, (wht, wht), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    # print(f'Layers in the model:- {layerNames}')
    # print(f'Layers:- {net.getUnconnectedOutLayers()}')
    outputNames = [(layerNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    #  print(f'Output Names:- {outputNames}')
    outputs = net.forward(outputNames)
    # print(f'Output of 1:- {outputs[0].shape}')
    # print(f'Output of 1:- {outputs[1].shape}')
    # print(f'Output of 1:- {outputs[2].shape}')
    # print(f'All the features with the predictions: {outputs[0][0]}')
    findObjects(outputs, img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)