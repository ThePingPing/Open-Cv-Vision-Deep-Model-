import torch
from ultralytics import YOLO
import numpy as np
import cv2
import cvzone ## for display and action
import math
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4") ## for the Video File Dont set the size coz the video it's already seting
model = YOLO('../YoloWeights/yolov8m.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")

""" Create a instance for the tracker in the sort import """

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

"""Create a instant for the counter so implement a limites"""

limits = [400, 297, 673, 297]
total_count = []

while True:
    success, img = cap.read()
    select_image_region = cv2.bitwise_and(img, mask)
    results = model(select_image_region, stream=True) ## change "img" to "select_image_region" selct and show detection only in the region


    detections = torch.tensor(np.empty((0, 5))) ## that's to feed the update fonction
    for r in results:
        boxes = r.boxes
        for box in boxes:
            """ For Open Cv implementtation Bounding Box """
            #x1, y1, x2, y2 = box.xyxy[0]
            #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            #print(x1, y1, x2, y2)

            """ For Cv Zone implementtation """
            x1, y1, x2, y2 = box.xyxy[0]

            weight, height = x2 - x1, y2 - y1
            x1, y1, weight, height = int(x1), int(y1), int(weight), int(height)



            """ Confidence """

            conf_value = (math.ceil(box.conf[0] * 100) / 100)
            #print(conf_box)

            #cvzone.putTextRect(img, f'{conf_box}', (max(x1, 0), max(50, y1)))

            """ Classification Per Names"""

            class_box = int(box.cls[0]) ## if you want to show the Name of the object was detected convert to integer
            current_cls = classNames[class_box] ## take only the currente element


            if current_cls == "car" or current_cls == "bus" or current_cls == "motorbike" or current_cls == "truck" and conf_value > 0.3: ## check the element
                #cvzone.putTextRect(img, f'{current_cls}, {conf_value}', (max(x1, 0), max(50, y1)), scale=1, thickness=1,offset=3)  ## and Here also put ClassNames
               # cvzone.cornerRect(img, (x1, y1, weight, height), l=10 , rt=5)

                current_tensor = torch.tensor([x1, y1, x2, y2, conf_value]) ## implement the arrays
                detections = torch.vstack((detections, current_tensor))


    results_tracker = tracker.update(detections)
    my_line = cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 4)

    """The real number of the id it's not important , you have to check the same car dont get a new id """

    for res_track in results_tracker:
        x1, y1, x2, y2, id = res_track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        weight, height = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, weight, height), l=10, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(x1, 0), max(50, y1)), scale=2, thickness=3,
                           offset=10)  ## and Here also put ClassNames

        x_centre, y_centre = int((x1+x2) /2), int((y1 + y2)/2) ## that's the centre for each car on the frame detection
        cv2.circle(img=img, center=(x_centre, y_centre), radius=5, color=(255, 0, 255) , thickness=cv2.FILLED)


        print("The Centre X is :" ,  x_centre, "The Centre y is :",  y_centre)

        if(limits[0] < x_centre < limits[2] and limits[1] -15 < y_centre< limits[1] + 15):

            if total_count.count(id) == 0: ## you have to check if the id is already present
                total_count.append(id) ## if not put it on the list
                my_line = cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 4)

        cvzone.putTextRect(img, f'countCar {len(total_count)}', (50, 50))



        print(res_track)


    cv2.imshow("Image", img)
    #cv2.imshow("ImageRegion", select_image_region) ## ETREMELY IMPORTANT you have to be sure they have the same Size the Video or a picture from the video and the mask
    cv2.waitKey(1)