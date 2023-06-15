from ultralytics import YOLO

import cv2
import cvzone ## for display and action
import math
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) ## for My Web Cam
#cap = cv2.VideoCapture("../Videos/motorbikes.mp4") ## for the Video File Dont set the size coz the video it's already seting
cap.set(3, 1280)
cap.set(4, 720)
hand_mp = mp.solutions.hands
hands = hand_mp.Hands() ## by default he detect 2 hands with 50% confidence
draw_mp = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    success, img = cap.read()


    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_RGB)

    ## if it's Detected

    if results.multi_hand_landmarks:
        for hands_landM_single in results.multi_hand_landmarks:
            for id, landmark in enumerate(hands_landM_single.landmark):

                height, weight, chanel = img.shape
                x_center, y_center = int((landmark.x * weight)), int((landmark.y*height)) ## take the centre of x, y in each landmark

                print(id, x_center, y_center)
                draw_mp.draw_landmarks(img, hands_landM_single, hand_mp.HAND_CONNECTIONS) ## Connecte the points on the Hand

                if id == 0:
                    cv2.circle(img, (x_center, y_center), 25, color=(255, 0, 255))



    current_time = time.time()
    fps = 1 / (current_time-previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 255), thickness=3)

    cv2.imshow("Myweb Cam ", img)
    cv2.waitKey(1)

