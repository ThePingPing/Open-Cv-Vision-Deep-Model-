from ultralytics import YOLO

import cv2
import cvzone  ## for display and action
import math
import mediapipe as mp
import time


cap = cv2.VideoCapture("../PositionEstimation/Video_vision_project/yoga_video_2.mp4")
cap.set(3, 1280)
cap.set(4, 720)
previous_time = 0
current_time = 0

pose_mp = mp.solutions.pose
pose = pose_mp.Pose()
draw_mp = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## convert the img to use the mediapip Lib
    results = pose.process(img_RGB)
    if results.pose_landmarks:
        draw_mp.draw_landmarks(img, results.pose_landmarks, pose_mp.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            height, weight, chanel = img.shape
            x_center, y_center = int((landmark.x * weight)), int((landmark.y * height))
            cv2.circle(img, (x_center, y_center), 10, color=(255, 0, 255))



    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 255), thickness=3)
    cv2.imshow("my Video", img)
    cv2.waitKey(10)