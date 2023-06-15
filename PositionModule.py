from ultralytics import YOLO

import cv2
import cvzone  ## for display and action
import math
import mediapipe as mp
import time


class PoseDetector():

    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose_mp = mp.solutions.pose
        self.pose = self.pose_mp.Pose(self.mode, self.model_complexity,  self.smooth_landmarks, self.enable_segmentation,
                                      self.smooth_segmentation,self.min_detection_confidence,
                                      self.min_tracking_confidence)  ## initialized the Hands with all the params with self
        self.draw_mp = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):

        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_RGB)
        if self.results.pose_landmarks:
            if draw:
                self.draw_mp.draw_landmarks(img, self.results.pose_landmarks, self.pose_mp.POSE_CONNECTIONS)

        return img

    def get_position_points(self, img, hands_numbers=0, draw=True):

        landmark_list = []
        for id, landmark in enumerate(self.results.pose_landmarks.landmark):
            height, weight, chanel = img.shape
            x_center, y_center = int((landmark.x * weight)), int((landmark.y * height))
            landmark_list.append([id, x_center, y_center])
            if draw:
                cv2.circle(img, (x_center, y_center), 10, color=(255, 0, 255))

        return landmark_list



def initialized():
    cap = cv2.VideoCapture("../PositionEstimation/Video_vision_project/yoga_video_2.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    previous_time = 0
    current_time = 0
    my_detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = my_detector.find_pose(img)
        position_list = my_detector.get_position_points(img)
        if len(position_list) != 0:
            print(position_list)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 255), thickness=3)
        cv2.imshow("my Video", img)
        cv2.waitKey(10)


if __name__ == '__main__':
    initialized()