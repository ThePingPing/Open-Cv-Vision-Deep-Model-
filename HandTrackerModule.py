from ultralytics import YOLO

import cv2
import cvzone  ## for display and action
import math
import mediapipe as mp
import time


class HandDetector():

    def __init__(self, mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mode = mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hand_mp = mp.solutions.hands
        self.hands = self.hand_mp.Hands(self.mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)  ## initialized the Hands with all the params with self
        self.draw_mp = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_RGB)

        ## if it's Detected

        if self.results.multi_hand_landmarks:
            for hands_landM_single in self.results.multi_hand_landmarks:
                if draw:
                    self.draw_mp.draw_landmarks(img, hands_landM_single,
                                                self.hand_mp.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hands_numbers=0, draw=True):

        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hands = self.results.multi_hand_landmarks[hands_numbers] ## the current hand

            for id, landmark in enumerate(my_hands.landmark):
                height, weight, chanel = img.shape
                x_center, y_center = int((landmark.x * weight)), int((landmark.y * height))  ## take the centre of x, y in each landmark

                #print(id, x_center, y_center)  ## Connecte the points on the Hand
                landmark_list.append([id, x_center, y_center])
                if draw:
                    cv2.circle(img, (x_center, y_center), 25, color=(255, 0, 255))

        return landmark_list


def initialized():
    cap = cv2.VideoCapture(0)  ## for My Web Cam

    my_detector = HandDetector()  ## create my Detector


    cap.set(3, 1280)
    cap.set(4, 720)

    previous_time = 0
    current_time = 0

    while True:
        success, img = cap.read()

        img = my_detector.find_hands(img)  ## call The Fonction with the img
        position_list = my_detector.find_position(img)

        if len(position_list) != 0:
            print(position_list[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 255), thickness=3)

        cv2.imshow("Myweb Cam ", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    initialized()
