import cv2
import numpy as np
import time
import HandTrackerModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


####################
cam_weight, cam_height = 1280, 720
####################
cap = cv2.VideoCapture(0)
cap.set(3, cam_weight)
cap.set(4, cam_height)
previous_time = 0
current_time = 0
my_detector = htm.HandDetector(min_detection_confidence=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()

min_volume, max_volume = volume_range[0], volume_range[1]

line_to_volume = 0
bar_volume = 0




while True:
    success, img = cap.read()
    img = my_detector.find_hands(img)
    position_list = my_detector.find_position(img, draw=False)
    if len(position_list) != 0:
        #print(position_list[4], position_list[8])
        x1, y1 = position_list[4][1], position_list[4][2]
        x2, y2 = position_list[8][1], position_list[8][2]
        x_center, y_center = ((x1 + x2) // 2), ((y1 + y2) // 2)

        cv2.circle(img, (x1, y1), 15, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, (x2, y2), 15, color=(0, 0, 255), thickness=-1)
        cv2.line(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3)
        cv2.circle(img, (x_center, y_center), 15, color=(0, 0, 255), thickness=-1)

        lenght_line = math.hypot(x2-x1, y2- y1)
        #print(lenght_line)

        ## now Convert the Lenght Line on Volume Range use Numpy to converte
        line_to_volume = np.interp(lenght_line, [50, 350], [min_volume, max_volume])
        bar_volume = np.interp(lenght_line, [50, 350], [400, 150])

        print(lenght_line, line_to_volume)
        volume.SetMasterVolumeLevel(line_to_volume, None)
        if lenght_line < 50:
            cv2.circle(img, (x_center, y_center), 15, color=(0, 255, 0), thickness=-1)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(bar_volume)), (85, 400), (0, 255, 0), thickness=-1)




    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 0, 255), thickness=3)
    cv2.imshow("My image", img)
    cv2.waitKey(1)

