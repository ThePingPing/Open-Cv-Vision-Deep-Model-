from ultralytics import YOLO
import cv2

model = YOLO('../YoloWeights/yolov8l.pt') ## give the wights
results = model("Images/3.png", show=True)
cv2.waitKey(0) ## kep it open while you dont close it