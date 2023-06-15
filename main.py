import cv2
import numpy as np
import wget as wget
from cv2.data import haarcascades
from matplotlib import pyplot as plt
from sklearn.datasets import images

def download_data():
    url1 = 'https://moderncomputervision.s3.eu-west-2.amazonaws.com/videos.zip'
    filename1 = wget.download(url1)


def image_show(title="Image", image=None, size=10):
    dim_row, dim_col = image.shape[0], image.shape[1]
    aspect_ratio = dim_col / dim_row
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


def video_recon():
    cap_video = cv2.VideoCapture('videos/walking.mp4')
    body_classifier = cv2.CascadeClassifier('haarcascades/Haarcascades/haarcascade_fullbody.xml')
    ret, frame = cap_video.read()

    if ret:

        # Grayscale our image for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our body classifier
        bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

        # Extract bounding boxes for any bodies identified
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Release our video capture
    cap_video.release()
    image_show("Pedestrian Detector", frame)


def video_recon_body_multi_frame():
    cap_video = cv2.VideoCapture('videos/walking.mp4')
    body_classifier = cv2.CascadeClassifier('haarcascades/Haarcascades/haarcascade_fullbody.xml')

    # Get the height and width of the frame (required to be an interfer)
    w = int(cap_video.get(3))
    h = int(cap_video.get(4))

    out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

    # Loop once video is successfully loaded
    while (True):

        ret, frame = cap_video.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Pass frame to our body classifier
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

            # Extract bounding boxes for any bodies identified
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Write the frame into the file 'output.avi'
            out.write(frame)
        else:
            break

    cap_video.release()
    out.release()


def video_recon_car_multi_frame():
    cap_video = cv2.VideoCapture('videos/cars.mp4')
    car_classifier = cv2.CascadeClassifier('haarcascades/Haarcascades/haarcascade_car.xml')

    # Get the height and width of the frame (required to be an interfer)
    w = int(cap_video.get(3))
    h = int(cap_video.get(4))

    out = cv2.VideoWriter('car_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

    # Loop once video is successfully loaded
    while (True):

        ret, frame = cap_video.read()
        if ret:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Pass frame to our body classifier
            cars = car_classifier.detectMultiScale(gray, 1.2, 3)

            # Extract bounding boxes for any bodies identified
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Write the frame into the file 'output.avi'
            out.write(frame)
        else:
            break

    cap_video.release()
    out.release()

def show_video(path_video):


    read_video = cv2.VideoCapture(path_video)
    while (read_video.isOpened()):
        ret, frame = read_video.read()
        cv2.imshow('MultiDetection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    read_video.release()





if __name__ == '__main__':
    #download_data()
    #video_recon_body_multi_frame()
    show_video('car_output.avi')
    #video_recon_car_multi_frame()


