import cv2
import time
import numpy as np
frozen_graph = "./Face_detection_model/240x180_depth075_ssd_mobilenetv1/frozen_inference_graph.pb"
text_graph = "./Face_detection_model/240x180_depth075_ssd_mobilenetv1/graph.pbtxt"
cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)


def drawBoundingBox(img, bbox):
    x, y, w, h = [int(c) for c in bbox]

    m = 0.2

    # Upper left corner
    pt1 = (x, y)
    pt2 = (int(x + m*w), y)
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    pt1 = (x, y)
    pt2 = (x, int(y + m*h))
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    # Upper right corner
    pt1 = (x + w, y)
    pt2 = (x + w, int(y + m*h))
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    pt1 = (x + w, y)
    pt2 = (int(x + w - m * w), y)
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    # Lower left corner
    pt1 = (x, y + h)
    pt2 = (x, int(y + h - m*h))
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    pt1 = (x, y + h)
    pt2 = (int(x + m * w), y + h)
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    # Lower right corner
    pt1 = (x + w, y + h)
    pt2 = (x + w, int(y + h - m*h))
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

    pt1 = (x + w, y + h)
    pt2 = (int(x + w - m * w), y + h)
    cv2.line(img, pt1, pt2, color=[255, 0, 0], thickness=2)

count = 0
video = cv2.VideoCapture(0)
while True:
    stat, img = video.read()
    frame = img.copy()
    if stat:
        rows, cols = img.shape[0:2]
        cvNet.setInput(cv2.dnn.blobFromImage(
            img, size=(240, 180), swapRB=True, crop=False))
        cvOut = cvNet.forward()
        for detection in cvOut[0, 0, :, :]:
            score = float(detection[2])

            left = int(detection[3] * cols)
            top = int(detection[4] * rows)
            right = int(detection[5] * cols)
            bottom = int(detection[6] * rows)
            width = right - left
            height = bottom - top
            if score > 0.2:
                # cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
                # cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
                drawBoundingBox(frame, [left, top, width, height])

                cropped_face = img[top:top + height + 5, left: left + width + 5]
                cropped_face = cv2.resize(cropped_face, (90, 90))
                cv2.imwrite("./faces/"+ str(count) + "img.jpg", cropped_face)
                count+=1

                # cv2.imshow("cropped_face", cropped_face)
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

