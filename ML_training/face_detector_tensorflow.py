import cv2
# import cvlib as cv
import imutils
import time
import numpy as np
import  os
frozen_graph = "./Face_detection_model/240x180_depth075_ssd_mobilenetv1/frozen_inference_graph.pb"
text_graph = "./Face_detection_model/240x180_depth075_ssd_mobilenetv1/graph.pbtxt"
cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)


def drawBoundingBox (img, bbox):
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
    return img


count = 0
# video = cv2.VideoCapture(0)
base_path= os.path.dirname(os.path.abspath(__file__))
img = cv2.imread("NAMAL.png")
# img = cv2.resize(img,(600,700))
img = imutils.resize(img, width=800)

frame = img.copy()

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
    # print(score)
    if score > 0.3:
        # cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
        try:
            cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
            frame = drawBoundingBox(frame, [left, top, width, height])
            cropped_face = img[top+10:top + height+10, left+5: left+5 + width]
            # gender, confidence = cv.detect_gender(cropped_face)
            #
            # gender = gender[int(np.argmax(confidence))]
            # print(gender)

            cropped_face = cv2.resize(cropped_face, (90, 90))
            # cv2.imwrite("{base_path}/img.jpg".format(base_path=base_path),cropped_face)
        except:
            continue

        # cv2.imshow("cropped_face", cropped_face)
        # cv2.waitKey()
print(frame.shape)
cv2.imshow('Frame', frame)
cv2.waitKey()

exit()

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
            # print(score)
            if score > 0.5:
                # cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
                try:
                    cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
                    frame = drawBoundingBox(frame, [left, top, width, height])
                    cropped_face = img[top+10:top+5 + height+10, left+5: left+5 + width+5]
                    gender, confidence = cv.detect_gender(cropped_face)

                    gender = gender[int(np.argmax(confidence))]
                    print(gender)

                    cropped_face = cv2.resize(cropped_face, (90, 90))
                    cv2.imwrite("{base_path}/img.jpg".format(base_path=base_path),cropped_face)
                except:
                    continue

                cv2.imshow("cropped_face", cropped_face)
                # cv2.waitKey()
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
    else:
        continue





