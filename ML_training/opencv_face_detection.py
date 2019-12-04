import numpy as np
import imutils
import cv2

# construct the argument parse and parse the arguments
prototxt = "./Face_detection_model/deploy.prototxt.txt"
caffe_model = "./Face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"


net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
while True:
    ret,frame = vs.read()
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.2:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cropped_face = frame[startY:endY, startX:endX]
        cv2.imshow("sds",cropped_face)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()