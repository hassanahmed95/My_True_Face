
import imutils
import cv2
from keras.models import model_from_json
import numpy as np


model_architecture= "Expressions_models/model.json"
model_weights = "Expressions_models/_mini_XCEPTION.57-0.75.hdf5"

prototxt = "./Face_detection_model/deploy.prototxt.txt"
caffe_model = "./Face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)


def emotion_predictor(model_json_file = model_architecture , model_weights_file=model_weights):
    # the reconstruction of the model from the json file
    with open(model_json_file, 'r') as f:
        ml_model = model_from_json(f.read())
    ml_model.load_weights(model_weights_file)
    return ml_model


def video_processing():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2)
    model = emotion_predictor()
    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=700)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        if ret:
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.2:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cropped_face = frame[startY:endY, startX:endX]
                cropped_face = cv2.resize(cropped_face,(90,90))
                # the model pre-processing for emotion detection model
                # cv2.imshow("cropped face",cropped_face)
                dst = np.expand_dims(cropped_face, axis=0)
                # loading the emotions detecting model
                prediction = emotion_list[(int(np.argmax(model.predict(dst))))]
                print(prediction)

                image = cv2.putText(frame, prediction, (00,185), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255) , 2, cv2.LINE_AA, False)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break


if __name__ == "__main__":
    video_processing()













# import cv2
# import numpy as np
#
# img1 = cv2.imread('1.png')
# img2 = cv2.imread('messi.jpg')
#
# # Read about the resize method parameters here: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html?highlight=resize#resize
# img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
# dst = cv2.addWeighted(img1, 0.7, img2_resized, 0.3, 0)
#
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()