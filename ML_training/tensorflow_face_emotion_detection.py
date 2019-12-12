import cv2
import numpy as np
from keras.models import model_from_json
import os

base_path = os.path.dirname(os.path.abspath(__file__))
# models for the face detection using opencv
frozen_graph = "./Face_detection_model/240x180_depth075_ssd_mobilenetv1/frozen_inference_graph.pb"
text_graph = "./Face_detection_model/240x180_depth075_ssd_mobilenetv1/graph.pbtxt"
cvNet = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)


# models for expressions recognition
model_architecture= "Expressions_models/model.json"
model_weights = "Expressions_models/_mini_XCEPTION.57-0.75.hdf5"


# model for gender recogniton
mean_filename = '{base_path}/Gender_Models/mean.binaryproto'.format(base_path=base_path)
gender_net_pretrained = '{base_path}/Gender_Models/gender_net.caffemodel'.format(base_path=base_path)
gender_net_model_file = "{base_path}/Gender_Models/deploy_gender.prototxt".format(base_path=base_path)


def emotion_predictor(model_json_file=model_architecture, model_weights_file=model_weights):
    # the reconstruction of the model from the json file
    with open(model_json_file, 'r') as f:
        ml_model = model_from_json(f.read())
    ml_model.load_weights(model_weights_file)
    return ml_model


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

# def gender_recognition()


def frame_processing():
    count = 0
    video = cv2.VideoCapture(0)
    model = emotion_predictor()
    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]

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
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        frame = drawBoundingBox(frame, [left, top, width, height])
                        cropped_face = img[top + 10:top + 5 + height + 10, left + 5: left + 5 + width + 5]
                        cropped_face = cv2.resize(cropped_face, (90, 90))
                        dst = np.expand_dims(cropped_face, axis=0)
                        prediction = emotion_list[(int(np.argmax(model.predict(dst))))]

                        cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA, False)
                        cv2.imwrite("{base_path}/img.jpg".format(base_path=base_path), cropped_face)
                    except:
                        continue

                    # cv2.imshow("cropped_face", cropped_face)
                    # cv2.waitKey()
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
        else:
            continue


if __name__ == '__main__':
    frame_processing()



