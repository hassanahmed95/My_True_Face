from mtcnn.mtcnn import MTCNN
import imutils
import cv2
from keras.models import model_from_json
import numpy as np


model_architecture= "models/model.json"
model_weights = "models/_mini_XCEPTION.57-0.75.hdf5"


def emotion_predictor(model_json_file = model_architecture , model_weights_file=model_weights):
    # the reconstruction of the model from the json file
    with open(model_json_file, 'r') as f:
        ml_model = model_from_json(f.read())
    ml_model.load_weights(model_weights_file)
    return ml_model


def video_processing():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2)
    detector = MTCNN()
    model = emotion_predictor()
    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]
    while True:
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=500)
            faces = detector.detect_faces(frame)
            for face in faces:
                x, y, width, height = face['box']

                cropped_face = frame[y - 20: y + height + 20, x: x + width + 10]
                cropped_face = cv2.resize(cropped_face, (90, 90))
                dst = np.expand_dims(cropped_face, axis=0)
                cv2.rectangle(frame, (x, y), (x + width+20, y + height+20), (0, 255, 0), 2)

                prediction = emotion_list[(int(np.argmax(model.predict(dst))))]
                print(prediction)
            cv2.imshow("detections", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_processing()
