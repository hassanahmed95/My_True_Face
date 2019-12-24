from mtcnn.mtcnn import MTCNN
import cv2
from keras.models import model_from_json
import numpy as np
from collections import Counter
model_architecture = "Expressions_models/model.json"
model_weights = "Expressions_models/_mini_XCEPTION.57-0.75.hdf5"


def emotion_predictor(model_json_file=model_architecture, model_weights_file=model_weights):
    # the reconstruction of the model from the json file
    with open(model_json_file, 'r') as f:
        ml_model = model_from_json(f.read())
    ml_model.load_weights(model_weights_file)
    return ml_model


test = None


def video_processing():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC, 2)
    # detector = MTCNN(min_face_size=10,scale_factor=0.909,steps_threshold=([0.6,0.8,0.92])   )
    detector = MTCNN(min_face_size=10, steps_threshold=([0.6, 0.8, 0.92]))
    # detector = MTCNN()
    model = emotion_predictor()
    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]

    track_lsit = []
    count =0

    # test = None

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = detector.detect_faces(frame)
            try:
                for face in faces:
                    face_confidence = face['confidence']
                    # print(face_confidence)
                    x, y, width, height = face['box']
                    cropped_face = frame[y - 20: y + height + 20, x: x + width + 10]
                    cropped_face = cv2.resize(cropped_face, (90, 90))
                    dst = np.expand_dims(cropped_face, axis=0)
                    cv2.rectangle(frame, (x, y), (x + width+20, y + height+20), (0, 255, 0), 2)

                    prediction = emotion_list[(int(np.argmax(model.predict(dst))))]

                    track_lsit.append(prediction)
                    print(len(track_lsit))

                    if len(track_lsit) > 10:
                        rep_prediction = Counter(track_lsit)
                        rep_prediction = rep_prediction.most_common(1)[0][0]
                        track_lsit.clear()
                        track_lsit.append(rep_prediction)

                    print(track_lsit[0])
                    cv2.putText(frame, track_lsit[0], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA, False)
                    # cv2.imshow("detections", frame)

            except:
                continue
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_processing()
