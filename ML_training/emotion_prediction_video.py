from mtcnn.mtcnn import MTCNN
import imutils
from keras.models import model_from_json
from keras.preprocessing import image
import cv2
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
    video_capture = cv2.VideoCapture(0)
    detector = MTCNN()
    model = emotion_predictor()
    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]
    while True:
        ret, frame = video_capture.read()
        # frame = cv2.resize(frame,(500,500))
        frame = imutils.resize(frame, width=700)
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, width, height = face['box']
            cropped_face = frame[y-20: y+height + 20,  x : x+width+10]
            cv2.rectangle(frame, (x,y), (x+width,  y+ height),(0,255,0),2)
            cropped_face=cv2.resize(cropped_face, (90, 90))
            # dst = cv2.fastNlMeansDenoisingColored(cropped_face, None, 4, 4, 2, 5)
            dst = np.expand_dims(cropped_face, axis=0)
            # here I am performing the emotions prediction model
            prediction = emotion_list[(int(np.argmax(model.predict(dst))))]
            print(prediction)
            cv2.putText(frame, prediction, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # I am performing the drowsiness detection filter.  . .


            # cv2.imwrite("test.jpg",dst)
            # cv2.imshow("face", dst)
            # cv2.waitKey()

        cv2.imshow("detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_processing()