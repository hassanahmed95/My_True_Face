from mtcnn.mtcnn import MTCNN
import imutils
import cv2
import numpy as np
from keras.models import model_from_json

model_architecture= "Expressions_models/model.json"
model_weights = "Expressions_models/_mini_XCEPTION.57-0.75.hdf5"


def emotion_predictor(model_json_file=model_architecture, model_weights_file=model_weights):
    # the reconstruction of the model from the json file
    with open(model_json_file, 'r') as f:
        ml_model = model_from_json(f.read())
    ml_model.load_weights(model_weights_file)
    return ml_model


def draw_image_with_boxes():
    image, result_list = get_faces()
    print(len(result_list))
    model = emotion_predictor()
    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]

    for result in result_list:

        x, y, width, height = result['box']
        cropped_face = image[y: y+height,  x: x+width]
        cv2.rectangle(image, (x,y), (x+width,  y+ height),(0,255,0),2)
        cropped_face = cv2.resize(cropped_face, (90, 90))
        cropped = cv2.resize(cropped_face, (48, 48))
        # cropped_face = image.img_to_array(cropped_face)
        # cv2.imshow("B",cropped_face)
        # # cv2.waitKey()
        # cv2.imshow("",cropped)
        # cv2.waitKey()
        # exit()

        dst = cv2.fastNlMeansDenoisingColored(cropped_face, None, 4, 4, 2, 5)
        # cv2.imshow("face", dst)

        dst = np.expand_dims(dst, axis=0)
        # cv2.rectangle(image, (x, y), (x + width + 20, y + height + 20), (0, 255, 0), 2)

        prediction = emotion_list[(int(np.argmax(model.predict(dst))))]
        # print(prediction)

        if prediction == "NEUTRAL":
            prediction = "Ne"
        elif prediction == "HAPPY":
            prediction = "Ha"
        elif prediction == "BORE":
            prediction = "Bo"
        else:
            prediction = "Su"
        # print(prediction)

        # exit()cropped_face = cv2.resize(cropped_face, (90, 90))
        cv2.putText(image, prediction, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 1, cv2.LINE_AA, False)

        # cv2.imshow(" ", image)
        # # cv2.waitKey()

    cv2.imshow("detections", image)
    cv2.waitKey()


def get_faces():
    filename = "faces.jpg"
    image = cv2.imread(filename)
    # image= imutils.resize(image,width=800)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    detector = MTCNN(min_face_size=9,scale_factor=0.909,steps_threshold=([0.6,0.8,0.92]))
    faces = detector.detect_faces(image)
    print(type(faces))
    return image, faces


if __name__ == "__main__":
    draw_image_with_boxes()
