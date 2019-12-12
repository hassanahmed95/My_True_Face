from keras.models import model_from_json
import pdb
from keras.preprocessing import image
from keras.models import load_model
import os
import numpy as np
import cv2


def model_testing(model_json_file, model_weights_file):
    # the reconstruction of the model from the json file
    with open(model_json_file, 'r') as f:
        ml_model = model_from_json(f.read())

    ml_model.load_weights(model_weights_file)
    return ml_model


if __name__ == '__main__':
    base_path = os.path.dirname(__file__)

    json_file = "{base_path}/Expressions_models/model.json".format(base_path=base_path)
    weights_file = "{base_path}/Expressions_models/_mini_XCEPTION.57-0.75.hdf5".format(base_path= base_path)

    emotion_list = ["BORE", "HAPPY", "NEUTRAL", "SURPRISE"]
    model = model_testing(json_file, weights_file)
    # pdb.set_trace()

    test_image = image.load_img('img.jpg',target_size=(90,90))
    test_image = image.img_to_array(test_image)

    test_image = np.expand_dims(test_image, axis=0)

    prediction = emotion_list[(int(np.argmax(model.predict(test_image))))]
    print(prediction)
#




