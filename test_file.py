# make a prediction for a new image.
#
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
#
# # img = loadd_image('Updated_Data/train/Surprise_faces/Surprise3.jpg')
# # # load model

import numpy as np
from keras.preprocessing import image
model = load_model('weights.best.hdf5')
test_image = image.load_img('Updated_Data/test/Surprise_faces/Surprise1451.jpg', target_size=(90,90))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = model.predict(test_image)
# training_set.class_indices
print(result)
# for i in result[0]:
#     print(i)

exit()













test_dir = "/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Data_Testing/Testing"

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(90,90),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

# print(test_generator)
filenames = test_generator.filenames
# print(filenames)
nb_samples = len(filenames)

model = load_model('weights.best.hdf5')

predict = model.predict_generator(test_generator,steps = nb_samples)
# print(len(predict))
# print(type(predict))
#
print(predict)

exit()

# load and prepare the image
def loadd_image(filename):
    # load the image
    img = load_img(filename)
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 90, 90, 3)
    # center pixel data
    # img = img.astype('float32')
    # img = img - [123.68, 116.779, 103.939]
    return img


# load an image and predict the class
def run_example():
    # load the image
    img = loadd_image('Updated_Data/test/Bore_faces/Bore111.jpg')
    # load model
    model = load_model('weights.best.hdf5')
    # predict the class
    result = model.predict(img)
    # print(result)
    for i in result[0]:
        print(i)


# entry point, run the example
run_example()