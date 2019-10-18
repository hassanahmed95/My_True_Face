from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from keras.layers import Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import os
# he_uniform is the method for weights initalization, if we use relu as our activation function
np.random.seed(1)


def dl_model():

    nb_classes = 4
    # Initialising the CNN
    model = Sequential()

    # 1st Convolution
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # it is quite necessary to flat the inputs of conv layer before feeding to to dense layer
    # it make it uni dimension . . .
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes, activation ='softmax'))

    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_fitting():
    model = dl_model()

    batch_size = 128
    BASE_PATH = os.path.dirname(__file__)

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_it = datagen.flow_from_directory("{base_path}/Data_Testing/Training".format(base_path=BASE_PATH),
        class_mode='categorical', batch_size=batch_size, target_size=(90, 90))
    test_it = datagen.flow_from_directory(  "{base_path}/Data_Testing/Testing".format(base_path=BASE_PATH),
        class_mode='categorical', batch_size=16, target_size=(90, 90))

    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=50, verbose=1,
                                  callbacks=callbacks_list)


