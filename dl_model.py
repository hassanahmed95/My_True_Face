from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras .layers import Activation
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# he_uniform is the method for weights initalization, if we use relu as our activation function
np.random.seed(1)

# kernel_regularizer = regularizers.l2(0.0001)

def dl_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(90, 90, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add((Activation(activation="relu")))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))
    #
    # model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.2))

    model.add(Dense(4, activation= 'softmax'))
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_fitting():
    model = dl_model()
    datagen = ImageDataGenerator(rescale = 1.0/ 255.0)

    train_it = datagen.flow_from_directory('/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Data_Testing/Training',
                                           class_mode='categorical', batch_size=16, target_size=(90,90 ))
    test_it = datagen.flow_from_directory('/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Data_Testing/Testing',
                                          class_mode='categorical', batch_size=16, target_size=(90,90))

    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacklist = [checkpoint]

    # model.load_weights("My_DL_Models/Accuracy- 0.59500-100epochs.hdf5")
    # print(len(train_it))
    # exit()
    # the steps per epoch are found by dividing total number of images by batch size. .
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                validation_data = test_it, validation_steps=len(test_it), epochs= 50, verbose=1, callbacks = callbacklist)

    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == "__main__":

    model_fitting()
