
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
import sys
from keras.preprocessing.image import ImageDataGenerator
# from test_file import hello
# he_uniform is the method for weights initalization, if we use relu as our activation function

def dl_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(250, 250, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    # model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Flatten is the function which is used to convert the feature map to
    # a single column that is passed to a fully connected layer. . .
    # the connector between the con layer and dense layer is flatten layer . . .

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = 'softmax'))

    # compile model
    opt = Adam(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title("Cross Entropy loss . . .")
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='blue', label='test')

    # plot accuracy
    pyplot.subplot(212)
    pyplot.title("Classification accuracy. . .")
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='test')

    # save plot to file
    # filename = sys.argv[0].split('/')[-1]
    filename = "Matplotlib_graph"
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def model_fitting():
    model = dl_model()
    datagen = ImageDataGenerator(rescale= 1.0/ 255.0)

    train_it = datagen.flow_from_directory('/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Data_Testing/Training/',
                                           class_mode='categorical', batch_size=8, target_size=(250, 250))
    test_it = datagen.flow_from_directory('/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Data_Testing/Testing/',
                                          class_mode='categorical', batch_size=8, target_size=(250, 250))

    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs= 50, verbose=1)

    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)




model_fitting()