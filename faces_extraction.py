import os
from mtcnn.mtcnn import MTCNN
import cv2
from matplotlib import pyplot

Data_Source = "Expressions_Dataset"

# images_paths = []


def draw_image_with_boxes(filename, result_list):
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    count = 1
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        cropped_face = data[y: y + height, x: x + width]
        return cropped_face
        # cropped_face = cv2.resize(cropped_face, (224, 224))
        # # cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
        # # cv2.imwrite("cropped_face/" + str(count) + ".jpg", cropped_face)


def get_data(data_path = Data_Source, class_labels=("Bore_faces","Happy_faces","Neutral_faces", "Surprise_faces")):
    count = 0
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        os.chdir(directory)
        count = 0
        print(directory)
        for filename in os.listdir('.'):
            count += 1
            image_path = os.getcwd() + "/"+filename
            image = pyplot.imread(image_path)
            print(image_path)
            detector = MTCNN()
            faces = detector.detect_faces(image)
            cropped_face = draw_image_with_boxes(filename, faces)
            if cropped_face is None:
                continue
            if cropped_face.size == 0:
                # print("I have been stucked in the loop . . .")
                continue
            face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            cropped_face_path = "/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Cropped_faces/" +directory+"/"
            cv2.imwrite(cropped_face_path + directory.split("_")[0] + str(count) + ".jpg",face)
            #  cv2.imshow("my_face", cropped_face)
            print("DONE . .")
        os.chdir("..")
        # exit()
    print("Total number of the images are" + str(count))


if __name__ == '__main__':
    # print("Here I am printing the all of the images extracted from the local directories. .  .")

    print("hello words.  . .. .")
    get_data()
