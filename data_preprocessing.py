#in that file data shuffling , train and test data merging has been implemented
#cos, the data generator is expecting to have train and test 2 directories
import numpy as np
from os import makedirs
from random import seed
import os
from shutil import copyfile,copy2
import math
# import random
from random import random


Data_Source = "Cropped_faces/"
# the method for making relevant directories . . .


def makedir():
    data_home = "Expressions_Dataset_Home/"
    sub_dirs = ["train/", "test/"]
    for sub_dir in sub_dirs:
        label_dirs = ["Bore_faces","Happy_faces","Neutral_faces", "Surprise_faces"]
        for label_dir in label_dirs:
            new_dir = data_home + sub_dir + label_dir
            makedirs(new_dir, exist_ok=True)


def data_copying(data_path = Data_Source, class_labels=("Bore_faces","Happy_faces","Neutral_faces", "Surprise_faces")):
    print("Data copying has been started. . .")
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        os.chdir(directory)
        data = os.listdir('.')
        count = 0
        for file_name in data:
            src_file = os.getcwd() + "/" + file_name
            # print(file_name)
            # exit()
            if count < math.ceil((len(data)) * 0.25):

                dst_dir = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Expressions_Dataset_Home/test/' + directory +'/'+ file_name
                # print("Pointer in the test directory. . .")

            else:
                dst_dir = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Expressions_Dataset_Home/train/' + directory + '/' + file_name

            copy2(src_file, dst_dir)
            # print(dst_dir)
            count +=1

        # exit()
        os.chdir("..")
        # os.getcwd()


if __name__ == '__main__':
    # makedir()
    data_copying()
    print("DONE")