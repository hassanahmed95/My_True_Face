#in that file data shuffling , train and test data merging has been implemented
#cos, the data generator is expecting to have train and test 2 directories
import numpy as np
from os import makedirs
from random import seed
import os
from shutil import copyfile,copy2
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
    seed(1)
    val_ratio = 0.25
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        os.chdir(directory)

        for file_name in os.listdir("."):
            src_file = os.getcwd() + "/" + file_name
            # print(src_file)
            dst_dir = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Expressions_Dataset_Home/train/' + directory +'/'+ file_name
            # print(random())

            if random() < val_ratio:
                dst_dir = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Expressions_Dataset_Home/test/' + directory +'/'+ file_name

            copy2(src_file, dst_dir)
            # exit()

        os.chdir("..")
        os.getcwd()


if __name__ == '__main__':
    # makedir()
    data_copying()
    print("DONE")





#
#
#
# def get_data(data_path = Data_Source, class_labels=("Bore_faces","Happy_faces","Neutral_faces", "Surprize_faces")):
#
#     data   =  []
#     labels =  []
#     names  =  []
#
#     os.chdir(data_path)
#     for i, directory in enumerate(class_labels):
#         os.chdir(directory)
#         print(directory)
#
#         for filename in os.listdir('.'):
#             print(filename)
#             filepath = os.getcwd() + '/' + filename
#
#             # feature_vector = get_feature_vector_from_mfcc(file_path=filepath,mfcc_len=mfcc_len)
#             feature_vector = "data, extracted from the image fodlers. . . "
#
#             data.append(feature_vector)
#             labels.append(i)
#             names.append(filename)
#         os.chdir("..")
#
#     return np.array(data), np.array(labels)