#In that file the data reading in iterative mode from the
#local repoistory has been done. Moreover finalized
#data list and labels list has been made foe the trainig of
#of the deep CNN
#the method for iterating through the all directories to get the images

import os
import numpy as np
Data_Source = "Expressions_Dataset"


def get_data(data_path = Data_Source, class_labels=("Bore_faces","Happy_faces","Neutral_faces", "Surprize_faces")):
    data = []
    labels = []
    names = []

    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        os.chdir(directory)
        print(directory)

        for filename in os.listdir('.'):
            print(filename)
            filepath = os.getcwd() + '/' + filename

            # feature_vector = get_feature_vector_from_mfcc(file_path=filepath,mfcc_len=mfcc_len)
            feature_vector = "data, extracted from the image fodlers. . . "

            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
        os.chdir("..")

    return np.array(data), np.array(labels)