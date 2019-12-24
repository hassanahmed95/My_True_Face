import cv2
import numpy as np
from  keras.models import load_model
image = cv2 .imread("Bore2.jpg",0)
image = cv2.resize(image, (48, 48))
image = image[..., np.newaxis]

# dst = np.expand_dims(image, axis=1)
print(image.shape)
# exit()
model = load_model("model_v6_23.hdf5")


predicted_class = np.argmax(model.predict(image))


print(predicted_class)
exit()

#
# image = cv2.imread("1.jpg")
# cv2.imshow("asli", image)
# cv2.waitKey()
# print(image.shape)
# # image_size=(48,e48)
# width, height = 48, 48
# # face = np.asarray(image).reshape(width, height)
# frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# # face = cv2.resize(image.astype('uint8'),image_size)
# print(frame.shape)
# cv2.imshow("", frame)
# cv2.waitKey()

# from collections import Counter
#
#
# def most_frequent(List):
#     occurence_count = Counter(List)
#     print(occurence_count)
#     print(type(occurence_count))
#     charater =  occurence_count.most_common(1)[0][0]
#     print(charater)
#
#     exit()
#     # chracter =  occurence_count.most_common(1)[0][0]
#     # repr(chracter)
#
#
# List =['Cat', 'Cat', 'Dog']
# print(most_frequent(List))




