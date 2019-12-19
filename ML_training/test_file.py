import cv2
import numpy as np


image = cv2.imread("1.jpg")
cv2.imshow("asli", image)
cv2.waitKey()
print(image.shape)
# image_size=(48,e48)
width, height = 48, 48
# face = np.asarray(image).reshape(width, height)
frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
# face = cv2.resize(image.astype('uint8'),image_size)
print(frame.shape)
cv2.imshow("", frame)
cv2.waitKey()