from mtcnn.mtcnn import MTCNN
import cv2
import imutils
filename = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Expressions_Dataset(original)/Happy_faces/image0000017.jpg'
data = cv2.imread(filename)
data = cv2.resize(data,(48,48))
print(data.shape)
# data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
print(data.shape)
detector = MTCNN()
faces = detector.detect_faces(data)

for result in faces:
    x,y,width,height = result['box']
    data = cv2.rectangle(data, (x, y), (x + width, y + height), (255, 0, 0), 1)
    cropped_face = data[y: y + height, x: x + width]
    # cropped_face = imutils.resize(cropped_face,width=)
    # cropped_face =  cv2.resize(cropped_face,(150,150))
    cv2.imshow("Cropped", cropped_face)
    cv2.waitKey()
cv2.imshow("Data", data)
cv2.waitKey()
cv2.destroyAllWindows()
# exit()
