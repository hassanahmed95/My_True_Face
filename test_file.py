from mtcnn.mtcnn import MTCNN
import cv2
from matplotlib import pyplot
import imutils
filename = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Data_Testing/Training/Bore_faces/Bore25.jpg'
data = cv2.imread(filename)
# data =  pyplot.imread(filename)
# data = cv2.resize(data,(90,90))

# data = cv2.cvtColor(data,cv2.COLOR_RGB2GRAY)
print(data.shape)
# exit()
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
