from mtcnn.mtcnn import MTCNN
import imutils
import cv2


def draw_image_with_boxes():
    image, result_list = get_faces()
    print(len(result_list))

    for result in result_list:

        x, y, width, height = result['box']
        cropped_face = image[y: y+height,  x: x+width]
        cv2.rectangle(image, (x,y), (x+width,  y+ height),(0,255,0),2)
        # cropped_face = cv2.resize(cropped_face, (90, 90))
        dst = cv2.fastNlMeansDenoisingColored(cropped_face, None, 4, 4, 2, 5)
        # cv2.imshow("face", dst)
        # cv2.waitKey()

    cv2.imshow("detections", image)
    cv2.waitKey()

def get_faces():
    filename = 'NAMAL.png'
    image = cv2.imread(filename)
    # image= imutils.resize(image,width=800)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    detector = MTCNN(min_face_size=10,scale_factor=0.909,steps_threshold=([0.6,0.8,0.92])   )
    faces = detector.detect_faces(image)
    print(type(faces))
    return image, faces


if __name__ == "__main__":
    draw_image_with_boxes()
