from mtcnn.mtcnn import MTCNN
import imutils
import cv2

video_capture = cv2.VideoCapture(0)
detector = MTCNN()
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # frame = cv2.resize(frame,(500,500))
    frame = imutils.resize(frame, width=600)
    print(frame.shape)
    # here I will get coordinates of all faces detected in the frame
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, width, height = face['box']
        cropped_face = frame[y: y+height,  x: x+width]
        cv2.rectangle(frame, (x,y), (x+width,  y+ height),(0,255,0),2)
        # cropped_face=cv2.resize(cropped_face, (90, 90))
        # dst = cv2.fastNlMeansDenoisingColored(cropped_face, None, 4, 4, 2, 5)
        # cv2.imshow("face", dst)
        # cv2.waitKey()

    cv2.imshow("detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()