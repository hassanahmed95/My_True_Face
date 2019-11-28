from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
from mtcnn.mtcnn import MTCNN


def eye_aspect_ratio(eye):
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    C = dist.euclidean(eye[0], eye[3])

    # the denominator has been multiplied with a specific factor to equalize the weight..
    ear = (A + B) / (2.0 * C)
    return ear


def drowsiness():
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 50
    COUNTER = 0
    detector = MTCNN()
    predictor = dlib.shape_predictor("Face_detection_model/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=250)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detect_faces(frame)
        for rect in rects:
            x, y, width, height= rect['box']
            rect = dlib.rectangle(x, y, x+width, y+height)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            shape = predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    drowsiness()