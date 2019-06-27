# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import cv2
# draw an image with detected objects


def draw_image_with_boxes(filename ,result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    count = 1

    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']

        cropped_face = data[y: y+height ,  x: x+width]
        cropped_face = cv2.resize(cropped_face, (224,224))
        # cropped_face =cv2.cvtColor(cropped_face,cv2.COLOR_RGB2GRAY)
        #
        # cv2.imshow("dfs",cropped_face)
        # cv2.waitKey()
        # exit()
        # cv2.imwrite("cropped_face/" + str(count) + ".jpg", cropped_face)
        # count += 1

        # pyplot.imshow(cropped_face)
        # pyplot.show()
        # exit()

        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot

    pyplot.show()


filename = '/home/hassan/Hassaan_Home/My_Python_Projects/My_true_face_update/Expressions_Dataset/Bore_faces/image0000490.jpg'
# load image from file
pixels = pyplot.imread(filename)
# create the detector, using default weights
detector = MTCNN()
# detect faces in the image
faces = detector.detect_faces(pixels)
# display faces on the original image
draw_image_with_boxes(filename, faces)