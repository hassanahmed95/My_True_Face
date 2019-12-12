import caffe
import cv2
import os

base_path=  os.path.dirname(os.path.abspath(__file__))


mean_filename ='{base_path}/Gender_Models/mean.binaryproto'.format(base_path=base_path)
gender_net_pretrained='{base_path}/Gender_Models/gender_net.caffemodel'.format(base_path=base_path)
gender_net_model_file= "{base_path}/Gender_Models/deploy_gender.prototxt".format(base_path=base_path)

proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
gender_list = ['Male','Female']
input_image  = cv2.imread("download.png")
prediction = gender_net.predict([input_image])
print(prediction)
print('predicted gender:', gender_list[prediction[0].argmax()])

