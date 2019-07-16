# import tensorflow as tf
# tf.test.gpu_device_name()


# for the info of current system ...
from tensorflow.python.client import device_lib
data = device_lib.list_local_devices()

# in order to check the detailed information of CPU
# cat /proc/cpuinfo

# in order to cehck the arguments of the
#any function of the tensorflow
#
# import tensorflow as tf
# tf.nn.sigmoid_cross_entropy_with_logits?
