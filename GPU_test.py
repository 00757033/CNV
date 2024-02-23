

# import torch
# print(torch.version.cuda)

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))

# from torch.backends import cudnn
# print(cudnn.is_available())

import tensorflow as tf

print(tf.__version__)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:

    print("Please install GPU version of TF")
    







