from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import *
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet101,ResNet50
import tensorflow as tf
import tensorflow.keras.layers as layers
import keras
def convolution_block(block_input,num_filters=256,kernel_size=3,dilation_rate=1,use_bias=False,):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer= "he_normal",
    )(block_input)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

class DeepLabV3Plus101():
    def __init__(self,image_size,learning_rate):
        self.img_size = image_size
        self.img_shape=(image_size, image_size, 3)
        self.learning_rate = learning_rate
    def build_model(self, filters):
        resnet50 = ResNet101(include_top=False, weights='imagenet', input_tensor=None, input_shape=self.img_shape, pooling=None)
        x = resnet50.get_layer('conv4_block23_2_relu').output
        input_a = layers.UpSampling2D(
        size=(self.img_size // 4 // x.shape[1], self.img_size // 4 // x.shape[2]),
        interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(self.img_size // x.shape[1], self.img_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(2, kernel_size=(1, 1), padding="same")(x)
        return keras.models.Model(inputs=resnet50.input, outputs=model_output)
    
class DeepLabV3Plus50():
    def __init__(self,image_size,learning_rate):
        self.img_size = image_size
        self.img_shape=(image_size, image_size, 3)
        self.learning_rate = learning_rate
    def build_model(self, filters):
        resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=self.img_shape, pooling=None)
        x = resnet50.get_layer('conv4_block6_2_relu').output
        input_a = layers.UpSampling2D(
        size=(self.img_size // 4 // x.shape[1], self.img_size // 4 // x.shape[2]),
        interpolation="bilinear",
        )(x)
        input_b = resnet50.get_layer("conv2_block3_2_relu").output
        input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

        x = layers.Concatenate(axis=-1)([input_a, input_b])
        x = convolution_block(x)
        x = convolution_block(x)
        x = layers.UpSampling2D(
            size=(self.img_size // x.shape[1], self.img_size // x.shape[2]),
            interpolation="bilinear",
        )(x)
        model_output = layers.Conv2D(2, kernel_size=(1, 1), padding="same")(x)
        return keras.models.Model(inputs=resnet50.input, outputs=model_output)