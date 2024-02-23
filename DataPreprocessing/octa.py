import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.applications import ResNet50
class Model():
    def __init__(self, image_size, learning_rate):
        self.input_size = (image_size, image_size, 3)
        self.learning_rate = learning_rate

    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def jaccard_index(self, y_true,y_pred, smooth=100):
        """ Calculates mean of Jaccard distance as a loss function """
        intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
        sum_ = tf.reduce_sum(y_true + y_pred, axis=(1,2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return jac

    def dice_coef_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

class up_conv(tf.keras.Model):
    def __init__(self, ch_in, ch_out):
        super(up_conv,self).__init__()
        self.up = tf.keras.Sequential(
            keras.layers.Conv2DTranspose(ch_out, kernel_size=2, strides=2),
        )
    
    def call(self, x):
        x = self.up(x)
        return x
    
class res_conv_block(tf.keras.Model):
    def __init__(self, ch_in, ch_out):
        super(res_conv_block, self).__init__()
        self.conv = tf.keras.Sequential(
            keras.layers.Conv2D(ch_out, kernel_size=3, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.SplAtConv2d(ch_out, ch_out, kernel_size=3,padding='same',groups=2,radix=2,norm_layer=BatchNormalization),
            keras.layers.ReLU(),
        )
        self.downsample = tf.keras.Sequential(
            keras.layers.Conv2D(ch_out, kernel_size=1, strides=1, use_bias=False),
            keras.layers.BatchNormalization(),
        )
        self.relu = keras.layers.ReLU()
    
    def call(self, x):
        residual = self.downsample(x)
        out = self.conv(x)
        return self.relu(out + residual)

class SRF_UNet(Model):
    def __init__(self, input_size, learning_rate):
        self.input_size = (input_size, input_size, 3)
        self.learning_rate = learning_rate



    def build_model(self, filters, img_ch=3, output_ch=1):
        inputs = keras.layers.Input(self.input_size)

        self.resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(None, None, img_ch))

        firstconv = self.resnet.get_layer('conv1')
        firstbn = self.resnet.get_layer('bn_conv1')
        firstrelu = self.resnet.get_layer('activation')
        firstmaxpool = self.resnet.get_layer('max_pooling2d')
        encoder1 = self.resnet.get_layer('conv2_')

        for layer in self.resnet.layers:
            print(layer.name, layer.output_shape)

        for i in self.resnet.layers:
            print(i.name, i.output_shape)

        print("1=========",self.resnet.layers[1].name, self.resnet.layers[1].output_shape)
        print("2=========",self.resnet.layers[2].name, self.resnet.layers[2].output_shape)
        print("3=========",self.resnet.layers[3].name, self.resnet.layers[3].output_shape)
        print("4=========",self.resnet.layers[4].name, self.resnet.layers[4].output_shape)




        x0 = firstconv(inputs)
        x0 = firstbn(x0)
        x0 = firstrelu(x0)
        x1 = firstmaxpool(x0)

        print(x1.shape, x0.shape)

        x2 = encoder1(x1)
        x3 = encoder2(x2)
        x4 = encoder3(x3)
        x5 = encoder4(x4)

        print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape, x0.shape)

        # Decoder

        down_pad = False
        right_pad = False
        if x4.shape[1] % 2 == 1:
            x4 = tf.pad(x4, [[0, 0], [0, 1], [0, 0], [0, 0]])
            down_pad = True
        if x4.shape[2] % 2 == 1:
            x4 = tf.pad(x4, [[0, 0], [0, 0], [0, 1], [0, 0]])

            right_pad = True

        d5_thick = up_conv(ch_in=2048, ch_out=1024)(x5)
        d5_thick = tf.concat([x4, d5_thick], axis=3)

        # Decoder
        if down_pad and (not right_pad):
            d5_thick = d5_thick[:, :, :-1, :]
        if (not down_pad) and right_pad:
            d5_thick = d5_thick[:, :, :, :-1]
        if down_pad and right_pad:
            d5_thick = d5_thick[:, :, :-1, :-1]

        d5_thick = res_conv_block(ch_in=2048, ch_out=1024)(d5_thick)

        d4_thick = up_conv(ch_in=1024, ch_out=512)(d5_thick)
        d4_thick = tf.concat([x3, d4_thick], axis=3)
        d4_thick = res_conv_block(ch_in=1024, ch_out=512)(d4_thick)

        d3_thick = up_conv(ch_in=512, ch_out=256)(d4_thick)
        d3_thick = tf.concat([x2, d3_thick], axis=3)
        d3_thick = res_conv_block(ch_in=512, ch_out=256)(d3_thick)

        d2_thick = up_conv(ch_in=256, ch_out=64)(d3_thick)
        d2_thick = tf.concat([x0, d2_thick], axis=3)
        d2_thick = res_conv_block(ch_in=128, ch_out=64)(d2_thick)

        d1_thick =  up_conv(ch_in=64, ch_out=64)(d2_thick)
        # d1_thick = tf.concat([x, d1_thick], axis=3)
        d1_thick = res_conv_block(ch_in=64, ch_out=32)(d1_thick)
        
        d1_thick = tf.keras.layers.Conv2D(output_ch, kernel_size=1)(d1_thick)
        out_thick = tf.keras.activations.sigmoid(d1_thick)

        """
        d5_thin = up_conv(ch_in=2048, ch_out=1024)(x5)
        d5_thin = tf.concat([x4, d5_thin], axis=3)
        
        d5_thin = res_conv_block(ch_in=2048, ch_out=1024)(d5_thin)
        
        d4_thin = up_conv(ch_in=1024, ch_out=512)(d5_thin)
        d4_thin = tf.concat([x3, d4_thin], axis=3)
        d4_thin = res_conv_block(ch_in=1024, ch_out=512)(d4_thin)

        
        d3_thin = up_conv(ch_in=512, ch_out=256)(d4_thin)  # x3
        d3_thin = tf.concat([x2, d3_thin], axis=3)
        d3_thin = res_conv_block(ch_in=512, ch_out=256)(d3_thin)
        """
        
        d2_thin = up_conv(ch_in=256, ch_out=64)(x2)  # d3_thin
        d2_thin = tf.concat([x0, d2_thin], axis=3)
        d2_thin = res_conv_block(ch_in=128, ch_out=64)(d2_thin)
        
        d1_thin = up_conv(ch_in=64, ch_out=64)(d2_thin)
        # d1_thin = tf.concat([x, d1_thin], axis=3)
        d1_thin = res_conv_block(ch_in=64, ch_out=32)(d1_thin)

        d0_thin = self.Conv_1x1_thin(d1_thin)
        d0_thin = self.sigmoid(d0_thin)

        d0_thick = self.Conv_1x1_thick(d1_thick)
        d0_thick = self.sigmoid(d0_thick)



        d0 = tf.concat([d0_thick, d0_thin], axis=3)
        d0 = self.Up_conv1(d0)
        d0 = res_conv_block(ch_in=64, ch_out=32)(d0)
        d0 = self.Conv_1x1(d0)

        d0 = self.sigmoid(d0)

        outputs = d0
        model = keras.models.Model(inputs=inputs, outputs=d0)   
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss,  metrics=[self.dice_coef, self.jaccard_index])
        return model


