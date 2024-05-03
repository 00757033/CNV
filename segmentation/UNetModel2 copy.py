import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers

import numpy as np
#所有使用到的U-Net，使用統一的介面
#Model:UNet、UNet++、Attention UNet、BCD UNet、FRUNet

import keras
import keras.backend as K


# Author: An Jiaoyang
# =============================
import tensorflow as tf
from tensorflow.python.keras import backend as K


def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                          true_fn=lambda: inputs,
                          false_fn=drop)
        
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.cast(self.w, tf.float32), tf.cast(self.h, tf.float32)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DropBlock3D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock3D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 5
        _, self.d, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0= (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock3D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        d, w, h = tf.cast(self.d, tf.float32), tf.cast(self.w, tf.float32), tf.cast(self.h, tf.float32)
        self.gamma = ((1. - self.keep_prob) * (d * w * h) / (self.block_size ** 3) /
                      ((d - self.block_size + 1) * (w - self.block_size + 1) * (h - self.block_size + 1)))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.d - self.block_size + 1,
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool3d(mask, [1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask



# class DropBlock2D(tf.keras.layers.Layer):
#     """See: https://arxiv.org/pdf/1810.12890.pdf"""

#     def __init__(self,
#                  block_size,
#                  keep_prob,
#                  sync_channels=False,
#                  data_format=None,
#                  **kwargs):
#         """Initialize the layer.
#         :param block_size: Size for each mask block.
#         :param keep_prob: Probability of keeping the original feature.
#         :param sync_channels: Whether to use the same dropout for all channels.
#         :param data_format: 'channels_first' or 'channels_last' (default).
#         :param kwargs: Arguments for parent class.
#         """
#         super(DropBlock2D, self).__init__(**kwargs)
#         self.block_size = block_size
#         self.keep_prob = keep_prob
#         self.sync_channels = sync_channels
#         self.data_format = self.normalize_data_format(data_format)
#         self.input_spec = tf.keras.layers.InputSpec(ndim=4)
#         self.supports_masking = True

#     def get_config(self):
#         config = {'block_size': self.block_size,
#                   'keep_prob': self.keep_prob,
#                   'sync_channels': self.sync_channels,
#                   'data_format': self.data_format}
#         base_config = super(DropBlock2D, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

#     def normalize_data_format(self,data_format):
#         if data_format == 'channels_last' or data_format is None:
#             return 'NHWC'
#         elif data_format == 'channels_first':
#             return 'NCHW'
#         else:
#             raise ValueError('Invalid data_format:', data_format)



#     def compute_mask(self, inputs, mask=None):
#         return mask

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def _get_gamma(self, height, width):
#         """Get the number of activation units to drop"""
#         height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32) 
#         block_size = tf.constant(self.block_size, dtype=tf.float32)
#         return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
#                (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

#     def _compute_valid_seed_region(self, height, width):
#         positions = tf.concat([
#             tf.expand_dims(tf.tile(tf.expand_dims(tf.range(height), axis=1), [1, width]), axis=-1),
#             tf.expand_dims(tf.tile(tf.expand_dims(tf.range(width), axis=0), [height, 1]), axis=-1),
#         ], axis=-1)
#         half_block_size = self.block_size // 2
#         valid_seed_region = tf.where(
#             tf.reduce_all([
#                 positions[:, :, 0] >= half_block_size,
#                 positions[:, :, 1] >= half_block_size,
#                 positions[:, :, 0] < height - half_block_size,
#                 positions[:, :, 1] < width - half_block_size,
#             ], axis=0),
#             tf.ones((height, width)),
#             tf.zeros((height, width)),
#         )
#         return tf.expand_dims(tf.expand_dims(valid_seed_region, axis=0), axis=-1)


#     def _compute_drop_mask(self, shape):
#         height, width = shape[1], shape[2]
#         mask = tf.random.stateless_binomial(shape=shape, probs=self._get_gamma(height, width), seed=(0, 0))

#         mask *= self._compute_valid_seed_region(height, width)
#         mask = tf.nn.max_pool(mask, ksize=(1, self.block_size, self.block_size, 1), strides=(1, 1, 1, 1), padding='SAME')
#         return 1.0 - mask


#     def call(self, inputs, training=None):

#         def dropped_inputs():
#             outputs = inputs
#             if self.data_format == 'channels_first':
#                 outputs = tf.transpose(outputs, [0, 2, 3, 1])
#             shape = tf.shape(outputs)
#             if self.sync_channels:
#                 mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
#             else:
#                 mask = self._compute_drop_mask(tf.shape(outputs))
#             outputs = outputs * mask *\
#                 (tf.cast(tf.reduce_prod(shape), dtype=tf.float32) / tf.reduce_sum(mask))
#             if self.data_format == 'channels_first':
#                 outputs = tf.transpose(outputs, [0, 3, 1, 2])
#             return outputs

#         return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=training)



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

    def upsampling_block(self, inputs, filters,name = 'upsampling_block'):
        upsampling = keras.layers.UpSampling2D((2, 2))(inputs)
        conv = keras.layers.Conv2D(filters, (2, 2),  padding='same', kernel_initializer='he_normal')(upsampling)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.Activation('relu')(bn)
        # layer = keras.layers.Lambda(lambda x: x, name=name)(bn)
        return relu
    
    def recurrent_block(self,inputs, filters,time_steps = 2):
        x = inputs
        for i in range(time_steps):
          if i == 0:
            x1 = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
            x1 = keras.layers.BatchNormalization()(x)
            x1 = keras.layers.Activation('relu')(x)
          x1 = keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x + x1)
          x1 = keras.layers.BatchNormalization()(x1)
          x1 = keras.layers.Activation('relu')(x1)
        return x1
      
    def attention_block(self,F_g,F_l,F_int,bn=False):
      g=layers.Conv2D(F_int,kernel_size=(1,1),strides=(1,1),padding='valid')(F_g)
      if bn:
        g=layers.BatchNormalization()(g)
      x=layers.Conv2D(F_int,kernel_size=(1,1),strides=(1,1),padding='valid')(F_l)
      if bn:
        x=layers.BatchNormalization()(x)

      psi=layers.Add()([g,x])
      psi=layers.Activation('relu')(psi)
      
      psi=layers.Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='valid')(psi)
      
      if bn:
        psi=layers.BatchNormalization()(psi)
      psi=layers.Activation('sigmoid')(psi)
      
      return layers.Multiply()([F_l,psi])
      
    
    def standard_unit(self, inputs, filters, name = 'standard_unit'):
        conv1 = keras.layers.Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
        bn1 = keras.layers.BatchNormalization()(conv1)
        relu1 = keras.layers.Activation('relu')(bn1)
        conv2 = keras.layers.Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(relu1)
        
        bn2 = keras.layers.BatchNormalization()(conv2)
        relu2 = keras.layers.Activation('relu')(bn2)
        # 區域名稱
        # layer = keras.layers.Lambda(lambda x: x, name='standard_unit')(bn2)
        return relu2

    def residual_block(self,inputs,filters):
        conv = self.standard_unit(inputs, filters)
        addition = keras.layers.add([conv, inputs])
        return addition
      
      
    
    def build_model(self, filters):
        inputs = keras.layers.Input(self.input_size)

        # downsampling
        conv1 = self.standard_unit(inputs, filters[0])
        pooling1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.standard_unit(pooling1, filters[1])
        pooling2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.standard_unit(pooling2, filters[2])
        pooling3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.standard_unit(pooling3, filters[3])
        pooling4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        conv5 = self.standard_unit(pooling4, filters[4])

        # upsampling
        up1 = self.upsampling_block(conv5, filters[3])
        concat1 = keras.layers.Concatenate()([conv4, up1])
        conv6 = self.standard_unit(concat1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        concat2 = keras.layers.Concatenate()([conv3, up2])
        conv7 = self.standard_unit(concat2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        concat3 = keras.layers.Concatenate()([conv2, up3])
        conv8 = self.standard_unit(concat3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        concat4 = keras.layers.Concatenate()([conv1, up4])
        conv9 = self.standard_unit(concat4, filters[0])

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

        model = keras.models.Model(inputs, outputs)
        return model

class UNetPlusPlus(Model):

    def build_model(self, filters):
        inputs = keras.layers.Input(self.input_size)
        # encode
        conv1_1 = self.standard_unit(inputs, filters[0])
        pool1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)
        conv2_1 = self.standard_unit(pool1, filters[1])
        pool2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2_1)
        conv3_1 = self.standard_unit(pool2, filters[2])
        pool3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3_1)
        conv4_1 = self.standard_unit(pool3, filters[3])
        pool4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4_1)
        conv5_1 = self.standard_unit(pool4, filters[4])

        # skip connection
        up1_2 = self.upsampling_block(conv2_1, filters[1])
        concat1 = keras.layers.Concatenate()([up1_2, conv1_1])
        conv1_2 = self.standard_unit(concat1, filters[0])
        up2_2 = self.upsampling_block(conv3_1, filters[2])
        concat2 = keras.layers.Concatenate()([up2_2, conv2_1])
        conv2_2 = self.standard_unit(concat2, filters[1])
        up1_3 = self.upsampling_block(conv2_2, filters[1])
        concat3 = keras.layers.Concatenate()([up1_3, conv1_1, conv1_2])
        conv1_3 = self.standard_unit(concat3, filters[0])
        up3_2 = self.upsampling_block(conv4_1, filters[3])
        concat4 = keras.layers.Concatenate()([up3_2, conv3_1])
        conv3_2 = self.standard_unit(concat4, filters[2])
        up2_3 = self.upsampling_block(conv3_2, filters[2])
        concat5 = keras.layers.Concatenate()([up2_3, conv2_1, conv2_2])
        conv2_3 = self.standard_unit(concat5, filters[1])
        up1_4 = self.upsampling_block(conv2_3, filters[1])
        concat6 = keras.layers.Concatenate()([up1_4, conv1_1, conv1_2, conv1_3])
        conv1_4 = self.standard_unit(concat6, filters[0])

        # decode
        up4_2 = self.upsampling_block(conv5_1, filters[4])
        concat7 = keras.layers.Concatenate()([up4_2, conv4_1])
        conv4_2 = self.standard_unit(concat7, filters[3])
        up3_3 = self.upsampling_block(conv4_2, filters[3])
        concat8 = keras.layers.Concatenate()([up3_3, conv3_1, conv3_2])
        conv3_3 = self.standard_unit(concat8, filters[2])
        up2_4 = self.upsampling_block(conv3_3, filters[2])
        concat9 = keras.layers.Concatenate()([up2_4, conv2_1, conv2_2, conv2_3])
        conv2_4 = self.standard_unit(concat9, filters[1])
        up1_5 = self.upsampling_block(conv2_4, filters[1])
        concat10 = keras.layers.Concatenate()([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
        conv1_5 = self.standard_unit(concat10, filters[0])

        output4 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', padding='same')(conv1_5)

        model = keras.models.Model(inputs, output4)
        return model

class UNet(Model):
    
    def build_model(self, filters):
        inputs = keras.layers.Input(self.input_size)
        # downsampling
        # 取名name
        conv1 = self.standard_unit(inputs, filters[0])
        pooling1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.standard_unit(pooling1, filters[1])
        pooling2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.standard_unit(pooling2, filters[2])
        pooling3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.standard_unit(pooling3, filters[3])
        pooling4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        conv5 = self.standard_unit(pooling4, filters[4])

        # upsampling
        up1 = self.upsampling_block(conv5, filters[3])
        concat1 = keras.layers.Concatenate()([conv4, up1])
        conv6 = self.standard_unit(concat1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        concat2 = keras.layers.Concatenate()([conv3, up2])
        conv7 = self.standard_unit(concat2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        concat3 = keras.layers.Concatenate()([conv2, up3])
        conv8 = self.standard_unit(concat3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        concat4 = keras.layers.Concatenate()([conv1, up4])
        conv9 = self.standard_unit(concat4, filters[0])

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

        model = keras.models.Model(inputs, outputs)
        return model

class SDUNet(Model):
    def __init__(self, image_size, learning_rate):
        self.input_size = (image_size, image_size, 3)
        self.learning_rate = learning_rate

    def standard_unit(self, inputs, filters,keep_prob=0.01):
        conv1 = keras.layers.Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
        conv1 = tf.keras.layers.SpatialDropout2D(keep_prob)(conv1)
        bn1 = keras.layers.BatchNormalization()(conv1)
        relu1 = keras.layers.Activation('relu')(bn1)
        conv2 = keras.layers.Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same')(relu1)
        conv2 = tf.keras.layers.SpatialDropout2D(keep_prob)(conv2)
        bn2 = keras.layers.BatchNormalization()(conv2)
        relu2 = keras.layers.Activation('relu')(bn2)
        # layer = keras.layers.Lambda(lambda x: x, name=name)(bn2)
        return relu2

    def upsampling_block(self, inputs, filters,keep_prob=0.01,block_size=5):
        upsampling = keras.layers.UpSampling2D((2, 2))(inputs)
        conv = keras.layers.Conv2D(filters, (2, 2),  padding='same', kernel_initializer='he_normal')(upsampling)
        conv = tf.keras.layers.SpatialDropout2D(keep_prob)(conv)
        bn = keras.layers.BatchNormalization()(conv)
        relu = keras.layers.Activation('relu')(bn)
        # layer = keras.layers.Lambda(lambda x: x, name=name)(bn)
        return relu

    def build_model(self, filters):
        inputs = keras.layers.Input(self.input_size)
        # downsampling
        # 取名name
        conv1 = self.standard_unit(inputs, filters[0])
        pooling1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.standard_unit(pooling1, filters[1])
        pooling2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.standard_unit(pooling2, filters[2])
        pooling3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.standard_unit(pooling3, filters[3])
        pooling4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        conv5 = self.standard_unit(pooling4, filters[4])

        # upsampling
        up1 = self.upsampling_block(conv5, filters[3])
        concat1 = keras.layers.Concatenate()([conv4, up1])
        conv6 = self.standard_unit(concat1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        concat2 = keras.layers.Concatenate()([conv3, up2])
        conv7 = self.standard_unit(concat2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        concat3 = keras.layers.Concatenate()([conv2, up3])
        conv8 = self.standard_unit(concat3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        concat4 = keras.layers.Concatenate()([conv1, up4])
        conv9 = self.standard_unit(concat4, filters[0])

        outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

        model = keras.models.Model(inputs, outputs)
        return model

        

class AttentionUNet(Model):
    
  def __init__(self,image_size,learning_rate):
    self.img_shape=(image_size, image_size, 3)
    self.learning_rate = learning_rate
    self.df=64
    self.uf=64
    
  def build_model(self, filters):
    def conv2d(layer_input,filters,dropout_rate=0,bn=False):
      d=layers.Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(layer_input)
      if bn:
        d=layers.BatchNormalization()(d)
      d=layers.Activation('relu')(d)
      
      d=layers.Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(d)
      if bn:
        d=layers.BatchNormalization()(d)
      d=layers.Activation('relu')(d)
      
      if dropout_rate:
        d=layers.Dropout(dropout_rate)(d)
      
      return d
    
    def deconv2d(layer_input,filters,bn=False):
      u=layers.UpSampling2D((2,2))(layer_input)
      u=layers.Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same')(u)
      if bn:
        u=layers.BatchNormalization()(u)
      u=layers.Activation('relu')(u)
      
      return u
    
    
    
    inputs=layers.Input(shape=self.img_shape)
    
    conv1=conv2d(inputs,self.df)
    pool1=layers.MaxPooling2D((2,2))(conv1)
    
    conv2=conv2d(pool1,self.df*2,bn=True)
    pool2=layers.MaxPooling2D((2,2))(conv2)
    
    conv3=conv2d(pool2,self.df*4,bn=True)
    pool3=layers.MaxPooling2D((2,2))(conv3)
    
    conv4=conv2d(pool3,self.df*8,dropout_rate=0.5,bn=True)
    pool4=layers.MaxPooling2D((2,2))(conv4)
    
    conv5=conv2d(pool4,self.df*16,dropout_rate=0.5,bn=True)
    
    up6=deconv2d(conv5,self.uf*8,bn=True)
    conv6=self.attention_block(up6,conv4,self.uf*8,bn=True)
    up6=layers.Concatenate()([up6,conv6])
    conv6=conv2d(up6,self.uf*8)
    
    up7=deconv2d(conv6,self.uf*4,bn=True)
    conv7=self.attention_block(up7,conv3,self.uf*4,bn=True)
    up7=layers.Concatenate()([up7,conv7])
    conv7=conv2d(up7,self.uf*4)
    
    up8=deconv2d(conv7,self.uf*2,bn=True)
    conv8=self.attention_block(up8,conv2,self.uf*2,bn=True)
    up8=layers.Concatenate()([up8,conv8])
    conv8=conv2d(up8,self.uf*2)
    
    up9=deconv2d(conv8,self.uf,bn=True)
    conv9=self.attention_block(up9,conv1,self.uf,bn=True)
    up9=layers.Concatenate()([up9,conv9])
    conv9=conv2d(up9,self.uf)
    
    outputs=layers.Conv2D(1,kernel_size=(1,1),strides=(1,1),activation='sigmoid')(conv9)
    
    model=keras.models.Model(inputs=inputs,outputs=outputs)
    return model

class BCDUNet(Model):
    def __init__(self, image_size, learning_rate):
        self.input_size = (image_size, image_size, 3)
        self.learning_rate = learning_rate

    def BConvLSTM(self, in1, in2, d, fi, fo):
        x1 = keras.layers.Reshape(target_shape=(1, np.int32(self.input_size[0]/d), np.int32(self.input_size[1]/d), fi))(in1)
        x2 = keras.layers.Reshape(target_shape=(1, np.int32(self.input_size[0]/d), np.int32(self.input_size[1]/d), fi))(in2)
        merge = keras.layers.concatenate([x1,x2], axis=1) 
        merge = keras.layers.ConvLSTM2D(fo, (3, 3), padding='same', return_sequences=False, go_backwards=True,kernel_initializer='he_normal')(merge)
        return merge

    def build_model(self, filters):

        inputs = keras.layers.Input(self.input_size)

        # encode
        conv1 = self.standard_unit(inputs, filters[0])
        pool1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.standard_unit(pool1, filters[1])
        pool2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.standard_unit(pool2, filters[2])
        pool3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.standard_unit(pool3, filters[3])
        pool4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)
        # D1
        conv5_1 = self.standard_unit(pool4, filters[4])
        # D2
        conv5_2 = self.standard_unit(conv5_1, filters[4])
        # D3
        merge_dense = keras.layers.concatenate([conv5_2, conv5_1], axis=3)
        conv5_3 = self.standard_unit(merge_dense, filters[4])
        
        # decode
        up1 = self.upsampling_block(conv5_3, filters[3])
        LSTM1 = self.BConvLSTM(conv4, up1, 8, filters[3], filters[2])
        conv6 = self.standard_unit(LSTM1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        LSTM2 = self.BConvLSTM(conv3, up2, 4, filters[2], filters[1])
        conv7 = self.standard_unit(LSTM2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        LSTM3 = self.BConvLSTM(conv2, up3, 2, filters[1], filters[0])
        conv8 = self.standard_unit(LSTM3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        LSTM4 = self.BConvLSTM(conv1, up4, 1, filters[0], int(filters[0]/2))
        conv9 = self.standard_unit(LSTM4, filters[0])  

        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(conv9)
        model = keras.models.Model(inputs, outputs)
        return model

class R2UNet(Model):
  def __init__(self, image_size, learning_rate):
       self.input_size = (image_size, image_size, 3)
       self.learning_rate = learning_rate


  def RRCNN_block(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = self.recurrent_block(x1, out_n_filters)
      x3 = self.recurrent_block(x2, out_n_filters)
      x4 = keras.layers.add([x1, x3])
      return x4
   

  def build_model(self, filters):
        inputs = keras.layers.Input(self.input_size)
        # downsampling
        conv1 = self.RRCNN_block(inputs, filters[0])
        pooling1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = self.RRCNN_block(pooling1, filters[1])
        pooling2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
        conv3 = self.RRCNN_block(pooling2, filters[2])
        pooling3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
        conv4 = self.RRCNN_block(pooling3, filters[3])
        pooling4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)

        conv5 = self.RRCNN_block(pooling4, filters[4])

        # upsampling
        up1 = self.upsampling_block(conv5, filters[3])
        concat1 = keras.layers.concatenate([up1, conv4], axis=3)
        conv6 = self.RRCNN_block(concat1, filters[3])
        up2 = self.upsampling_block(conv6, filters[2])
        concat2 = keras.layers.concatenate([up2, conv3], axis=3)
        conv7 = self.RRCNN_block(concat2, filters[2])
        up3 = self.upsampling_block(conv7, filters[1])
        concat3 = keras.layers.concatenate([up3, conv2], axis=3)
        conv8 = self.RRCNN_block(concat3, filters[1])
        up4 = self.upsampling_block(conv8, filters[0])
        concat4 = keras.layers.concatenate([up4, conv1], axis=3)
        conv9 = self.RRCNN_block(concat4, filters[0])

       # Output
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='he_normal')(conv9)
        model = keras.models.Model(inputs, outputs)
        return model
  
# class R2AttentionUNet(Model):

#   def __init__(self, image_size, learning_rate):
#         self.input_size = (image_size, image_size, 3)
#         self.learning_rate = learning_rate
#         self.img_shape=(image_size, image_size, 3)
#         self.df=64
#         self.uf=64

#   def RRCNN_block(self, input_layer, out_n_filters):
     
#       x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
#       x2 = self.residual_block(x1, out_n_filters)
#       x3 = self.residual_block(x2, out_n_filters)
#       x4 = keras.layers.add([x1, x3])
#       return x4
  
#   def self.attention_block(self, F_g,F_l,F_int):
#       g1 = keras.layers.Conv2D(F_int, (1, 1), padding="same")(F_g)
#       g1 = keras.layers.BatchNormalization()(g1)

#       x1 = keras.layers.Conv2D(F_int, (1, 1), padding="same")(F_l)
#       x1 = keras.layers.BatchNormalization()(x1)

#       psi=keras.layers.add([g1, x1])
#       psi=keras.layers.Activation('relu')(psi)

#       psi=keras.layers.Conv2D(1, (1, 1), padding="same")(psi)
#       psi=keras.layers.BatchNormalization()(psi)
#       psi=keras.layers.Activation('sigmoid')(psi)
#       out=keras.layers.multiply([F_l,psi])
#       return out
  
#   def build_model(self, filters):
#         inputs = keras.layers.Input(self.input_size)

#         # downsampling
#         conv1 = self.RRCNN_block(inputs, filters[0])
#         pooling1 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
#         conv2 = self.RRCNN_block(pooling1, filters[1])
#         pooling2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
#         conv3 = self.RRCNN_block(pooling2, filters[2])
#         pooling3 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
#         conv4 = self.RRCNN_block(pooling3, filters[3])
#         pooling4 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)

#         conv5 = self.RRCNN_block(pooling4, filters[4])

#         # upsampling
#         up1 = self.upsampling_block(conv5, filters[3])
#         at1 = self.self.attention_block(up1, conv4, self.uf)
#         concat1 = keras.layers.concatenate([at1, conv4], axis=3)
#         conv6 = self.RRCNN_block(concat1, filters[3])

#         up2 = self.upsampling_block(conv6, filters[2])
#         at2 = self.self.attention_block(up2, conv3, self.uf)
#         concat2 = keras.layers.concatenate([at2, conv3], axis=3)
#         conv7 = self.RRCNN_block(concat2, filters[2])

#         up3 = self.upsampling_block(conv7, filters[1])
#         at3 = self.self.attention_block(up3, conv2, self.uf)
#         concat3 = keras.layers.concatenate([at3, conv2], axis=3)
#         conv8 = self.RRCNN_block(concat3, filters[1])

#         up4 = self.upsampling_block(conv8, filters[0])
#         at4 = self.self.attention_block(up4, conv1, self.uf)
#         concat4 = keras.layers.concatenate([at4, conv1], axis=3)
#         conv9 = self.RRCNN_block(concat4, filters[0])

#         # Output
#         outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='he_normal')(conv9)
#         model = keras.models.Model(inputs, outputs)
#         return model
  
class FRUNet(Model):
  def residual_block(self, inputs, filters, strides=1, is_first=False):
      # feature extraction
      if not is_first:
          bn1 = keras.layers.BatchNormalization()(inputs)
          relu1 = keras.layers.Activation("relu")(bn1)
          conv1 = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=strides)(relu1)
      else:
          conv1 = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=strides)(inputs)
      bn2 = keras.layers.BatchNormalization()(conv1)
      relu2 = keras.layers.Activation("relu")(bn2)
      conv2 = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1)(relu2)

      # shortcut
      shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(inputs)
      bn3 = keras.layers.BatchNormalization()(shortcut)

      # addition
      addition = keras.layers.add([conv2, bn3])
      return addition

  def build_model(self, filters):
      inputs = keras.layers.Input(self.input_size)

      # encode
      residual1 = self.residual_block(inputs, filters[0], 1, True)
      residual2 = self.residual_block(residual1, filters[1], 2)
      residual3 = self.residual_block(residual2, filters[2], 2)
      residual4 = self.residual_block(residual3, filters[3], 2)
      residual5 = self.residual_block(residual4, filters[4], 2)

      # decode
      up1 = self.upsampling_block(residual5, filters[3])
      concat1 = keras.layers.Concatenate()([up1, residual4])
      residual6 = self.residual_block(concat1, filters[3])
      up2 = self.upsampling_block(residual6, filters[2])
      concat2 = keras.layers.Concatenate()([up2, residual3])
      residual7 = self.residual_block(concat2, filters[2])
      up3 = self.upsampling_block(residual7, filters[1])
      concat3 = keras.layers.Concatenate()([up3, residual2])
      residual8 = self.residual_block(concat3, filters[1])
      up4 = self.upsampling_block(residual8, filters[0])
      concat4 = keras.layers.Concatenate()([up4, residual1])
      residual9 = self.residual_block(concat4, filters[0])
      
      outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(residual9)
      model = keras.models.Model(inputs, outputs)
      return model  

class R2UNetPlusPlus(Model):
  def RRCNN_block(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = self.residual_block(x1, out_n_filters)
      x3 = self.residual_block(x2, out_n_filters)
      x4 = keras.layers.add([x1, x3])
      return x4
  
  def residual_block(self,inputs,filters):
      conv1 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
      bn1 = keras.layers.BatchNormalization()(conv1)
      conv2 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
      bn2 = keras.layers.BatchNormalization()(conv2)
      addition = keras.layers.add([bn2, inputs])
      return addition
  def build_model(self, filters):
      inputs = keras.layers.Input(self.input_size)

      # down sampling
      conv1_1  = self.RRCNN_block(inputs, filters[0])
      pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1_1 )

      conv2_1 = self.RRCNN_block(pool1, filters[1])
      pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2_1)

      conv3_1 = self.RRCNN_block(pool2, filters[2])
      pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3_1)

      conv4_1 = self.RRCNN_block(pool3, filters[3])
      pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4_1)

      conv5_1 = self.RRCNN_block(pool4, filters[4])

      #skip connection
      up1_2 = self.upsampling_block(conv2_1, filters[1])     
      concat1 = keras.layers.Concatenate()([up1_2, conv1_1])    
      conv1_2 = self.RRCNN_block(concat1, filters[0])

      up2_2 = self.upsampling_block(conv3_1, filters[2])
      concat2 = keras.layers.Concatenate()([up2_2, conv2_1]) 
      conv2_2 = self.RRCNN_block(concat2, filters[1])

      up1_3 = self.upsampling_block(conv2_2, filters[1])
      concat3 = keras.layers.Concatenate()([up1_3, conv1_1, conv1_2])
      conv1_3 = self.RRCNN_block(concat3, filters[0])


      up3_2 = self.upsampling_block(conv4_1, filters[3])
      concat4 = keras.layers.Concatenate()([up3_2, conv3_1])
      conv3_2 = self.RRCNN_block(concat4, filters[2])
      up2_3 = self.upsampling_block(conv3_2, filters[2])
      concat5 = keras.layers.Concatenate()([up2_3, conv2_1, conv2_2])
      conv2_3 = self.RRCNN_block(concat5, filters[1])
      up1_4 = self.upsampling_block(conv2_3, filters[1])
      concat6 = keras.layers.Concatenate()([up1_4, conv1_1, conv1_2, conv1_3])
      conv1_4 = self.RRCNN_block(concat6, filters[0])

      # upsampling
      up4_2 = self.upsampling_block(conv5_1, filters[4])
      concat7 = keras.layers.Concatenate()([up4_2, conv4_1])
      conv4_2 = self.RRCNN_block(concat7, filters[3])
      up3_3 = self.upsampling_block(conv4_2, filters[3])
      concat8 = keras.layers.Concatenate()([up3_3, conv3_1, conv3_2])
      conv3_3 = self.RRCNN_block(concat8, filters[2])
      up2_4 = self.upsampling_block(conv3_3, filters[2])
      concat9 = keras.layers.Concatenate()([up2_4, conv2_1, conv2_2, conv2_3])
      conv2_4 = self.RRCNN_block(concat9, filters[1])
      up1_5 = self.upsampling_block(conv2_4, filters[1])
      concat10 = keras.layers.Concatenate()([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
      conv1_5 = self.RRCNN_block(concat10, filters[0])

      # Output
      outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(conv1_5)
      model = keras.models.Model(inputs, outputs)

      return model

class DenseUNet(Model):
  def dense_block(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = keras.layers.BatchNormalization()(x1)
      x3 = keras.layers.Activation("relu")(x2)

      x4 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(x3)
      x5 = keras.layers.BatchNormalization()(x4)
      x6 = keras.layers.Activation("relu")(x5)

      x7 = keras.layers.Concatenate()([input_layer, x6])
      return x7
   
  def transition_down(self, input_layer, out_n_filters):
    x1 = keras.layers.Conv2D(out_n_filters, (1, 1), padding="same")(input_layer)
    x2 = keras.layers.BatchNormalization()(x1)
    x3 = keras.layers.Activation("relu")(x2)
    x4 = keras.layers.MaxPooling2D((2, 2))(x3)
    return x4
   
  def transition_up(self, input_layer, out_n_filters):
    x1 = keras.layers.Conv2DTranspose(out_n_filters, (3, 3), strides=(2, 2), padding="same")(input_layer)
    return x1
  
  def build_model(self, filters):
    inputs = keras.layers.Input(self.input_size)
    conv1 = keras.layers.Conv2D(filters[0], (3, 3), padding="same")(inputs)

    # down sampling 
    dense1 = self.dense_block(conv1, filters[0])
    trans1 = self.transition_down(dense1, filters[1])

    dense2 = self.dense_block(trans1, filters[1])
    trans2 = self.transition_down(dense2, filters[2])

    dense3 = self.dense_block(trans2, filters[2])
    trans3 = self.transition_down(dense3, filters[3])

    dense4  = self.dense_block(trans3, filters[3])
    trans4 = self.transition_down(dense4, filters[4])

    dense5 = self.dense_block(trans4, filters[4])

    # up sampling
    trans5 = self.transition_up(dense5, filters[4])
    concat1 = keras.layers.Concatenate()([trans5, dense4])

    dense6 = self.dense_block(concat1, filters[4])
    trans6 = self.transition_up(dense6, filters[3])
    concat2 = keras.layers.Concatenate()([trans6, dense3])
  
    dense7 = self.dense_block(concat2, filters[3])
    trans7 = self.transition_up(dense7, filters[2])
    concat3 = keras.layers.Concatenate()([trans7, dense2])

    dense8 = self.dense_block(concat3, filters[2])
    trans8 = self.transition_up(dense8, filters[1])
    concat4 = keras.layers.Concatenate()([trans8, dense1])

    dense9 = self.dense_block(concat4, filters[1])

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(dense9)
    model = keras.models.Model(inputs, outputs)
    return model

class RecurrentUNet(Model):
    # def recurrent_block(self, input_layer, out_n_filters):
    #   x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
    #   x2 = keras.layers.BatchNormalization()(x1)
    #   x3 = keras.layers.Activation("relu")(x2)

    #   x4 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(x3)
    #   x5 = keras.layers.BatchNormalization()(x4)
    #   x6 = keras.layers.Activation("relu")(x5)

    #   x7 = keras.layers.Concatenate()([input_layer, x6])
    #   return x7
   
    def transition_down(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (1, 1), padding="same")(input_layer)
      x2 = keras.layers.BatchNormalization()(x1)
      x3 = keras.layers.Activation("relu")(x2)
      x4 = keras.layers.MaxPooling2D((2, 2))(x3)
      return x4
   
    def transition_up(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2DTranspose(out_n_filters, (3, 3), strides=(2, 2), padding="same")(input_layer)
      return x1
   
    def build_model(self, filters):
      inputs = keras.layers.Input(self.input_size)
      conv1 = keras.layers.Conv2D(filters[0], (3, 3), padding="same")(inputs)

      # down sampling 
      recurrent1 = self.recurrent_block(conv1, filters[0])
      trans1 = self.transition_down(recurrent1, filters[1])

      recurrent2 = self.recurrent_block(trans1, filters[1])
      trans2 = self.transition_down(recurrent2, filters[2])

      recurrent3 = self.recurrent_block(trans2, filters[2])
      trans3 = self.transition_down(recurrent3, filters[3])

      recurrent4  = self.recurrent_block(trans3, filters[3])
      trans4 = self.transition_down(recurrent4, filters[4])

      recurrent5 = self.recurrent_block(trans4, filters[4])

      # up sampling
      trans5 = self.transition_up(recurrent5, filters[4])
      concat1 = keras.layers.Concatenate()([trans5, recurrent4])

      recurrent6 = self.recurrent_block(concat1, filters[4])
      trans6 = self.transition_up(recurrent6, filters[3])
      concat2 = keras.layers.Concatenate()([trans6, recurrent3])
  
      recurrent7 = self.recurrent_block(concat2, filters[3])
      trans7 = self.transition_up(recurrent7, filters[2])
      concat3 = keras.layers.Concatenate()([trans7, recurrent2])

      recurrent8 = self.recurrent_block(concat3, filters[2])
      trans8 = self.transition_up(recurrent8, filters[1])
      concat4 = keras.layers.Concatenate()([trans8, recurrent1])

      recurrent9 = self.recurrent_block(concat4, filters[1])

      outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(recurrent9)
      model = keras.models.Model(inputs, outputs)

      return model
  
class ResUNet(Model):
    def residual_block(self,inputs,filters):
        conv1 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        bn1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
        bn2 = keras.layers.BatchNormalization()(conv2)
        add = keras.layers.add([conv1, bn2])
        return add
    
    def residual_block_first(self, inputs, filters):
        bn=keras.layers.BatchNormalization()(inputs)
        conv1 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn)
        bn1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
        bn2 = keras.layers.BatchNormalization()(conv2)
        add = keras.layers.add([inputs, bn2])
        return add
    
    def build_model(self, filters):
      inputs = keras.layers.Input(self.input_size)
      conv1 = keras.layers.Conv2D(filters[0], (3, 3), padding="same")(inputs)


      # down sampling
      res1 = self.residual_block_first(conv1, filters[0])
      pool1 = keras.layers.MaxPooling2D((2, 2))(res1)

      res2 = self.residual_block(pool1, filters[1])
      pool2 = keras.layers.MaxPooling2D((2, 2))(res2)

      res3 = self.residual_block(pool2, filters[2])
      pool3 = keras.layers.MaxPooling2D((2, 2))(res3)

      res4 = self.residual_block(pool3, filters[3])
      pool4 = keras.layers.MaxPooling2D((2, 2))(res4)

      res5 = self.residual_block(pool4, filters[4])

      # up sampling
      ups = keras.layers.UpSampling2D((2, 2))(res5)
      concat1 = keras.layers.Concatenate()([ups, res4])
      res6 = self.residual_block(concat1, filters[4])

      ups = keras.layers.UpSampling2D((2, 2))(res6)
      concat2 = keras.layers.Concatenate()([ups, res3])
      res7 = self.residual_block(concat2, filters[3])

      ups = keras.layers.UpSampling2D((2, 2))(res7)
      concat3 = keras.layers.Concatenate()([ups, res2])
      res8 = self.residual_block(concat3, filters[2])
    
      ups = keras.layers.UpSampling2D((2, 2))(res8)
      concat4 = keras.layers.Concatenate()([ups, res1])
      res9 = self.residual_block(concat4, filters[1])

      outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(res9)
      model = keras.models.Model(inputs, outputs)
      return model

# https://github.com/nibtehaz/MultiResUNet/blob/master/tensorflow/MultiResUNet.py
class MultiResUNet(Model):
  def conv2d_bn(self, x, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False):
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if activation is not None:
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.Activation(activation)(x)
    return x

  def trans_conv2d_bn(self, x, filters, kernel_size, strides=1, padding='same'):
    x = keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = keras.layers.BatchNormalization()(x)
    return x

  def MultiResBlock(self,U, inp, alpha = 1.67):
    W = alpha * U
    shortcut = inp
    shortcut = self.conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                        int(W*0.5), (1,1), activation=None, padding='same')
    conv3x3 = self.conv2d_bn(inp, int(W*0.167), (3,3), activation='relu', padding='same')
    conv5x5 = self.conv2d_bn(conv3x3, int(W*0.333), (3,3), activation='relu', padding='same')
    conv7x7 = self.conv2d_bn(conv5x5, int(W*0.5), (3,3), activation='relu', padding='same')
    out = keras.layers.Concatenate()([conv3x3, conv5x5, conv7x7])
    out = keras.layers.BatchNormalization()(out)
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization()(out)
    return out

  def ResPath(self,filters, length, inp):
    shortcut = inp
    shortcut = self.conv2d_bn(shortcut, filters, (1,1), activation=None, padding='same')
    out = self.conv2d_bn(inp, filters, (3,3), activation='relu', padding='same')
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization()(out)
    for i in range(length-1):
      shortcut = out
      shortcut = self.conv2d_bn(shortcut, filters, (1,1), activation=None, padding='same')
      out = self.conv2d_bn(out, filters, (3,3), activation='relu', padding='same')
      out = keras.layers.add([shortcut, out])
      out = keras.layers.Activation('relu')(out)
      out = keras.layers.BatchNormalization()(out)
    return out

  def build_model(self, filters):
    inputs = keras.layers.Input(self.input_size)
    mresblock1 = self.MultiResBlock(filters[0], inputs)
    pool1 = keras.layers.MaxPooling2D((2, 2))(mresblock1)
    mresblock1 = self.ResPath(filters[0], 4, mresblock1)

    mresblock2 = self.MultiResBlock(filters[1], pool1)
    pool2 = keras.layers.MaxPooling2D((2, 2))(mresblock2)
    mresblock2 = self.ResPath(filters[1], 3, mresblock2)

    mresblock3 = self.MultiResBlock(filters[2], pool2)
    pool3 = keras.layers.MaxPooling2D((2, 2))(mresblock3)
    mresblock3 = self.ResPath(filters[2], 2, mresblock3)

    mresblock4 = self.MultiResBlock(filters[3], pool3)
    pool4 = keras.layers.MaxPooling2D((2, 2))(mresblock4)
    mresblock4 = self.ResPath(filters[3], 1, mresblock4)

    mresblock5 = self.MultiResBlock(filters[4], pool4)

    up6 = keras.layers.Concatenate()([keras.layers.Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4])
    mresblock6 = self.MultiResBlock(filters[3], up6)
    mresblock6 = self.ResPath(filters[3], 1, mresblock6)

    up7 = keras.layers.Concatenate()([keras.layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3])
    mresblock7 = self.MultiResBlock(filters[2], up7)
    mresblock7 = self.ResPath(filters[2], 2, mresblock7)

    up8 = keras.layers.Concatenate()([keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2])
    mresblock8 = self.MultiResBlock(filters[1], up8)
    mresblock8 = self.ResPath(filters[1], 3, mresblock8)

    up9 = keras.layers.Concatenate()([keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1])
    mresblock9 = self.MultiResBlock(filters[0], up9)
    mresblock9 = self.ResPath(filters[0], 4, mresblock9)

    conv10 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(mresblock9)

    model = keras.models.Model(inputs=[inputs], outputs=[conv10])

    return model
       
# https://github.com/AngeLouCN/DC-UNet/blob/main/model.py
class DCUNet(Model):
  def __init__(self, image_size, learning_rate):
    self.input_size = (image_size, image_size, 3)
    self.learning_rate = learning_rate
    self.img_shape=(image_size, image_size, 3)
    self.df=64
    self.uf=64
    
  def conv2d_bn(self,x, filters, kernel_size, padding='same', strides=(1, 1), activation='relu', name=None):
    x = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    if activation is not None:
      x = keras.layers.Activation(activation)(x)
    return x
  
  def trans_conv2d_bn(self,x, filters, kernel_size, padding='same', strides=(2, 2), activation='relu', name=None):
    x = keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = keras.layers.BatchNormalization(scale=False)(x)
    if activation is not None:
      x = keras.layers.Activation(activation)(x)
    return x
  
  def ResPath(self,filters, length, inp):
    shortcut = inp
    shortcut = self.conv2d_bn(shortcut, filters, (1, 1), activation=None, padding='same')
    out = self.conv2d_bn(inp, filters, (3, 3), activation='relu', padding='same')
    out = keras.layers.add([shortcut, out])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization()(out)
    for i in range(length-1):
      shortcut = out
      shortcut = self.conv2d_bn(shortcut, filters, (1, 1), activation=None, padding='same')
      out = self.conv2d_bn(out, filters, (3, 3), activation='relu', padding='same')
      out = keras.layers.add([shortcut, out])
      out = keras.layers.Activation('relu')(out)
      out = keras.layers.BatchNormalization()(out)
    return out
  

  def DCBlock(self,U, inp, alpha = 1.67):
    W = alpha * U
    shortcut = inp
    shortcut = self.conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                        int(W*0.5), (1,1), activation=None, padding='same')
    conv3x3_1 = self.conv2d_bn(inp, int(W*0.167), (3,3), activation='relu', padding='same')
    conv5x5_1 = self.conv2d_bn(conv3x3_1, int(W*0.333), (3,3), activation='relu', padding='same')
    conv7x7_1 = self.conv2d_bn(conv5x5_1, int(W*0.5), (3,3), activation='relu', padding='same')
    out1 = keras.layers.Concatenate(axis=3)([conv3x3_1, conv5x5_1, conv7x7_1])
    out1 = keras.layers.BatchNormalization(axis=3)(out1)

    conv3x3_2 = self.conv2d_bn(inp, int(W*0.167), (3,3), activation='relu', padding='same')
    conv5x5_2 = self.conv2d_bn(conv3x3_2, int(W*0.333), (3,3), activation='relu', padding='same')
    conv7x7_2 = self.conv2d_bn(conv5x5_2, int(W*0.5), (3,3), activation='relu', padding='same')
    out2 = keras.layers.Concatenate(axis=3)([conv3x3_2, conv5x5_2, conv7x7_2])
    out2 = keras.layers.BatchNormalization(axis=3)(out2)

    out = keras.layers.Add()([shortcut,out1,out2])
    out = keras.layers.Activation('relu')(out)
    out = keras.layers.BatchNormalization(axis=3)(out)
    return out

  def build_model(self, filters):
    inputs = keras.layers.Input(self.input_size)
    dcblock1 = self.DCBlock(filters[0], inputs)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(dcblock1)
    respath1 = self.ResPath(filters[0], 4, dcblock1)

    dcblock2 = self.DCBlock(filters[1], pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(dcblock2)
    respath2 = self.ResPath(filters[1], 3, dcblock2)

    dcblock3 = self.DCBlock(filters[2], pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(dcblock3)
    respath3 = self.ResPath(filters[2], 2, dcblock3)

    dcblock4 = self.DCBlock(filters[3], pool3)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(dcblock4)
    respath4 = self.ResPath(filters[3], 1, dcblock4)

    dcblock5 = self.DCBlock(filters[4], pool4)

    up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters[3], (2, 2), strides=(2, 2), padding='same')(dcblock5) , respath4], axis=3)
    dcblock6 = self.DCBlock(filters[3], up6)
    
    up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(dcblock6) , respath3], axis=3)
    dcblock7 = self.DCBlock(filters[2], up7)

    up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(dcblock7) , respath2], axis=3)
    dcblock8 = self.DCBlock(filters[1], up8)

    up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(dcblock8) , respath1], axis=3)
    dcblock9 = self.DCBlock(filters[0], up9)

    conv10 = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(dcblock9)

    model = keras.models.Model(inputs=[inputs], outputs=[conv10])

    return model

# https://github.com/kiharalab/ACC-UNet/blob/main/ACC_UNet/ACC_UNet_lite.py#L16
# class ACCUNet(Model):

# https://github.com/1044197988/TF.Keras-Commonly-used-models/blob/master/%E5%B8%B8%E7%94%A8%E5%88%86%E5%89%B2%E6%A8%A1%E5%9E%8B/Unet_family/Unet_family.py#L427    
class NestedUNet(Model):
  def conv_block_nested(self, inputs, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(inputs)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.ReLU()(conv)

    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.ReLU()(conv)

    return conv

  def up_conv(self, inputs, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides)(inputs)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.ReLU()(conv)

    return conv

  def build_model(self, filters):
    inputs = keras.layers.Input((None, None, 1))
    conv1 = self.conv_block_nested(inputs, filters[0])
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = self.conv_block_nested(pool1, filters[1])
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = self.conv_block_nested(pool2, filters[2])
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = self.conv_block_nested(pool3, filters[3])
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = self.conv_block_nested(pool4, filters[4])

    up6 = self.up_conv(conv5, filters[3])
    merge6 = keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = self.conv_block_nested(merge6, filters[3])

    up7 = self.up_conv(conv6, filters[2])
    merge7 = keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = self.conv_block_nested(merge7, filters[2])

    up8 = self.up_conv(conv7, filters[1])
    merge8 = keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = self.conv_block_nested(merge8, filters[1])

    up9 = self.up_conv(conv8, filters[0])
    merge9 = keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = self.conv_block_nested(merge9, filters[0])

    conv10 = keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = keras.models.Model(inputs=[inputs], outputs=[conv10])

    return model

# https://github.com/clguo/CAR-UNet
class CARUNet(Model):
  def meca_block(self,input_feature, k_size=3):
      channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
      channel = input_feature.shape[channel_axis]

      shared_layer_one = tf.keras.layers.Conv1D(filters=1, kernel_size=k_size, strides=1, kernel_initializer='he_normal', use_bias=False,
                                padding="same")

      avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
      avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)

      assert avg_pool.shape[1:] == (1, 1, channel)
      avg_pool = tf.keras.layers.Permute((3, 1, 2))(avg_pool)
      avg_pool = layers.Lambda(self.squeeze)(avg_pool)
      avg_pool = shared_layer_one(avg_pool)
      avg_pool = layers.Lambda(self.unsqueeze)(avg_pool)
      avg_pool = tf.keras.layers.Permute((2, 3, 1))(avg_pool)
      assert avg_pool.shape[1:] == (1, 1, channel)

      max_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
      max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
      assert max_pool.shape[1:] == (1, 1, channel)
      max_pool = tf.keras.layers.Permute((3, 1, 2))(max_pool)
      max_pool = tf.keras.layers.Lambda(self.squeeze)(max_pool)
      max_pool = shared_layer_one(max_pool)
      max_pool = tf.keras.layers.Lambda(self.unsqueeze)(max_pool)
      max_pool = tf.keras.layers.Permute((2, 3, 1))(max_pool)
      assert max_pool.shape[1:] == (1, 1, channel)

      eca_feature = tf.keras.layers.add([avg_pool, max_pool])
      eca_feature = tf.keras.layers.Activation('sigmoid')(eca_feature)

      if tf.keras.backend.image_data_format() == "channels_first":
          eca_feature = tf.keras.layers.Permute((3, 1, 2))(eca_feature)

      return tf.keras.layers.Multiply()([input_feature, eca_feature])


  def unsqueeze(self,input):
      return tf.keras.backend.expand_dims(input, axis=-1)


  def squeeze(self,input):
      return tf.keras.backend.squeeze(input, axis=-1)

  def DropBlock2D(self,net, keep_prob, block_size, data_format='channels_last'):
      if data_format == 'channels_last':
          _, h, w, _ = net.get_shape().as_list()
      else:
          _, _, h, w = net.get_shape().as_list()
      gamma = (1. - keep_prob) / block_size ** 2
      for i in range(block_size):
          for j in range(block_size):
              if data_format == 'channels_last':
                  net = net - tf.cast((tf.random.uniform([1, h - block_size + 1, w - block_size + 1, 1]) > gamma), tf.float32) * net
              else:
                  net = net - tf.cast((tf.random.uniform([1, 1, h - block_size + 1, w - block_size + 1]) > gamma), tf.float32) * net
      return net
  
  def convolution_block_dropblock(self,inputs, filters, size,  activation=True,keep_prob=0.01,block_size=7):
      x = layers.Conv2D(filters, kernel_size=size,strides=(1,1), padding='same')(inputs)
      x = tf.keras.layers.SpatialDropout2D(keep_prob)(x)
      if activation == True:
          x = keras.layers.Activation('relu')(x)
      return x
  def BatchActivate(self,x):
      x = keras.layers.BatchNormalization()(x)
      x = keras.layers.Activation('relu')(x)
      return x
    
  def residual_drop_block(self,inputs, filters=16, batch_activate=False,keep_prob=0.01,block_size=7):
    x = self.BatchActivate(inputs)
    x = self.convolution_block_dropblock(x, filters, size=(3,3), keep_prob=keep_prob,block_size=block_size)
    x = self.convolution_block_dropblock(x, filters, size=(3,3), keep_prob=keep_prob,block_size=block_size)
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, kernel_size=(1,1), padding="same")(inputs)
         
    x = layers.add([x, inputs])
    if batch_activate:
        x = self.BatchActivate(x)
    return x
  
  def RCAB(self,inputs, filters, batch_activate=True,k_size=3, keep_prob=0.01,block_size=7):
    
    f = self.BatchActivate(inputs)
    f = self.convolution_block_dropblock(f, filters, size=(3,3), keep_prob=keep_prob, block_size=block_size)
    f = self.convolution_block_dropblock(f, filters, size=(3,3), keep_prob=keep_prob, block_size=block_size)
    x = self.meca_block(f, k_size)
    x = tf.keras.layers.Add()([x, inputs])
    return x
   
  def build_model(self, filters, keep_prob=0.8,block_size=7,lr=1e-3):
    inputs = keras.layers.Input(self.input_size)
    
    conv1 = self.residual_drop_block(inputs, filters=filters[0] ,keep_prob=keep_prob,block_size=block_size)
    conv1 = self.RCAB(conv1, filters=filters[0], keep_prob=keep_prob)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    meca1  = self.meca_block(conv1, k_size=3)
    
    conv2 = self.residual_drop_block(pool1, filters=filters[1],keep_prob=keep_prob,block_size=block_size)
    conv2 = self.RCAB(conv2, filters=filters[1], keep_prob=keep_prob)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    meca2  = self.meca_block(conv2, k_size=3)
    
    conv3 = self.residual_drop_block(pool2, filters=filters[2],keep_prob=keep_prob,block_size=block_size)
    conv3 = self.RCAB(conv3, filters=filters[2], keep_prob=keep_prob)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    mec3  = self.meca_block(conv3, k_size=3)
    
    conv4 = self.residual_drop_block(pool3, filters=filters[3],keep_prob=keep_prob,block_size=block_size)
    conv4 = self.RCAB(conv4, filters[3], keep_prob=keep_prob)
    
    deconv3 = keras.layers.Conv2DTranspose(filters[2], (2, 2), strides=(2, 2), padding='same')(conv4)
    uconv3 = keras.layers.concatenate([deconv3, mec3])
    uconv3 = self.residual_drop_block(uconv3, filters=filters[2],keep_prob=keep_prob,block_size=block_size)
    uconv3 = self.RCAB(uconv3, filters[2], keep_prob=keep_prob)
    
    deconv2 = keras.layers.Conv2DTranspose(filters[1], (2, 2), strides=(2, 2), padding='same')(uconv3)
    uconv2 = keras.layers.concatenate([deconv2, meca2])
    uconv2 = self.residual_drop_block(uconv2, filters=filters[1],keep_prob=keep_prob,block_size=block_size)
    uconv2 = self.RCAB(uconv2, filters[1], keep_prob=keep_prob)
    
    deconv1 = keras.layers.Conv2DTranspose(filters[0], (2, 2), strides=(2, 2), padding='same')(uconv2)
    uconv1 = keras.layers.concatenate([deconv1, meca1])
    uconv1 = self.residual_drop_block(uconv1, filters=filters[0],keep_prob=keep_prob,block_size=block_size)
    uconv1 = self.RCAB(uconv1, filters[0], keep_prob=keep_prob)
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)
    model = keras.models.Model(inputs, outputs)
    return model
  
# todo
# https://github.com/hiyaroy12/multitask_learning/blob/master/refinenet-image-segmentation/nets/model_v2.py#L94
# class RefineNet(Model):
#   def ResidualConvUnit(self, inputs, filters, kernel_size=(3, 3), padding="same", strides=1):
#     x = keras.layers.ReLU()(inputs)
#     x = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
#     x = keras.layers.ReLU()(x)
#     x = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    
#     return keras.layers.add([inputs, x])
  
#   def unpool(self,inputs,scale):
#     return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale, tf.shape(inputs)[2]*scale])
  
#   def MultiResolutionFusion(self,high_inputs=None,low_inputs=None,up0=2,up1=1,n_i=256):
#     x = keras.layers.Conv2D(n_i, (3, 3), padding="same")(high_inputs)
#     g0 = self.unpool(x, scale=up0)
    
#     if low_inputs is  None:
#       return g0
#     else:
#       g1 = keras.layers.Conv2D(n_i, (3, 3), padding="same")(low_inputs)
#       g1 = self.unpool (g1, scale=up1)
      
#       return keras.layers.add([g0, g1])
    
#   def ChainedResidualPooling(self,inputs, n_i=256):
#     x = keras.layers.ReLU()(inputs)
#     x = keras.layers.MaxPooling2D((5, 5), strides=1, padding="same")(x)
#     x = keras.layers.Conv2D(n_i, (3, 3), padding="same")(x)
    
#     return keras.layers.add([inputs, x])
    
#   def RefineBlock(self,inputs, high_inputs, low_inputs, n_i=256):
#     if low_inputs is not None:
#       rcu_high = self.ResidualConvUnit(high_inputs, n_i)
#       rcu_low = self.ResidualConvUnit(low_inputs, n_i)
#       fuse = self.MultiResolutionFusion(rcu_high, rcu_low, n_i=n_i)
#       fuse_pool = self.ChainedResidualPooling(fuse, n_i=n_i)
#       output= self.ResidualConvUnit(fuse_pool, n_i)
#     else:
#       rcu_high = self.ResidualConvUnit(high_inputs, n_i)
#       fuse_pool = self.ChainedResidualPooling(rcu_high, n_i=n_i)
#       output= self.ResidualConvUnit(fuse_pool, n_i)
#     return output
  
#   def mean_image_subtraction(images, means=[123.68, 116.779, 103.939]):
    
    
  
#   def build_model(self, filters):
#     images = mean_image_subtraction(image)
    
      

      
    
      
      
  
    
    
  
    
    
    
    
    
  