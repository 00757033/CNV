import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

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

    def DiceBCELoss(self, y_true, y_pred):
        return 1- (0.5*self.dice_coef(y_true, y_pred) + 0.5*keras.losses.binary_crossentropy(y_true, y_pred))
        
class ActBnLayer(tf.keras.Model):
    def __init__(self, filters, **kwargs):
        super(ActBnLayer, self).__init__()
        self.bn_layer = tf.keras.layers.BatchNormalization()
    def gelu(x):
        return 0.5 * x * (1 + tf.math.erf(x / tf.math.sqrt(2.0)))
    def call(self, inputs, training=False):
        return self.bn_layer(self.gelu(inputs))
    
class ConvMixer(Model):
    def __init__(self, image_size, learning_rate):
        self.input_size = (image_size, image_size, 3)
        self.learning_rate = learning_rate


    def ConvMixerLayer(self, h, kernel_size=7, **kwargs):
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, padding="same")
        x = ActBnLayer(h)
        x = tf.keras.layers.Conv2D(h, 1)
        x = ActBnLayer(h)
        return x
    
    def build_model(self, filters):
        inputs = keras.layers.Input(self.input_size)

        h = 256
        depth = 10
        patch_size = 4
        kernel_size = 7

        # Embedding layer
        x2 = tf.keras.layers.Conv2D(h, patch_size, strides=patch_size, padding="same")(inputs)
        x2 = ActBnLayer(h)(x2)

        # ConvMixer layers
        for i in range(depth):
            x2 = self.ConvMixerLayer(h, kernel_size=kernel_size)(x2) + x2

        # Output layer
        x2 = tf.keras.layers.Conv2DTranspose(1, patch_size, strides=patch_size, padding="same")(x2)
        x2 = tf.keras.layers.Add()([x2, tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)])
        outputs = tf.keras.activations.sigmoid(x2)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
        return model
    

