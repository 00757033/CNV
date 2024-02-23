import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
#所有使用到的U-Net，使用統一的介面
#Model:UNet、UNet++、Attention UNet、BCD UNet、FRUNet

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

    def upsampling_block(self, inputs, filters):
        upsampling = keras.layers.UpSampling2D((3, 3))(inputs)
        conv = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(upsampling)
        bn = keras.layers.BatchNormalization()(conv)
        return bn
    
    def standard_unit(self, inputs, filters):
        conv1 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        bn1 = keras.layers.BatchNormalization()(conv1)
        conv2 = keras.layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
        bn2 = keras.layers.BatchNormalization()(conv2)
        return bn2

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
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef, self.jaccard_index])
        return model

class UNet(Model):
    
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
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef, self.jaccard_index])
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
    
    def attention_block(F_g,F_l,F_int,bn=False):
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
    conv6=attention_block(up6,conv4,self.uf*8,bn=True)
    up6=layers.Concatenate()([up6,conv6])
    conv6=conv2d(up6,self.uf*8)
    
    up7=deconv2d(conv6,self.uf*4,bn=True)
    conv7=attention_block(up7,conv3,self.uf*4,bn=True)
    up7=layers.Concatenate()([up7,conv7])
    conv7=conv2d(up7,self.uf*4)
    
    up8=deconv2d(conv7,self.uf*2,bn=True)
    conv8=attention_block(up8,conv2,self.uf*2,bn=True)
    up8=layers.Concatenate()([up8,conv8])
    conv8=conv2d(up8,self.uf*2)
    
    up9=deconv2d(conv8,self.uf,bn=True)
    conv9=attention_block(up9,conv1,self.uf,bn=True)
    up9=layers.Concatenate()([up9,conv9])
    conv9=conv2d(up9,self.uf)
    
    outputs=layers.Conv2D(1,kernel_size=(1,1),strides=(1,1),activation='sigmoid')(conv9)
    
    model=keras.models.Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef, self.jaccard_index])
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
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss,  metrics=[self.dice_coef, self.jaccard_index])
        return model

class R2UNet(Model):
  def __init__(self, image_size, learning_rate):
       self.input_size = (image_size, image_size, 3)
       self.learning_rate = learning_rate

  def RRCNN_block(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = self.residual_block(x1, out_n_filters)
      x3 = self.residual_block(x2, out_n_filters)
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
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss,  metrics=[self.dice_coef, self.jaccard_index])
        return model
  
class R2AttentionUNet(Model):

  def __init__(self, image_size, learning_rate):
        self.input_size = (image_size, image_size, 3)
        self.learning_rate = learning_rate
        self.img_shape=(image_size, image_size, 3)
        self.df=64
        self.uf=64

  def RRCNN_block(self, input_layer, out_n_filters):
     
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = self.residual_block(x1, out_n_filters)
      x3 = self.residual_block(x2, out_n_filters)
      x4 = keras.layers.add([x1, x3])
      return x4
  
  def attention_block(self, F_g,F_l,F_int):
      g1 = keras.layers.Conv2D(F_int, (1, 1), padding="same")(F_g)
      g1 = keras.layers.BatchNormalization()(g1)

      x1 = keras.layers.Conv2D(F_int, (1, 1), padding="same")(F_l)
      x1 = keras.layers.BatchNormalization()(x1)

      psi=keras.layers.add([g1, x1])
      psi=keras.layers.Activation('relu')(psi)

      psi=keras.layers.Conv2D(1, (1, 1), padding="same")(psi)
      psi=keras.layers.BatchNormalization()(psi)
      psi=keras.layers.Activation('sigmoid')(psi)
      out=keras.layers.multiply([F_l,psi])
      return out
  
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
        at1 = self.attention_block(up1, conv4, self.uf)
        concat1 = keras.layers.concatenate([at1, conv4], axis=3)
        conv6 = self.RRCNN_block(concat1, filters[3])

        up2 = self.upsampling_block(conv6, filters[2])
        at2 = self.attention_block(up2, conv3, self.uf)
        concat2 = keras.layers.concatenate([at2, conv3], axis=3)
        conv7 = self.RRCNN_block(concat2, filters[2])

        up3 = self.upsampling_block(conv7, filters[1])
        at3 = self.attention_block(up3, conv2, self.uf)
        concat3 = keras.layers.concatenate([at3, conv2], axis=3)
        conv8 = self.RRCNN_block(concat3, filters[1])

        up4 = self.upsampling_block(conv8, filters[0])
        at4 = self.attention_block(up4, conv1, self.uf)
        concat4 = keras.layers.concatenate([at4, conv1], axis=3)
        conv9 = self.RRCNN_block(concat4, filters[0])

        # Output
        outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid", kernel_initializer='he_normal')(conv9)
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss,  metrics=[self.dice_coef, self.jaccard_index])
        return model
  
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
      addition = keras.layers.Add()([conv2, bn3])
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
      model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
      return model  

class R2UNetPlusPlus(Model):
  def RRCNN_block(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = self.residual_block(x1, out_n_filters)
      x3 = self.residual_block(x2, out_n_filters)
      x4 = keras.layers.add([x1, x3])
      return x4
  
  def residual_block(self,inputs,filters):
      bn = keras.layers.BatchNormalization()(inputs)
      relu = keras.layers.Activation("relu")(bn)
      conv = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(relu)
      bn2 = keras.layers.BatchNormalization()(conv)
      relu2 = keras.layers.Activation("relu")(bn2)
      conv2 = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(relu2)
      add = keras.layers.add([inputs, conv2])
      return add
  
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
      model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
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
    model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
    return model

class RecurrentUNet(Model):
    def recurrent_block(self, input_layer, out_n_filters):
      x1 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(input_layer)
      x2 = keras.layers.BatchNormalization()(x1)
      x3 = keras.layers.Activation("relu")(x2)

      x4 = keras.layers.Conv2D(out_n_filters, (3, 3), padding="same")(x3)
      x5 = keras.layers.BatchNormalization()(x4)
      x6 = keras.layers.Activation("relu")(x5)

      x7 = keras.layers.concatenate()([input_layer, x6])
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
      model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
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
      model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss=self.dice_coef_loss, metrics=[self.dice_coef])
      return model
    
    
class MultiResUNet(Model):
   
       
     

  