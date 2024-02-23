import tensorflow as tf
import tensorflow.keras as keras
import keras 
import numpy as np
from segmentation.function import setFolder
from pathlib import Path

# Unet model interface

class Model():
    def __init__(self,image_size =(304,304,3),learning_rate= 0.0001):
        self.image_size = image_size
        self.learning_rate = learning_rate

    def dice_coef(self,y_true, y_pred):
        smooth = 1
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)
    
    def dice_coef_loss(self,y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def jaccard_index(self,y_true, y_pred,smooth=100):
        intersection = keras.backend.sum(y_true * y_pred, axis=[1,2,3])
        sum_ = keras.backend.sum(y_true + y_pred, axis=[1,2,3])
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return keras.backend.mean(jac)
    
    def jaccard_index_loss(self,y_true, y_pred):
        return 1 - self.jaccard_index(y_true, y_pred)
    




    
