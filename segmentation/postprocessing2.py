import numpy as np
import cv2
import os
import glob
from pathlib import Path
import json
import tools.tools as tools
import tensorflow as tf
import pathlib as pl
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix 

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_labels

from skimage import io
from skimage.color import gray2rgb
from skimage import img_as_float
from skimage import img_as_uint


def jaccard_index(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    score = metrics.jaccard_score(y_true, y_pred)
    return score

def dice_coefficient(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    dice_score = np.sum(y_pred[y_true==1])*2.0 / (np.sum(y_pred) + np.sum(y_true))
    return dice_score

class postprocessing():
    def __init__(self,image_path,result_path,post_path):
        self.image_path = image_path
        self.result_path = result_path
        self.post_path = post_path

    def postprocessing(self,model_path):
        # all image in file to crf
        print('.\\'+model_path)
        model = tf.keras.models.load_model('.\\'+model_path)
        # model.load_weights('./'+model_path)
        print(model.summary())
        # for i in os.listdir(self.image_path):
        #     image = cv2.imread(self.image_path + i)
        #     annotated_image = cv2.imread(self.result_path + i)
        #     output_image = self.post_path + i
        #     crf(image, annotated_image,output_image, use_2d = True)





def crf(original_image, predict_image,output_image, use_2d = True):
    # 用在血管分割上
    print("crf")
    # 將分割結果轉成rgb
    annotated_image = predict_image.copy()
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image)

    # 轉成uint32
    annotated_image = annotated_image.astype(np.uint32)
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    print("n_labels",n_labels)
    # print(annotated_image > 0)
    # print("--------------------")

    if use_2d :
        if n_labels > 1:
            # Example using the DenseCRF2D code
            d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

            # 得到一元勢函數 unary potential
            U = unary_from_labels(labels, n_labels, gt_prob=0.5, zero_unsure=False)
            d.setUnaryEnergy(U)
            # 建立二元勢函數 pairwise potential 二元势就引入了邻域像素对当前像素的影响，所以需要同时考虑像素的位置和其观测值
            # sdims = (3, 3)  # 邻域的平滑度
            # schan = (0.01,)  # 颜色通道的平滑度

            pairwise_energy = create_pairwise_bilateral(sdims=(1,1), schan=(0.01,), img=original_image, chdim=2)
            d.addPairwiseEnergy(pairwise_energy, compat=10)

            Q = d.inference(5)
            final = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

            final = colorize[final,:]
            print(final)
            cv2.imwrite(output_image, final)
        elif predict_image is not None:
            final = predict_image
            cv2.imwrite(output_image, predict_image)
            print("only one class")
        else:
            print(predict_image)
            final = predict_image
            print("no class")

        return final






if __name__ == '__main__':
    for i in pl.Path('./Result/PCV_1011/').iterdir():
        for j in i.iterdir():
            for k in j.iterdir():
                if "PCV_1011_otsu_bil_clahe_42_aug_CC" in str(k):
                    model_path = str(k).replace('Result','Model')+ '_1'
                else:
                    model_path = str(k).replace('Result','Model')

                if not os.path.exists(k/'postcrf'):
                    os.makedirs(k/'postcrf')
                postprocess= postprocessing(k/'images/',k/'results/',k/'postcrf/')

                postprocess.postprocessing(model_path)
                print("finish")